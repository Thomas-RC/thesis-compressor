// Neural lossless compressor: the Llama-style model predicts the next byte and
// the arithmetic coder turns those predictions into bits.
//
// Determinism: encoder and decoder run the same sequence of tensor operations
// on identical inputs (same shapes, same device), so the model yields
// bit-identical logits on both sides; logits become integer frequencies on
// the CPU. A file therefore decodes only with the same checkpoint, device
// kind and binary that produced it (checked via the header and a checksum).
//
// Layout: the input is split into `streams` equal segments coded in lockstep,
// one byte per stream per step, batched on the device. Each stream keeps a KV
// cache of at most `seq_len` positions; when it is full it is reset and
// re-filled with the last `refill` bytes, so every prediction sees between
// `refill` and `seq_len - 1` bytes of context. The first byte of each stream
// is coded with a uniform distribution.

use anyhow::{Context, Result, bail, ensure};
use candle_core::{Device, Tensor};
use std::time::Instant;

use crate::coder::{CumFreq, Decoder, Encoder, N_SYMBOLS, quantize_logits, uniform_cum};
use crate::model_llama::LlamaModel;

const MAGIC: &[u8; 4] = b"TCZ\x01";
const ARCH_LLAMA: u8 = 1;

pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu = 0,
    Cuda = 1,
}

impl DeviceKind {
    pub fn of(device: &Device) -> Result<Self> {
        match device {
            Device::Cpu => Ok(DeviceKind::Cpu),
            Device::Cuda(_) => Ok(DeviceKind::Cuda),
            other => bail!("nieobsługiwane urządzenie: {other:?}"),
        }
    }

    fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(DeviceKind::Cpu),
            1 => Ok(DeviceKind::Cuda),
            other => bail!("nieznany typ urządzenia w nagłówku: {other}"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Header {
    pub preset: String,
    pub device: DeviceKind,
    pub model_hash: u64,
    pub orig_len: u64,
    pub streams: u32,
    pub refill: u32,
    pub checksum: u64,
}

impl Header {
    fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(MAGIC);
        out.push(ARCH_LLAMA);
        out.push(self.preset.len() as u8);
        out.extend_from_slice(self.preset.as_bytes());
        out.push(self.device as u8);
        out.extend_from_slice(&self.model_hash.to_le_bytes());
        out.extend_from_slice(&self.orig_len.to_le_bytes());
        out.extend_from_slice(&self.streams.to_le_bytes());
        out.extend_from_slice(&self.refill.to_le_bytes());
        out.extend_from_slice(&self.checksum.to_le_bytes());
    }

    /// Parses the header and returns it with the remaining payload.
    pub fn read(bytes: &[u8]) -> Result<(Header, &[u8])> {
        let mut r = bytes;
        let mut take = |n: usize| -> Result<&[u8]> {
            ensure!(r.len() >= n, "plik ucięty: niekompletny nagłówek");
            let (head, rest) = r.split_at(n);
            r = rest;
            Ok(head)
        };
        ensure!(take(4)? == MAGIC, "to nie jest plik TCZ (zły magic)");
        ensure!(take(1)?[0] == ARCH_LLAMA, "nieobsługiwana architektura w nagłówku");
        let preset_len = take(1)?[0] as usize;
        let preset = String::from_utf8(take(preset_len)?.to_vec()).context("preset nie jest UTF-8")?;
        let device = DeviceKind::from_u8(take(1)?[0])?;
        let u64_at = |b: &[u8]| u64::from_le_bytes(b.try_into().unwrap());
        let u32_at = |b: &[u8]| u32::from_le_bytes(b.try_into().unwrap());
        let model_hash = u64_at(take(8)?);
        let orig_len = u64_at(take(8)?);
        let streams = u32_at(take(4)?);
        let refill = u32_at(take(4)?);
        let checksum = u64_at(take(8)?);
        let header = Header {
            preset,
            device,
            model_hash,
            orig_len,
            streams,
            refill,
            checksum,
        };
        Ok((header, r))
    }
}

#[derive(Clone, Debug)]
pub struct Options {
    pub streams: usize,
    /// Bytes re-fed into a reset KV cache; defaults to seq_len / 2 when None.
    pub refill: Option<usize>,
    pub progress: bool,
}

/// Lengths of the equal segments the input is split into (no empty segments).
fn segment_lengths(total: usize, streams: usize) -> Vec<usize> {
    if total == 0 {
        return Vec::new();
    }
    let seg = total.div_ceil(streams.clamp(1, total));
    let n = total.div_ceil(seg);
    (0..n).map(|s| seg.min(total - s * seg)).collect()
}

/// Runs the model over all streams in lockstep. For every (stream, position)
/// `code` receives the frequency table and returns the actual byte: the
/// encoder reads it from the input and encodes it, the decoder decodes it.
fn code_streams(
    model: &LlamaModel,
    seg_lens: &[usize],
    refill: usize,
    progress: bool,
    mut code: impl FnMut(usize, usize, &CumFreq) -> u8,
) -> Result<Vec<Vec<u8>>> {
    let n_streams = seg_lens.len();
    let mut hist: Vec<Vec<u8>> = seg_lens.iter().map(|&l| Vec::with_capacity(l)).collect();
    let Some(&steps) = seg_lens.first() else {
        return Ok(hist);
    };
    let seq_len = model.config().seq_len;
    ensure!(refill >= 1 && refill < seq_len, "refill musi być w zakresie 1..{seq_len}");
    let device = model.device();
    let mut cache = model.new_cache();
    let mut pos = 0usize;
    let mut logits: Option<Vec<f32>> = None;
    let mut cum: CumFreq = [0; N_SYMBOLS + 1];
    let t0 = Instant::now();
    let report_every = (steps / 50).max(1);

    for i in 0..steps {
        let mut step_bytes = vec![0u32; n_streams];
        for s in 0..n_streams {
            if i >= seg_lens[s] {
                continue;
            }
            match &logits {
                None => uniform_cum(&mut cum),
                Some(l) => quantize_logits(&l[s * N_SYMBOLS..(s + 1) * N_SYMBOLS], &mut cum),
            }
            let byte = code(s, i, &cum);
            hist[s].push(byte);
            step_bytes[s] = byte as u32;
        }
        if i + 1 == steps {
            break;
        }

        let out = if pos == seq_len {
            // Cache full: restart it from the last `refill` bytes of each stream.
            cache.iter_mut().for_each(|c| c.reset());
            let mut buf = Vec::with_capacity(n_streams * refill);
            for h in &hist {
                buf.extend((i + 1 - refill..=i).map(|j| h.get(j).copied().unwrap_or(0) as u32));
            }
            let x = Tensor::from_vec(buf, (n_streams, refill), device)?;
            pos = refill;
            model.forward_cached(&x, 0, &mut cache)?.narrow(1, refill - 1, 1)?
        } else {
            let x = Tensor::from_vec(step_bytes, (n_streams, 1), device)?;
            let out = model.forward_cached(&x, pos, &mut cache)?;
            pos += 1;
            out
        };
        logits = Some(out.flatten_all()?.to_vec1::<f32>()?);

        if progress && (i + 1) % report_every == 0 {
            let done: usize = seg_lens.iter().map(|&l| l.min(i + 1)).sum();
            let total: usize = seg_lens.iter().sum();
            let el = t0.elapsed().as_secs_f64();
            eprintln!(
                "  [{:>5.1}%] {done}/{total} B | {:.1} kB/s | {el:.0}s",
                100.0 * done as f64 / total as f64,
                done as f64 / el / 1e3
            );
        }
    }
    Ok(hist)
}

pub fn compress(
    model: &LlamaModel,
    preset: &str,
    model_hash: u64,
    data: &[u8],
    opts: &Options,
) -> Result<Vec<u8>> {
    let refill = opts.refill.unwrap_or(model.config().seq_len / 2);
    let seg_lens = segment_lengths(data.len(), opts.streams);
    let seg = seg_lens.first().copied().unwrap_or(0);
    let mut enc = Encoder::new();
    code_streams(model, &seg_lens, refill, opts.progress, |s, i, cum| {
        let byte = data[s * seg + i];
        enc.encode(byte, cum);
        byte
    })?;
    let header = Header {
        preset: preset.to_string(),
        device: DeviceKind::of(model.device())?,
        model_hash,
        orig_len: data.len() as u64,
        streams: seg_lens.len() as u32,
        refill: refill as u32,
        checksum: fnv1a64(data),
    };
    let mut out = Vec::new();
    header.write(&mut out);
    out.extend_from_slice(&enc.finish());
    Ok(out)
}

/// Decodes `payload`; the caller must load the model named by `header`.
pub fn decompress(
    model: &LlamaModel,
    header: &Header,
    model_hash: u64,
    payload: &[u8],
    progress: bool,
) -> Result<Vec<u8>> {
    ensure!(
        model_hash == header.model_hash,
        "inny checkpoint niż przy kompresji (hash {model_hash:016x} != {:016x})",
        header.model_hash
    );
    let device = DeviceKind::of(model.device())?;
    ensure!(
        device == header.device,
        "plik skompresowano na {:?}, a dekompresja działa na {device:?}: wyniki modelu nie byłyby identyczne",
        header.device
    );
    let seg_lens = segment_lengths(header.orig_len as usize, header.streams as usize);
    ensure!(
        seg_lens.len() == header.streams as usize,
        "niespójny nagłówek: liczba strumieni"
    );
    let mut dec = Decoder::new(payload);
    let hist = code_streams(model, &seg_lens, header.refill as usize, progress, |_, _, cum| {
        dec.decode(cum)
    })?;
    let data: Vec<u8> = hist.concat();
    ensure!(
        fnv1a64(&data) == header.checksum,
        "suma kontrolna się nie zgadza: dane zdekodowane niepoprawnie"
    );
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::init::InitScheme;
    use candle_core::DType;
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn segments_cover_input_without_empty_streams() {
        for (total, streams) in [(0, 4), (1, 4), (9, 4), (10, 4), (1000, 7), (5, 100)] {
            let lens = segment_lengths(total, streams);
            assert_eq!(lens.iter().sum::<usize>(), total);
            assert!(lens.iter().all(|&l| l > 0));
            assert!(lens.windows(2).all(|w| w[0] >= w[1]));
        }
    }

    #[test]
    fn roundtrip_with_untrained_model() {
        let device = Device::Cpu;
        let cfg = Config {
            vocab_size: 256,
            seq_len: 16,
            d_model: 32,
            n_layers: 2,
            n_heads: 4,
            ffn_mult: 4,
            init: InitScheme::Gpt2,
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = LlamaModel::new(cfg, vb).unwrap();
        let data: Vec<u8> = b"<page><title>Test</title><text>Lorem ipsum, dolor sit amet. "
            .iter()
            .cycle()
            .take(1000)
            .copied()
            .collect();
        for streams in [1, 3, 8] {
            let opts = Options {
                streams,
                refill: Some(5),
                progress: false,
            };
            let packed = compress(&model, "test", 42, &data, &opts).unwrap();
            let (header, payload) = Header::read(&packed).unwrap();
            let unpacked = decompress(&model, &header, 42, payload, false).unwrap();
            assert_eq!(unpacked, data, "roundtrip failed for streams={streams}");
        }
    }
}
