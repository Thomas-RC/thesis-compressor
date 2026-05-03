use anyhow::{Context, Result, ensure};
use candle_core::{Device, Tensor};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::fs;
use std::path::Path;

pub const ENWIK8_TOTAL: usize = 100_000_000;
pub const TRAIN_BYTES: usize = 90_000_000;
pub const VAL_BYTES: usize = 5_000_000;
pub const TEST_BYTES: usize = 5_000_000;

pub struct Splits {
    pub train: Vec<u8>,
    pub val: Vec<u8>,
    pub test: Vec<u8>,
}

pub fn load_enwik8<P: AsRef<Path>>(path: P) -> Result<Splits> {
    let path = path.as_ref();
    let bytes = fs::read(path)
        .with_context(|| format!("nie udało się wczytać {}", path.display()))?;
    ensure!(
        bytes.len() >= ENWIK8_TOTAL,
        "plik {} ma {} B, oczekiwano >= {} B (10^8)",
        path.display(),
        bytes.len(),
        ENWIK8_TOTAL
    );
    let bytes = &bytes[..ENWIK8_TOTAL];
    let train = bytes[..TRAIN_BYTES].to_vec();
    let val = bytes[TRAIN_BYTES..TRAIN_BYTES + VAL_BYTES].to_vec();
    let test = bytes[TRAIN_BYTES + VAL_BYTES..].to_vec();
    Ok(Splits { train, val, test })
}

pub struct BatchSampler<'a> {
    data: &'a [u8],
    seq_len: usize,
    batch_size: usize,
    rng: StdRng,
}

impl<'a> BatchSampler<'a> {
    pub fn new(data: &'a [u8], seq_len: usize, batch_size: usize, seed: u64) -> Self {
        assert!(
            data.len() > seq_len + 1,
            "dane krótsze niż seq_len+1 ({} <= {})",
            data.len(),
            seq_len + 1
        );
        Self {
            data,
            seq_len,
            batch_size,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn sample(&mut self, device: &Device) -> Result<(Tensor, Tensor)> {
        let bs = self.batch_size;
        let l = self.seq_len;
        let mut input_buf: Vec<u32> = Vec::with_capacity(bs * l);
        let mut target_buf: Vec<u32> = Vec::with_capacity(bs * l);
        let max_start = self.data.len() - l - 1;
        for _ in 0..bs {
            let start = self.rng.random_range(0..=max_start);
            let chunk = &self.data[start..start + l + 1];
            input_buf.extend(chunk[..l].iter().map(|&b| b as u32));
            target_buf.extend(chunk[1..].iter().map(|&b| b as u32));
        }
        let input = Tensor::from_vec(input_buf, (bs, l), device)?;
        let target = Tensor::from_vec(target_buf, (bs, l), device)?;
        Ok((input, target))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_is_input_shifted_by_one() {
        let data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        let mut s = BatchSampler::new(&data, 16, 4, 0);
        let (inp, tgt) = s.sample(&Device::Cpu).unwrap();
        let inp: Vec<Vec<u32>> = inp.to_vec2().unwrap();
        let tgt: Vec<Vec<u32>> = tgt.to_vec2().unwrap();
        for b in 0..inp.len() {
            for t in 0..inp[b].len() - 1 {
                assert_eq!(
                    inp[b][t + 1],
                    tgt[b][t],
                    "shift mismatch at b={b}, t={t}"
                );
            }
        }
    }

    #[test]
    fn deterministic_with_seed() {
        let data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        let mut a = BatchSampler::new(&data, 16, 4, 42);
        let mut b = BatchSampler::new(&data, 16, 4, 42);
        let (ia, _) = a.sample(&Device::Cpu).unwrap();
        let (ib, _) = b.sample(&Device::Cpu).unwrap();
        assert_eq!(
            ia.to_vec2::<u32>().unwrap(),
            ib.to_vec2::<u32>().unwrap()
        );
    }
}
