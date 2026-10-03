use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::kv_cache::KvCache;
use candle_nn::{Embedding, Linear, Module, VarBuilder};

use crate::config::Config;
use crate::init::{embedding, linear};
use crate::norm::{RmsNorm, rms_norm};

fn build_causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            data[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    Ok(Tensor::from_vec(data, (seq_len, seq_len), device)?)
}

/// Llama-style hidden size: (8/3) * d_model rounded up to a multiple of 32,
/// so that SwiGLU FFN has compute roughly comparable to a 4×d_model GELU FFN.
fn ffn_hidden(d_model: usize) -> usize {
    let raw = 8 * d_model / 3;
    raw.div_ceil(32) * 32
}

pub struct Rope {
    pub cos: Tensor,
    pub sin: Tensor,
}

impl Rope {
    pub fn new(max_seq_len: usize, head_dim: usize, device: &Device) -> Result<Self> {
        anyhow::ensure!(head_dim % 2 == 0, "RoPE requires even head_dim, got {head_dim}");
        let theta: f32 = 10_000.0;
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, half), device)?;
        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::from_vec(positions, (max_seq_len, 1), device)?;
        let freqs = positions.broadcast_mul(&inv_freq)?;
        let cos = freqs.cos()?.contiguous()?;
        let sin = freqs.sin()?.contiguous()?;
        Ok(Self { cos, sin })
    }
}

struct LlamaAttention {
    qkv: Linear,
    out: Linear,
    n_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl LlamaAttention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let qkv = linear(cfg, cfg.d_model, 3 * cfg.d_model, false, false, vb.pp("qkv"))?;
        let out = linear(cfg, cfg.d_model, cfg.d_model, false, true, vb.pp("out"))?;
        let head_dim = cfg.head_dim();
        Ok(Self {
            qkv,
            out,
            n_heads: cfg.n_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    fn forward(&self, x: &Tensor, rope: &Rope, mask: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let qkv = self.qkv.forward(x)?;
        let chunks = qkv.chunk(3, 2)?;
        let to_heads = |t_in: &Tensor| -> Result<Tensor> {
            Ok(t_in
                .reshape((b, t, self.n_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?)
        };
        let q = to_heads(&chunks[0])?;
        let k = to_heads(&chunks[1])?;
        let v = to_heads(&chunks[2])?;

        // RoPE on Q and K (rope_slow because rope/sdpa use no_bwd CustomOps).
        let q = candle_nn::rotary_emb::rope_slow(&q, &rope.cos, &rope.sin)?;
        let k = candle_nn::rotary_emb::rope_slow(&k, &rope.cos, &rope.sin)?;

        // Scale Q instead of the (B, H, T, T) scores: one fewer large tensor kept for backward.
        let q = (q * self.scale)?;
        let scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;

        let mask = mask
            .narrow(0, 0, t)?
            .narrow(1, 0, t)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let scores = scores.broadcast_add(&mask)?;
        let weights = crate::softmax::softmax_last_dim(&scores)?;

        let out = weights.matmul(&v)?;
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, self.n_heads * self.head_dim))?;
        Ok(self.out.forward(&out)?)
    }

    /// Inference-only attention over a KV cache: `x` holds `t` new positions
    /// starting at `offset`; their keys and values are appended to `cache`.
    fn forward_cached(
        &self,
        x: &Tensor,
        rope: &Rope,
        offset: usize,
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let qkv = self.qkv.forward(x)?;
        let chunks = qkv.chunk(3, 2)?;
        let to_heads = |t_in: &Tensor| -> Result<Tensor> {
            Ok(t_in
                .reshape((b, t, self.n_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?)
        };
        let q = to_heads(&chunks[0])?;
        let k = to_heads(&chunks[1])?;
        let v = to_heads(&chunks[2])?;

        let cos = rope.cos.narrow(0, offset, t)?;
        let sin = rope.sin.narrow(0, offset, t)?;
        let q = candle_nn::rotary_emb::rope_slow(&q, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope_slow(&k, &cos, &sin)?;
        let q = (q * self.scale)?;

        let (k, v) = cache.append(&k.contiguous()?, &v)?;
        let scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let scores = if t > 1 {
            // New position i (absolute offset + i) may attend to keys 0..=offset + i.
            let total = offset + t;
            let mut data = vec![0f32; t * total];
            for i in 0..t {
                for j in (offset + i + 1)..total {
                    data[i * total + j] = f32::NEG_INFINITY;
                }
            }
            let mask = Tensor::from_vec(data, (1, 1, t, total), x.device())?;
            scores.broadcast_add(&mask)?
        } else {
            scores
        };
        let weights = crate::softmax::softmax_last_dim(&scores)?;

        let out = weights.matmul(&v.contiguous()?)?;
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, self.n_heads * self.head_dim))?;
        Ok(self.out.forward(&out)?)
    }
}

struct LlamaFFN {
    fc1: Linear, // d_model -> 2 * hidden  (gate + up combined for swiglu)
    fc2: Linear, // hidden -> d_model
}

impl LlamaFFN {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden = ffn_hidden(cfg.d_model);
        let fc1 = linear(cfg, cfg.d_model, 2 * hidden, false, false, vb.pp("fc1"))?;
        let fc2 = linear(cfg, hidden, cfg.d_model, false, true, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }
}

impl Module for LlamaFFN {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let h = self.fc1.forward(x)?;
        let h = candle_nn::ops::swiglu(&h)?;
        self.fc2.forward(&h)
    }
}

struct LlamaBlock {
    attn: LlamaAttention,
    ffn: LlamaFFN,
    norm1: RmsNorm,
    norm2: RmsNorm,
}

impl LlamaBlock {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let attn = LlamaAttention::new(cfg, vb.pp("attn"))?;
        let ffn = LlamaFFN::new(cfg, vb.pp("ffn"))?;
        let norm1 = rms_norm(cfg.d_model, 1e-5, vb.pp("norm1"))?;
        let norm2 = rms_norm(cfg.d_model, 1e-5, vb.pp("norm2"))?;
        Ok(Self { attn, ffn, norm1, norm2 })
    }

    fn forward(&self, x: &Tensor, rope: &Rope, mask: &Tensor) -> Result<Tensor> {
        let h = self.attn.forward(&self.norm1.forward(x)?, rope, mask)?;
        let x = (x + h)?;
        let h = self.ffn.forward(&self.norm2.forward(&x)?)?;
        Ok((&x + h)?)
    }

    fn forward_cached(
        &self,
        x: &Tensor,
        rope: &Rope,
        offset: usize,
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        let h = self
            .attn
            .forward_cached(&self.norm1.forward(x)?, rope, offset, cache)?;
        let x = (x + h)?;
        let h = self.ffn.forward(&self.norm2.forward(&x)?)?;
        Ok((&x + h)?)
    }
}

pub struct LlamaModel {
    tok_emb: Embedding,
    blocks: Vec<LlamaBlock>,
    norm_f: RmsNorm,
    head: Linear,
    rope: Rope,
    mask: Tensor,
    cfg: Config,
    device: Device,
}

impl LlamaModel {
    pub fn new(cfg: Config, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let tok_emb = embedding(&cfg, cfg.vocab_size, cfg.d_model, vb.pp("tok_emb"))?;
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            blocks.push(LlamaBlock::new(&cfg, vb.pp(&format!("block_{i}")))?);
        }
        let norm_f = rms_norm(cfg.d_model, 1e-5, vb.pp("norm_f"))?;
        let head = linear(&cfg, cfg.d_model, cfg.vocab_size, false, false, vb.pp("head"))?;
        let rope = Rope::new(cfg.seq_len, cfg.head_dim(), &device)?;
        let mask = build_causal_mask(cfg.seq_len, &device)?;
        Ok(Self {
            tok_emb,
            blocks,
            norm_f,
            head,
            rope,
            mask,
            cfg,
            device,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, t) = x.dims2()?;
        anyhow::ensure!(
            t <= self.cfg.seq_len,
            "input length {t} > seq_len {}",
            self.cfg.seq_len
        );
        anyhow::ensure!(
            x.device().same_device(&self.device),
            "input device differs from model device"
        );
        let mut h = self.tok_emb.forward(x)?;
        for block in &self.blocks {
            h = block.forward(&h, &self.rope, &self.mask)?;
        }
        let h = self.norm_f.forward(&h)?;
        Ok(self.head.forward(&h)?)
    }

    /// One empty KV cache per layer, sized for `seq_len` positions.
    pub fn new_cache(&self) -> Vec<KvCache> {
        (0..self.cfg.n_layers)
            .map(|_| KvCache::new(2, self.cfg.seq_len))
            .collect()
    }

    /// Inference forward over a KV cache: `x` (B, t) holds the positions
    /// `offset..offset + t`; returns logits for those positions (B, t, vocab).
    pub fn forward_cached(
        &self,
        x: &Tensor,
        offset: usize,
        cache: &mut [KvCache],
    ) -> Result<Tensor> {
        let (_b, t) = x.dims2()?;
        anyhow::ensure!(
            offset + t <= self.cfg.seq_len,
            "positions {offset}..{} exceed seq_len {}",
            offset + t,
            self.cfg.seq_len
        );
        anyhow::ensure!(cache.len() == self.blocks.len(), "one KV cache per layer expected");
        let mut h = self.tok_emb.forward(x)?;
        for (block, c) in self.blocks.iter().zip(cache.iter_mut()) {
            h = block.forward_cached(&h, &self.rope, offset, c)?;
        }
        let h = self.norm_f.forward(&h)?;
        Ok(self.head.forward(&h)?)
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn param_count_estimate(cfg: &Config) -> usize {
        let tok = cfg.vocab_size * cfg.d_model;
        let hidden = ffn_hidden(cfg.d_model);
        let per_block = 4 * cfg.d_model * cfg.d_model // qkv + out (no bias)
            + cfg.d_model * 2 * hidden                 // fc1 (gate+up)
            + hidden * cfg.d_model                     // fc2
            + 2 * cfg.d_model; // 2× RmsNorm scales
        let head = cfg.d_model * cfg.vocab_size;
        let norm_f = cfg.d_model;
        tok + cfg.n_layers * per_block + head + norm_f
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::init::InitScheme;
    use candle_core::DType;
    use candle_nn::VarMap;

    #[test]
    fn cached_forward_matches_full_forward() {
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
        let model = LlamaModel::new(cfg.clone(), vb).unwrap();
        let tokens: Vec<u32> = (0..2 * 16u32).map(|i| (i * 53 + 7) % 256).collect();
        let x = Tensor::from_vec(tokens, (2, 16), &device).unwrap();
        let full = model.forward(&x).unwrap();

        // Twice, to check that a reset cache behaves like a fresh one.
        let mut cache = model.new_cache();
        for _ in 0..2 {
            cache.iter_mut().for_each(|c| c.reset());
            let prefill = 6;
            let mut parts = vec![model.forward_cached(&x.narrow(1, 0, prefill).unwrap(), 0, &mut cache).unwrap()];
            for pos in prefill..16 {
                parts.push(model.forward_cached(&x.narrow(1, pos, 1).unwrap(), pos, &mut cache).unwrap());
            }
            let cached = Tensor::cat(&parts, 1).unwrap();
            let diff: f32 = (&cached - &full)
                .unwrap()
                .abs()
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar()
                .unwrap();
            assert!(diff < 1e-4, "cached logits differ from full forward by {diff}");
        }
    }
}
