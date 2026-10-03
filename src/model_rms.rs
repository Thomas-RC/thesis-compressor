// Single-component ablation of Llama: replaces LayerNorm with RmsNorm.
// Everything else identical to baseline model.rs (learned absolute pos
// embeddings, GELU MLP feed-forward, Linear with biases).

use anyhow::Result;
use candle_core::{Device, Tensor};
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

pub struct MultiHeadAttention {
    qkv: Linear,
    out: Linear,
    n_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl MultiHeadAttention {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let qkv = linear(cfg, cfg.d_model, 3 * cfg.d_model, true, false, vb.pp("qkv"))?;
        let out = linear(cfg, cfg.d_model, cfg.d_model, true, true, vb.pp("out"))?;
        let head_dim = cfg.head_dim();
        Ok(Self {
            qkv,
            out,
            n_heads: cfg.n_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    pub fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
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
}

pub struct FeedForward {
    fc1: Linear,
    fc2: Linear,
}

impl FeedForward {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.d_model * cfg.ffn_mult;
        let fc1 = linear(cfg, cfg.d_model, hidden, true, false, vb.pp("fc1"))?;
        let fc2 = linear(cfg, hidden, cfg.d_model, true, true, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }
}

impl Module for FeedForward {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = x.gelu()?;
        self.fc2.forward(&x)
    }
}

pub struct Block {
    attn: MultiHeadAttention,
    ffn: FeedForward,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl Block {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let attn = MultiHeadAttention::new(cfg, vb.pp("attn"))?;
        let ffn = FeedForward::new(cfg, vb.pp("ffn"))?;
        let ln1 = rms_norm(cfg.d_model, 1e-5, vb.pp("ln1"))?;
        let ln2 = rms_norm(cfg.d_model, 1e-5, vb.pp("ln2"))?;
        Ok(Self { attn, ffn, ln1, ln2 })
    }

    pub fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let h = self.attn.forward(&self.ln1.forward(x)?, mask)?;
        let x = (x + h)?;
        let h = self.ffn.forward(&self.ln2.forward(&x)?)?;
        Ok((&x + h)?)
    }
}

pub struct Model {
    tok_emb: Embedding,
    pos_emb: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    head: Linear,
    mask: Tensor,
    cfg: Config,
    device: Device,
}

impl Model {
    pub fn new(cfg: Config, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let tok_emb = embedding(&cfg, cfg.vocab_size, cfg.d_model, vb.pp("tok_emb"))?;
        let pos_emb = embedding(&cfg, cfg.seq_len, cfg.d_model, vb.pp("pos_emb"))?;
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            blocks.push(Block::new(&cfg, vb.pp(&format!("block_{i}")))?);
        }
        let ln_f = rms_norm(cfg.d_model, 1e-5, vb.pp("ln_f"))?;
        let head = linear(&cfg, cfg.d_model, cfg.vocab_size, true, false, vb.pp("head"))?;
        let mask = build_causal_mask(cfg.seq_len, &device)?;
        Ok(Self {
            tok_emb,
            pos_emb,
            blocks,
            ln_f,
            head,
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
        let positions = Tensor::arange(0u32, t as u32, &self.device)?;
        let pos = self.pos_emb.forward(&positions)?.unsqueeze(0)?;
        let tok = self.tok_emb.forward(x)?;
        let mut h = tok.broadcast_add(&pos)?;
        for block in &self.blocks {
            h = block.forward(&h, &self.mask)?;
        }
        let h = self.ln_f.forward(&h)?;
        Ok(self.head.forward(&h)?)
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
