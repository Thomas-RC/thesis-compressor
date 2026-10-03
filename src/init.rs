// Parameter initialisation schemes.
//
// `Candle` keeps candle-nn defaults: Kaiming-normal weights, uniform biases and
// N(0, 1) embeddings. `Gpt2` follows GPT-2: N(0, 0.02) weights and embeddings,
// zero biases, and residual output projections (attn.out, ffn.fc2) scaled by
// 1/sqrt(2 * n_layers) so the residual stream variance does not grow with depth.

use candle_core::Result;
use candle_nn::{Embedding, Init, Linear, VarBuilder};

use crate::config::Config;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum InitScheme {
    Candle,
    Gpt2,
}

impl InitScheme {
    pub fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "candle" => Ok(InitScheme::Candle),
            "gpt2" => Ok(InitScheme::Gpt2),
            other => anyhow::bail!("nieznany schemat inicjalizacji: {other} (oczekiwano candle|gpt2)"),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            InitScheme::Candle => "candle",
            InitScheme::Gpt2 => "gpt2",
        }
    }
}

const GPT2_STD: f64 = 0.02;

/// Linear layer initialised according to `cfg.init`. `residual` marks output
/// projections that write into the residual stream.
pub fn linear(
    cfg: &Config,
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    residual: bool,
    vb: VarBuilder,
) -> Result<Linear> {
    match cfg.init {
        InitScheme::Candle => candle_nn::linear_b(in_dim, out_dim, bias, vb),
        InitScheme::Gpt2 => {
            let stdev = if residual {
                GPT2_STD / (2.0 * cfg.n_layers as f64).sqrt()
            } else {
                GPT2_STD
            };
            let ws = vb.get_with_hints((out_dim, in_dim), "weight", Init::Randn { mean: 0.0, stdev })?;
            let bs = if bias {
                Some(vb.get_with_hints(out_dim, "bias", Init::Const(0.0))?)
            } else {
                None
            };
            Ok(Linear::new(ws, bs))
        }
    }
}

pub fn embedding(cfg: &Config, n: usize, dim: usize, vb: VarBuilder) -> Result<Embedding> {
    match cfg.init {
        InitScheme::Candle => candle_nn::embedding(n, dim, vb),
        InitScheme::Gpt2 => {
            let ws = vb.get_with_hints(
                (n, dim),
                "weight",
                Init::Randn {
                    mean: 0.0,
                    stdev: GPT2_STD,
                },
            )?;
            Ok(Embedding::new(ws, dim))
        }
    }
}
