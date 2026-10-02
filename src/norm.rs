// Differentiable normalisation layers.
//
// `candle_nn::LayerNorm` / `candle_nn::RmsNorm` dispatch contiguous inputs to
// fused kernels built with `apply_op*_no_bwd`, which silently cut the autograd
// graph: no gradient reaches the input nor the weight/bias. These wrappers keep
// the same parameter layout (`weight`, `bias`) as candle-nn, so checkpoints stay
// compatible, but always use the composed (differentiable) implementation.

use candle_core::{Result, Tensor};
use candle_nn::{Init, Module, VarBuilder};

pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f32,
}

pub fn layer_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<LayerNorm> {
    let weight = vb.get_with_hints(size, "weight", Init::Const(1.0))?;
    let bias = vb.get_with_hints(size, "bias", Init::Const(0.0))?;
    Ok(LayerNorm {
        weight,
        bias,
        eps: eps as f32,
    })
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::layer_norm_slow(x, &self.weight, &self.bias, self.eps)
    }
}

pub struct RmsNorm {
    weight: Tensor,
    eps: f32,
}

pub fn rms_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<RmsNorm> {
    let weight = vb.get_with_hints(size, "weight", Init::Const(1.0))?;
    Ok(RmsNorm {
        weight,
        eps: eps as f32,
    })
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm_slow(x, &self.weight, self.eps)
    }
}
