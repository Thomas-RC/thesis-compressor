pub mod config;
pub mod data;
pub mod init;
pub mod model;
pub mod model_llama;
pub mod model_rms;
pub mod model_rope;
pub mod model_swiglu;
pub mod norm;
pub mod softmax;
pub mod train;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Arch {
    /// Reference: LayerNorm + learned absolute pos + GELU MLP, Linear with bias.
    Baseline,
    /// Full Llama-style: RMSNorm + RoPE + SwiGLU, Linear no bias.
    Llama,
    /// Single-component ablation: only RMSNorm replaces LayerNorm; rest = Baseline.
    LlamaRms,
    /// Single-component ablation: only RoPE replaces learned pos; rest = Baseline.
    LlamaRope,
    /// Single-component ablation: only SwiGLU replaces GELU MLP; rest = Baseline.
    LlamaSwiglu,
}

impl Arch {
    pub fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "baseline" => Ok(Arch::Baseline),
            "llama" => Ok(Arch::Llama),
            "llama-rms" | "llama_rms" => Ok(Arch::LlamaRms),
            "llama-rope" | "llama_rope" => Ok(Arch::LlamaRope),
            "llama-swiglu" | "llama_swiglu" => Ok(Arch::LlamaSwiglu),
            other => anyhow::bail!(
                "nieznana architektura: {other} (oczekiwano baseline|llama|llama-rms|llama-rope|llama-swiglu)"
            ),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Arch::Baseline => "baseline",
            Arch::Llama => "llama",
            Arch::LlamaRms => "llama-rms",
            Arch::LlamaRope => "llama-rope",
            Arch::LlamaSwiglu => "llama-swiglu",
        }
    }
}
