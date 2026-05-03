pub mod config;
pub mod data;
pub mod model;
pub mod model_llama;
pub mod train;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Arch {
    Baseline,
    Llama,
}

impl Arch {
    pub fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "baseline" => Ok(Arch::Baseline),
            "llama" => Ok(Arch::Llama),
            other => anyhow::bail!("nieznana architektura: {other} (oczekiwano baseline|llama)"),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Arch::Baseline => "baseline",
            Arch::Llama => "llama",
        }
    }
}
