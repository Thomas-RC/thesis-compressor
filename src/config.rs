#[derive(Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub seq_len: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub ffn_mult: usize,
}

impl Config {
    pub fn small() -> Self {
        Self {
            vocab_size: 256,
            seq_len: 256,
            d_model: 128,
            n_layers: 2,
            n_heads: 4,
            ffn_mult: 4,
        }
    }

    pub fn medium() -> Self {
        Self {
            vocab_size: 256,
            seq_len: 512,
            d_model: 256,
            n_layers: 4,
            n_heads: 4,
            ffn_mult: 4,
        }
    }

    pub fn large() -> Self {
        Self {
            vocab_size: 256,
            seq_len: 1024,
            d_model: 512,
            n_layers: 6,
            n_heads: 8,
            ffn_mult: 4,
        }
    }

    pub fn xlarge() -> Self {
        Self {
            vocab_size: 256,
            seq_len: 1024,
            d_model: 768,
            n_layers: 8,
            n_heads: 12,
            ffn_mult: 4,
        }
    }

    pub fn xxlarge() -> Self {
        Self {
            vocab_size: 256,
            seq_len: 1024,
            d_model: 1024,
            n_layers: 8,
            n_heads: 16,
            ffn_mult: 4,
        }
    }

    pub fn xxxlarge() -> Self {
        Self {
            vocab_size: 256,
            seq_len: 1024,
            d_model: 1024,
            n_layers: 16,
            n_heads: 16,
            ffn_mult: 4,
        }
    }

    pub fn from_preset(name: &str) -> anyhow::Result<Self> {
        match name {
            "small" => Ok(Self::small()),
            "medium" => Ok(Self::medium()),
            "large" => Ok(Self::large()),
            "xlarge" => Ok(Self::xlarge()),
            "xxlarge" => Ok(Self::xxlarge()),
            "xxxlarge" => Ok(Self::xxxlarge()),
            other => anyhow::bail!(
                "nieznany preset: {other} (oczekiwano small|medium|large|xlarge|xxlarge|xxxlarge)"
            ),
        }
    }

    pub fn head_dim(&self) -> usize {
        assert!(
            self.d_model % self.n_heads == 0,
            "d_model ({}) must be divisible by n_heads ({})",
            self.d_model,
            self.n_heads
        );
        self.d_model / self.n_heads
    }

    pub fn param_count_estimate(&self) -> usize {
        let tok = self.vocab_size * self.d_model;
        let pos = self.seq_len * self.d_model;
        let per_block = 4 * self.d_model * self.d_model
            + 2 * self.d_model * self.d_model * self.ffn_mult
            + 4 * self.d_model;
        let head = self.d_model * self.vocab_size;
        tok + pos + self.n_layers * per_block + head
    }
}
