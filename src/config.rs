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
    pub fn mvp() -> Self {
        Self {
            vocab_size: 256,
            seq_len: 256,
            d_model: 128,
            n_layers: 2,
            n_heads: 4,
            ffn_mult: 4,
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
