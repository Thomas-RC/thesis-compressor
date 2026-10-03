// Binary arithmetic coder (Witten, Neal & Cleary, 1987) with 32-bit precision,
// plus quantisation of model logits into integer cumulative frequencies.
//
// The coder works on integer frequencies only, so encoder and decoder agree
// bit-for-bit as long as they are given identical frequency tables.

const CODE_BITS: u32 = 32;
const TOP: u64 = (1 << CODE_BITS) - 1;
const HALF: u64 = 1 << (CODE_BITS - 1);
const QUARTER: u64 = 1 << (CODE_BITS - 2);

/// Total of every frequency table produced by [`quantize_logits`]. Must stay
/// well below 2^(CODE_BITS - 2) so every symbol keeps a non-empty interval.
pub const FREQ_TOTAL: u32 = 1 << 24;
pub const N_SYMBOLS: usize = 256;

/// Cumulative frequencies: symbol `s` owns `[cum[s], cum[s + 1])`, `cum[256] == total`.
pub type CumFreq = [u32; N_SYMBOLS + 1];

/// Turns logits into cumulative frequencies summing to [`FREQ_TOTAL`]. Every
/// symbol gets at least 1, so any byte stays encodable. Pure CPU code in f64:
/// deterministic for identical inputs.
pub fn quantize_logits(logits: &[f32], cum: &mut CumFreq) {
    assert_eq!(logits.len(), N_SYMBOLS);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut exps = [0f64; N_SYMBOLS];
    let mut sum = 0f64;
    for (e, &l) in exps.iter_mut().zip(logits) {
        *e = (l as f64 - max).exp();
        sum += *e;
    }
    let budget = (FREQ_TOTAL - N_SYMBOLS as u32) as f64;
    let mut freqs = [0u32; N_SYMBOLS];
    let mut assigned = 0u32;
    let mut argmax = 0;
    for (i, (f, e)) in freqs.iter_mut().zip(&exps).enumerate() {
        *f = 1 + ((e / sum) * budget) as u32;
        assigned += *f;
        if *e > exps[argmax] {
            argmax = i;
        }
    }
    freqs[argmax] += FREQ_TOTAL - assigned;
    cum[0] = 0;
    for i in 0..N_SYMBOLS {
        cum[i + 1] = cum[i] + freqs[i];
    }
}

/// Uniform table, used for the first byte of a stream (no context yet).
pub fn uniform_cum(cum: &mut CumFreq) {
    let f = FREQ_TOTAL / N_SYMBOLS as u32;
    for (i, c) in cum.iter_mut().enumerate() {
        *c = i as u32 * f;
    }
}

#[derive(Default)]
struct BitWriter {
    bytes: Vec<u8>,
    current: u8,
    n_bits: u8,
}

impl BitWriter {
    fn push(&mut self, bit: bool) {
        self.current = (self.current << 1) | bit as u8;
        self.n_bits += 1;
        if self.n_bits == 8 {
            self.bytes.push(self.current);
            self.current = 0;
            self.n_bits = 0;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.n_bits > 0 {
            self.current <<= 8 - self.n_bits;
            self.bytes.push(self.current);
        }
        self.bytes
    }
}

struct BitReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl BitReader<'_> {
    /// Bits past the end read as zeros (matching the encoder's flush).
    fn next(&mut self) -> u64 {
        let byte = self.pos / 8;
        let bit = if byte < self.bytes.len() {
            (self.bytes[byte] >> (7 - self.pos % 8)) & 1
        } else {
            0
        };
        self.pos += 1;
        bit as u64
    }
}

pub struct Encoder {
    low: u64,
    high: u64,
    pending: u64,
    out: BitWriter,
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Encoder {
    pub fn new() -> Self {
        Self {
            low: 0,
            high: TOP,
            pending: 0,
            out: BitWriter::default(),
        }
    }

    fn emit(&mut self, bit: bool) {
        self.out.push(bit);
        for _ in 0..self.pending {
            self.out.push(!bit);
        }
        self.pending = 0;
    }

    pub fn encode(&mut self, symbol: u8, cum: &CumFreq) {
        let total = cum[N_SYMBOLS] as u64;
        let (lo, hi) = (cum[symbol as usize] as u64, cum[symbol as usize + 1] as u64);
        debug_assert!(lo < hi, "symbol with zero frequency");
        let range = self.high - self.low + 1;
        self.high = self.low + range * hi / total - 1;
        self.low += range * lo / total;
        loop {
            if self.high < HALF {
                self.emit(false);
            } else if self.low >= HALF {
                self.emit(true);
                self.low -= HALF;
                self.high -= HALF;
            } else if self.low >= QUARTER && self.high < 3 * QUARTER {
                self.pending += 1;
                self.low -= QUARTER;
                self.high -= QUARTER;
            } else {
                break;
            }
            self.low <<= 1;
            self.high = (self.high << 1) | 1;
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        self.pending += 1;
        if self.low < QUARTER {
            self.emit(false);
        } else {
            self.emit(true);
        }
        self.out.finish()
    }
}

pub struct Decoder<'a> {
    low: u64,
    high: u64,
    value: u64,
    input: BitReader<'a>,
}

impl<'a> Decoder<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        let mut input = BitReader { bytes, pos: 0 };
        let mut value = 0;
        for _ in 0..CODE_BITS {
            value = (value << 1) | input.next();
        }
        Self {
            low: 0,
            high: TOP,
            value,
            input,
        }
    }

    pub fn decode(&mut self, cum: &CumFreq) -> u8 {
        let total = cum[N_SYMBOLS] as u64;
        let range = self.high - self.low + 1;
        let scaled = ((self.value - self.low + 1) * total - 1) / range;
        // Largest s with cum[s] <= scaled.
        let symbol = cum.partition_point(|&c| c as u64 <= scaled) - 1;
        let (lo, hi) = (cum[symbol] as u64, cum[symbol + 1] as u64);
        self.high = self.low + range * hi / total - 1;
        self.low += range * lo / total;
        loop {
            if self.high < HALF {
            } else if self.low >= HALF {
                self.low -= HALF;
                self.high -= HALF;
                self.value -= HALF;
            } else if self.low >= QUARTER && self.high < 3 * QUARTER {
                self.low -= QUARTER;
                self.high -= QUARTER;
                self.value -= QUARTER;
            } else {
                break;
            }
            self.low <<= 1;
            self.high = (self.high << 1) | 1;
            self.value = (self.value << 1) | self.input.next();
        }
        symbol as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    fn random_logits(rng: &mut StdRng, sharp: f32) -> Vec<f32> {
        (0..N_SYMBOLS).map(|_| rng.random_range(-1.0f32..1.0) * sharp).collect()
    }

    #[test]
    fn quantized_table_is_valid() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut cum = [0u32; N_SYMBOLS + 1];
        for sharp in [0.0, 1.0, 10.0, 100.0, 1000.0] {
            quantize_logits(&random_logits(&mut rng, sharp), &mut cum);
            assert_eq!(cum[N_SYMBOLS], FREQ_TOTAL);
            assert!(cum.windows(2).all(|w| w[1] > w[0]), "zero frequency at sharp={sharp}");
        }
    }

    #[test]
    fn roundtrip_with_changing_distributions() {
        let mut rng = StdRng::seed_from_u64(7);
        let n = 20_000;
        let mut tables = Vec::with_capacity(n);
        let mut symbols = Vec::with_capacity(n);
        let mut ideal_bits = 0f64;
        for i in 0..n {
            let mut cum = [0u32; N_SYMBOLS + 1];
            if i == 0 {
                uniform_cum(&mut cum);
            } else {
                quantize_logits(&random_logits(&mut rng, 8.0), &mut cum);
            }
            // Mostly likely symbols, sometimes the rarest one.
            let s = if i % 97 == 0 {
                (0..N_SYMBOLS).min_by_key(|&s| cum[s + 1] - cum[s]).unwrap()
            } else {
                (0..N_SYMBOLS).max_by_key(|&s| cum[s + 1] - cum[s]).unwrap()
            } as u8;
            let p = (cum[s as usize + 1] - cum[s as usize]) as f64 / cum[N_SYMBOLS] as f64;
            ideal_bits -= p.log2();
            tables.push(cum);
            symbols.push(s);
        }
        let mut enc = Encoder::new();
        for (s, cum) in symbols.iter().zip(&tables) {
            enc.encode(*s, cum);
        }
        let bytes = enc.finish();
        let mut dec = Decoder::new(&bytes);
        for (i, (s, cum)) in symbols.iter().zip(&tables).enumerate() {
            assert_eq!(dec.decode(cum), *s, "mismatch at symbol {i}");
        }
        let actual_bits = bytes.len() as f64 * 8.0;
        assert!(
            actual_bits <= ideal_bits * 1.001 + 64.0,
            "coder overhead too large: {actual_bits} vs ideal {ideal_bits:.0}"
        );
    }
}
