use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use thesis_compressor::config::Config;
use thesis_compressor::data::{BatchSampler, ENWIK8_TOTAL, load_enwik8};
use thesis_compressor::model::Model;

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/enwik8".to_string());

    println!("[+] wczytuję enwik8 z: {path}");
    let splits = load_enwik8(&path)?;
    println!(
        "    train: {:>10} B | val: {:>9} B | test: {:>9} B | total: {} B",
        splits.train.len(),
        splits.val.len(),
        splits.test.len(),
        ENWIK8_TOTAL
    );

    let device = Device::Cpu;
    let cfg = Config::mvp();
    println!(
        "[+] config: vocab={} seq_len={} d_model={} n_layers={} n_heads={} ffn_mult={}",
        cfg.vocab_size, cfg.seq_len, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.ffn_mult
    );
    println!(
        "    przybliżona liczba parametrów: ~{} ({:.2} M)",
        cfg.param_count_estimate(),
        cfg.param_count_estimate() as f64 / 1.0e6
    );

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = Model::new(cfg.clone(), vb)?;
    println!("[+] model zainicjalizowany");

    let batch_size = 4;
    let mut sampler = BatchSampler::new(&splits.train, cfg.seq_len, batch_size, 42);
    let (input, target) = sampler.sample(&device)?;
    println!(
        "    batch input shape={:?}, target shape={:?}",
        input.shape(),
        target.shape()
    );

    println!("[+] forward pass...");
    let logits = model.forward(&input)?;
    println!("    logits shape={:?} dtype={:?}", logits.shape(), logits.dtype());
    let (b, t, v) = logits.dims3()?;
    anyhow::ensure!(
        b == batch_size && t == cfg.seq_len && v == cfg.vocab_size,
        "nieoczekiwany kształt logitów: ({b}, {t}, {v})"
    );

    let logits_flat = logits.reshape((b * t, v))?;
    let target_flat = target.reshape(b * t)?;
    let loss = candle_nn::loss::cross_entropy(&logits_flat, &target_flat)?;
    let loss_value: f32 = loss.to_scalar()?;
    let bpc = loss_value / 2f32.ln();
    let uniform_bpc = (cfg.vocab_size as f32).log2();

    println!(
        "[+] loss = {loss_value:.4} nat | BPC = {bpc:.4} | uniform-baseline BPC = {uniform_bpc:.4}"
    );
    println!(
        "    sanity: BPC blisko log2(256)={uniform_bpc:.2} oznacza, że nieuczeniona sieć daje rozkład ≈ uniformiczny — OK przed treningiem"
    );

    println!("[+] smoke test (faza 3) zakończony pomyślnie");
    Ok(())
}
