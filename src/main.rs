use anyhow::Result;
use candle_core::Device;
use thesis_compressor::config::Config;
use thesis_compressor::data::load_enwik8;
use thesis_compressor::train::{TrainConfig, train};

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/enwik8".to_string());

    println!("[+] wczytuję enwik8 z: {path}");
    let splits = load_enwik8(&path)?;
    println!(
        "    train={} B | val={} B | test={} B",
        splits.train.len(),
        splits.val.len(),
        splits.test.len()
    );

    let device = Device::Cpu;
    let model_cfg = Config::mvp();
    let train_cfg = TrainConfig::default();

    println!(
        "[+] config: vocab={} seq_len={} d_model={} n_layers={} n_heads={}",
        model_cfg.vocab_size,
        model_cfg.seq_len,
        model_cfg.d_model,
        model_cfg.n_layers,
        model_cfg.n_heads
    );
    println!(
        "[+] trening: steps={} batch={} lr={} weight_decay={} eval_every={}",
        train_cfg.n_steps,
        train_cfg.batch_size,
        train_cfg.lr,
        train_cfg.weight_decay,
        train_cfg.eval_every
    );

    train(model_cfg, &splits.train, &splits.val, train_cfg, &device)?;

    Ok(())
}
