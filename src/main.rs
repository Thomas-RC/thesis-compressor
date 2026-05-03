use anyhow::Result;
use candle_core::Device;
use thesis_compressor::config::Config;
use thesis_compressor::data::load_enwik8;
use thesis_compressor::train::{TrainConfig, train};

fn pick_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        match Device::new_cuda(0) {
            Ok(d) => {
                println!("[+] device: CUDA(0)");
                return d;
            }
            Err(e) => {
                eprintln!("[!] CUDA niedostępne ({e}), fallback CPU");
            }
        }
    }
    println!("[+] device: CPU");
    Device::Cpu
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| "data/enwik8".to_string());
    let preset = args.next().unwrap_or_else(|| "small".to_string());

    println!("[+] wczytuję enwik8 z: {path}");
    let splits = load_enwik8(&path)?;
    println!(
        "    train={} B | val={} B | test={} B",
        splits.train.len(),
        splits.val.len(),
        splits.test.len()
    );

    let device = pick_device();
    let model_cfg = Config::from_preset(&preset)?;
    let train_cfg = TrainConfig::default();

    println!(
        "[+] preset='{preset}' vocab={} seq_len={} d_model={} n_layers={} n_heads={} ffn_mult={}",
        model_cfg.vocab_size,
        model_cfg.seq_len,
        model_cfg.d_model,
        model_cfg.n_layers,
        model_cfg.n_heads,
        model_cfg.ffn_mult
    );
    println!(
        "[+] trening: steps={} batch={} lr_max={} lr_min={} warmup={} decay_start={}%",
        train_cfg.n_steps,
        train_cfg.batch_size,
        train_cfg.lr_max,
        train_cfg.lr_min,
        train_cfg.warmup_steps,
        (train_cfg.decay_start_frac * 100.0) as u32,
    );

    train(model_cfg, &splits.train, &splits.val, train_cfg, &device)?;

    Ok(())
}
