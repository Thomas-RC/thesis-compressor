use anyhow::Result;
use candle_core::Device;
use clap::{Parser, ValueEnum};
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use thesis_compressor::config::Config;
use thesis_compressor::data::{Splits, load_enwik8};
use thesis_compressor::train::{TrainConfig, train};

#[derive(Copy, Clone, Debug, ValueEnum)]
enum DeviceArg {
    Auto,
    Cpu,
    Cuda,
}

#[derive(Parser, Debug)]
#[command(name = "grid", about = "Grid search over (d_model, n_layers, seq_len)")]
struct Args {
    #[arg(long, default_value = "data/enwik8")]
    data: String,

    #[arg(long, value_enum, default_value_t = DeviceArg::Auto)]
    device: DeviceArg,

    /// Steps per configuration
    #[arg(long, default_value_t = 2000)]
    steps: usize,

    /// Mini-batch size (lower if you hit OOM on large configs)
    #[arg(long, default_value_t = 16)]
    batch_size: usize,

    #[arg(long, default_value_t = 3e-4)]
    lr_max: f64,

    #[arg(long, default_value_t = 3e-5)]
    lr_min: f64,

    #[arg(long, default_value_t = 50)]
    warmup_steps: usize,

    #[arg(long, default_value_t = 0.8)]
    decay_start_frac: f64,

    /// Comma-separated list, e.g. "128,256,512"
    #[arg(long, default_value = "128,256,512")]
    d_models: String,

    #[arg(long, default_value = "2,4,6")]
    n_layers: String,

    #[arg(long, default_value = "256,512,1024")]
    seq_lens: String,

    /// Aggregated CSV with one row per configuration
    #[arg(long, default_value = "results/grid.csv")]
    output: String,

    /// Directory for per-configuration training CSVs and checkpoints
    #[arg(long, default_value = "results/grid_detail")]
    detail_dir: String,
}

fn parse_list(s: &str) -> Result<Vec<usize>> {
    s.split(',')
        .map(|x| x.trim().parse::<usize>().map_err(Into::into))
        .collect()
}

fn n_heads_for(d_model: usize) -> usize {
    if d_model >= 512 {
        8
    } else if d_model >= 128 {
        4
    } else {
        2
    }
}

fn pick_device(arg: DeviceArg) -> Device {
    match arg {
        DeviceArg::Cpu => Device::Cpu,
        DeviceArg::Cuda | DeviceArg::Auto => try_cuda().unwrap_or(Device::Cpu),
    }
}

#[cfg(feature = "cuda")]
fn try_cuda() -> Option<Device> {
    Device::new_cuda(0).ok()
}

#[cfg(not(feature = "cuda"))]
fn try_cuda() -> Option<Device> {
    None
}

fn run_one(
    splits: &Splits,
    device: &Device,
    args: &Args,
    d_model: usize,
    n_layers: usize,
    seq_len: usize,
) -> Result<(f32, f32, f32, usize)> {
    let cfg = Config {
        vocab_size: 256,
        seq_len,
        d_model,
        n_layers,
        n_heads: n_heads_for(d_model),
        ffn_mult: 4,
    };
    let tag = format!("d{d_model}_L{n_layers}_s{seq_len}");
    let csv_path = format!("{}/{tag}.csv", args.detail_dir);
    let last_ckpt = format!("{}/{tag}_last.safetensors", args.detail_dir);
    let best_ckpt = format!("{}/{tag}_best.safetensors", args.detail_dir);

    let train_cfg = TrainConfig {
        arch: thesis_compressor::Arch::Baseline,
        n_steps: args.steps,
        batch_size: args.batch_size,
        lr_max: args.lr_max,
        lr_min: args.lr_min,
        warmup_steps: args.warmup_steps,
        decay_start_frac: args.decay_start_frac,
        weight_decay: 0.01,
        eval_every: args.steps / 10,
        eval_batches: 8,
        log_every: args.steps / 10,
        seed: 42,
        csv_path,
        last_checkpoint_path: last_ckpt,
        best_checkpoint_path: best_ckpt,
    };

    println!("=== running {tag} ===");
    let result = train(cfg, &splits.train, &splits.val, train_cfg, device)?;
    Ok((
        result.final_val_bpc,
        result.best_val_bpc,
        result.wall_s,
        result.n_params,
    ))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let d_models = parse_list(&args.d_models)?;
    let n_layers_list = parse_list(&args.n_layers)?;
    let seq_lens = parse_list(&args.seq_lens)?;

    println!(
        "[+] grid: d_models={d_models:?} n_layers={n_layers_list:?} seq_lens={seq_lens:?} = {} kombinacji",
        d_models.len() * n_layers_list.len() * seq_lens.len()
    );
    println!("[+] każda konfiguracja: {} kroków, batch={}", args.steps, args.batch_size);

    let device = pick_device(args.device);
    println!("[+] device={device:?}");

    println!("[+] wczytuję enwik8...");
    let splits = load_enwik8(&args.data)?;

    create_dir_all(&args.detail_dir)?;
    if let Some(parent) = std::path::Path::new(&args.output).parent() {
        create_dir_all(parent)?;
    }
    let mut out = BufWriter::new(File::create(&args.output)?);
    writeln!(
        out,
        "d_model,n_layers,seq_len,n_heads,n_params,final_val_bpc,best_val_bpc,wall_s"
    )?;
    out.flush()?;

    let mut idx = 0;
    let total = d_models.len() * n_layers_list.len() * seq_lens.len();
    for &d in &d_models {
        for &l in &n_layers_list {
            for &s in &seq_lens {
                idx += 1;
                let n_heads = n_heads_for(d);
                println!(
                    "[+] [{idx}/{total}] d_model={d} n_layers={l} seq_len={s} n_heads={n_heads}"
                );
                match run_one(&splits, &device, &args, d, l, s) {
                    Ok((final_bpc, best_bpc, wall_s, n_params)) => {
                        writeln!(
                            out,
                            "{d},{l},{s},{n_heads},{n_params},{final_bpc:.6},{best_bpc:.6},{wall_s:.2}"
                        )?;
                        out.flush()?;
                        println!(
                            "    OK: final={final_bpc:.4} best={best_bpc:.4} time={wall_s:.1}s"
                        );
                    }
                    Err(e) => {
                        eprintln!("    BŁĄD: {e}");
                        writeln!(out, "{d},{l},{s},{n_heads},NA,NA,NA,NA")?;
                        out.flush()?;
                    }
                }
            }
        }
    }

    println!("[+] grid zakończony, wyniki: {}", args.output);
    Ok(())
}
