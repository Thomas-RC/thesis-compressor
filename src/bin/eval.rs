use anyhow::{Result, ensure};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use clap::{Parser, ValueEnum};
use std::time::Instant;
use thesis_compressor::config::Config;
use thesis_compressor::data::{Splits, load_enwik8};
use thesis_compressor::model::Model;

#[derive(Copy, Clone, Debug, ValueEnum)]
enum DeviceArg {
    Auto,
    Cpu,
    Cuda,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum SplitArg {
    Train,
    Val,
    Test,
}

impl SplitArg {
    fn pick<'a>(self, splits: &'a Splits) -> &'a [u8] {
        match self {
            SplitArg::Train => &splits.train,
            SplitArg::Val => &splits.val,
            SplitArg::Test => &splits.test,
        }
    }

    fn name(self) -> &'static str {
        match self {
            SplitArg::Train => "train",
            SplitArg::Val => "val",
            SplitArg::Test => "test",
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "eval",
    about = "Deterministic full-split BPC evaluation (non-overlapping windows)"
)]
struct Args {
    /// Path to enwik8 corpus
    #[arg(long, default_value = "data/enwik8")]
    data: String,

    /// Path to safetensors checkpoint produced by training
    #[arg(long)]
    checkpoint: String,

    /// Which split to evaluate
    #[arg(long, value_enum, default_value_t = SplitArg::Test)]
    split: SplitArg,

    /// Model preset matching the checkpoint architecture
    #[arg(long, default_value = "large")]
    preset: String,

    /// Mini-batch size during evaluation (limited by VRAM, not by accuracy)
    #[arg(long, default_value_t = 16)]
    batch_size: usize,

    /// Compute device
    #[arg(long, value_enum, default_value_t = DeviceArg::Auto)]
    device: DeviceArg,

    /// Optional path for a one-line CSV summary (split,bytes,predicted,bpc,wall_s)
    #[arg(long)]
    csv: Option<String>,
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

fn main() -> Result<()> {
    let args = Args::parse();
    let device = pick_device(args.device);
    let cfg = Config::from_preset(&args.preset)?;

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = Model::new(cfg.clone(), vb)?;
    varmap.load(&args.checkpoint)?;

    let splits = load_enwik8(&args.data)?;
    let data = args.split.pick(&splits);
    let split_name = args.split.name();

    let window = cfg.seq_len + 1;
    ensure!(data.len() >= window, "split too small for seq_len={}", cfg.seq_len);
    let n_windows = data.len() / window;
    let predicted = n_windows * cfg.seq_len;
    let coverage = predicted as f64 / data.len() as f64;

    println!("[+] checkpoint: {}", args.checkpoint);
    println!(
        "[+] preset='{}' seq_len={} batch={} device={:?}",
        args.preset, cfg.seq_len, args.batch_size, device
    );
    println!(
        "[+] split={} bytes={} windows={} predicted={} coverage={:.4}",
        split_name,
        data.len(),
        n_windows,
        predicted,
        coverage
    );

    let bs = args.batch_size;
    let n_batches = n_windows.div_ceil(bs);

    let t0 = Instant::now();
    let mut sum_loss = 0.0_f64;
    let mut sum_predicted = 0_usize;

    for batch_idx in 0..n_batches {
        let start_w = batch_idx * bs;
        let end_w = (start_w + bs).min(n_windows);
        let actual_bs = end_w - start_w;

        let mut input_buf: Vec<u32> = Vec::with_capacity(actual_bs * cfg.seq_len);
        let mut target_buf: Vec<u32> = Vec::with_capacity(actual_bs * cfg.seq_len);
        for w in start_w..end_w {
            let off = w * window;
            let chunk = &data[off..off + window];
            input_buf.extend(chunk[..cfg.seq_len].iter().map(|&b| b as u32));
            target_buf.extend(chunk[1..].iter().map(|&b| b as u32));
        }
        let input = Tensor::from_vec(input_buf, (actual_bs, cfg.seq_len), &device)?;
        let target = Tensor::from_vec(target_buf, (actual_bs, cfg.seq_len), &device)?;

        let logits = model.forward(&input)?;
        let (b, t, v) = logits.dims3()?;
        let logits_flat = logits.reshape((b * t, v))?;
        let target_flat = target.reshape((b * t,))?;
        let loss = candle_nn::loss::cross_entropy(&logits_flat, &target_flat)?;
        let loss_mean: f32 = loss.to_scalar()?;
        let n = actual_bs * cfg.seq_len;
        sum_loss += loss_mean as f64 * n as f64;
        sum_predicted += n;

        if batch_idx % 50 == 0 || batch_idx + 1 == n_batches {
            let progress = (batch_idx + 1) as f64 / n_batches as f64 * 100.0;
            let running_bpc = (sum_loss / sum_predicted as f64) / 2f64.ln();
            println!(
                "  [{progress:>5.1}%] batch {}/{n_batches} | running BPC = {running_bpc:.4}",
                batch_idx + 1
            );
        }
    }

    let avg_loss = sum_loss / sum_predicted as f64;
    let bpc = avg_loss / 2f64.ln();
    let elapsed = t0.elapsed().as_secs_f64();

    println!();
    println!(
        "[+] FULL {} EVAL ({} bytes / {} predicted, coverage {:.4})",
        split_name.to_uppercase(),
        data.len(),
        sum_predicted,
        sum_predicted as f64 / data.len() as f64
    );
    println!("    avg loss (nat)  = {avg_loss:.6}");
    println!("    BPC             = {bpc:.6}");
    println!("    wall            = {elapsed:.2} s");

    if let Some(csv_path) = args.csv {
        if let Some(parent) = std::path::Path::new(&csv_path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let exists = std::path::Path::new(&csv_path).exists();
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&csv_path)?;
        use std::io::Write;
        if !exists {
            writeln!(f, "checkpoint,split,bytes,predicted,bpc,wall_s")?;
        }
        writeln!(
            f,
            "{},{split_name},{},{sum_predicted},{bpc:.6},{elapsed:.2}",
            args.checkpoint,
            data.len()
        )?;
        println!("[+] CSV summary appended: {csv_path}");
    }

    Ok(())
}
