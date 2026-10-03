use anyhow::{Result, ensure};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use clap::{Parser, Subcommand, ValueEnum};
use std::fs;
use std::time::Instant;
use thesis_compressor::compressor::{self, Header, Options, fnv1a64};
use thesis_compressor::config::Config;
use thesis_compressor::model_llama::LlamaModel;

#[derive(Copy, Clone, Debug, ValueEnum)]
enum DeviceArg {
    Auto,
    Cpu,
    Cuda,
}

#[derive(Parser, Debug)]
#[command(
    name = "tcz",
    about = "Lossless byte-level compressor: Llama-style Transformer + arithmetic coding"
)]
struct Args {
    /// Model checkpoint (safetensors, llama architecture)
    #[arg(long, global = true, default_value = "checkpoints/long_large_llama_20000_lr5e-4_last.safetensors")]
    checkpoint: String,

    /// Compute device; decompression must use the same kind as compression
    #[arg(long, global = true, value_enum, default_value_t = DeviceArg::Auto)]
    device: DeviceArg,

    /// Hide progress output
    #[arg(long, global = true)]
    quiet: bool,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Compress INPUT into OUTPUT
    Compress {
        input: String,
        output: String,
        /// Model preset matching the checkpoint
        #[arg(long, default_value = "large")]
        preset: String,
        /// Independent streams coded in parallel (more = faster, slightly worse ratio)
        #[arg(long, default_value_t = 128)]
        streams: usize,
    },
    /// Decompress INPUT (.tcz) into OUTPUT
    Decompress { input: String, output: String },
    /// Compress and decompress in memory, verify and report statistics
    Roundtrip {
        input: String,
        #[arg(long, default_value = "large")]
        preset: String,
        #[arg(long, default_value_t = 128)]
        streams: usize,
    },
}

fn pick_device(arg: DeviceArg) -> Result<Device> {
    Ok(match arg {
        DeviceArg::Cpu => Device::Cpu,
        DeviceArg::Cuda => Device::new_cuda(0)?,
        DeviceArg::Auto => Device::new_cuda(0).unwrap_or(Device::Cpu),
    })
}

/// Loads the checkpoint as plain (non-trainable) tensors and hashes the file.
fn load_model(checkpoint: &str, preset: &str, device: &Device) -> Result<(LlamaModel, u64)> {
    let hash = fnv1a64(&fs::read(checkpoint)?);
    let cfg = Config::from_preset(preset)?;
    // SAFETY: the checkpoint file must not be modified while it is mapped.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[checkpoint], DType::F32, device)? };
    Ok((LlamaModel::new(cfg, vb)?, hash))
}

fn report(label: &str, orig: usize, packed: usize, secs: f64) {
    println!(
        "[+] {label}: {orig} B -> {packed} B | {:.4} bpc | ratio {:.2}x | {secs:.1} s ({:.1} kB/s)",
        packed as f64 * 8.0 / orig.max(1) as f64,
        orig as f64 / packed.max(1) as f64,
        orig as f64 / secs / 1e3
    );
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = pick_device(args.device)?;
    let progress = !args.quiet;
    println!("[+] device: {device:?} | checkpoint: {}", args.checkpoint);

    match args.cmd {
        Cmd::Compress {
            input,
            output,
            preset,
            streams,
        } => {
            let data = fs::read(&input)?;
            let (model, hash) = load_model(&args.checkpoint, &preset, &device)?;
            let opts = Options {
                streams,
                refill: None,
                progress,
            };
            let t0 = Instant::now();
            let packed = compressor::compress(&model, &preset, hash, &data, &opts)?;
            fs::write(&output, &packed)?;
            report("compress", data.len(), packed.len(), t0.elapsed().as_secs_f64());
        }
        Cmd::Decompress { input, output } => {
            let packed = fs::read(&input)?;
            let (header, payload) = Header::read(&packed)?;
            let (model, hash) = load_model(&args.checkpoint, &header.preset, &device)?;
            let t0 = Instant::now();
            let data = compressor::decompress(&model, &header, hash, payload, progress)?;
            fs::write(&output, &data)?;
            println!(
                "[+] decompress: {} B -> {} B | checksum OK | {:.1} s",
                packed.len(),
                data.len(),
                t0.elapsed().as_secs_f64()
            );
        }
        Cmd::Roundtrip {
            input,
            preset,
            streams,
        } => {
            let data = fs::read(&input)?;
            let (model, hash) = load_model(&args.checkpoint, &preset, &device)?;
            let opts = Options {
                streams,
                refill: None,
                progress,
            };
            let t0 = Instant::now();
            let packed = compressor::compress(&model, &preset, hash, &data, &opts)?;
            report("compress", data.len(), packed.len(), t0.elapsed().as_secs_f64());
            let (header, payload) = Header::read(&packed)?;
            let t1 = Instant::now();
            let unpacked = compressor::decompress(&model, &header, hash, payload, progress)?;
            ensure!(unpacked == data, "roundtrip: dane po dekompresji różnią się od wejścia");
            println!(
                "[+] decompress: OK, identyczne bajty | {:.1} s",
                t1.elapsed().as_secs_f64()
            );
        }
    }
    Ok(())
}
