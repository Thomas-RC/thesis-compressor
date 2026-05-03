use anyhow::Result;
use candle_core::Device;
use clap::{Parser, ValueEnum};
use thesis_compressor::config::Config;
use thesis_compressor::data::load_enwik8;
use thesis_compressor::train::{TrainConfig, train};

#[derive(Copy, Clone, Debug, ValueEnum)]
enum DeviceArg {
    Auto,
    Cpu,
    Cuda,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Preset {
    Small,
    Medium,
    Large,
}

impl Preset {
    fn as_str(self) -> &'static str {
        match self {
            Preset::Small => "small",
            Preset::Medium => "medium",
            Preset::Large => "large",
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "thesis-compressor", version, about = "Byte-level Transformer for lossless text compression")]
struct Args {
    /// Path to enwik8 corpus file
    #[arg(long, default_value = "data/enwik8")]
    data: String,

    /// Model size preset
    #[arg(long, value_enum, default_value_t = Preset::Medium)]
    preset: Preset,

    /// Compute device (auto = try CUDA, fallback CPU)
    #[arg(long, value_enum, default_value_t = DeviceArg::Auto)]
    device: DeviceArg,

    /// Number of training steps
    #[arg(long, default_value_t = 500)]
    steps: usize,

    /// Mini-batch size
    #[arg(long, default_value_t = 32)]
    batch_size: usize,

    /// Peak learning rate (after warmup)
    #[arg(long, default_value_t = 3e-4)]
    lr_max: f64,

    /// Final learning rate after decay
    #[arg(long, default_value_t = 3e-5)]
    lr_min: f64,

    /// Warmup steps (linear ramp from lr_min to lr_max)
    #[arg(long, default_value_t = 25)]
    warmup_steps: usize,

    /// Fraction of training after which linear decay begins (0.0-1.0)
    #[arg(long, default_value_t = 0.8)]
    decay_start_frac: f64,

    /// AdamW weight decay coefficient
    #[arg(long, default_value_t = 0.01)]
    weight_decay: f64,

    /// Run validation every N steps
    #[arg(long, default_value_t = 50)]
    eval_every: usize,

    /// Number of validation batches per eval
    #[arg(long, default_value_t = 10)]
    eval_batches: usize,

    /// Log training metrics every N steps
    #[arg(long, default_value_t = 10)]
    log_every: usize,

    /// RNG seed
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Path for CSV training log
    #[arg(long, default_value = "results/training.csv")]
    csv_path: String,

    /// Path for last-checkpoint safetensors
    #[arg(long, default_value = "checkpoints/model_last.safetensors")]
    last_checkpoint: String,

    /// Path for best-checkpoint safetensors (saved when val BPC improves)
    #[arg(long, default_value = "checkpoints/model_best.safetensors")]
    best_checkpoint: String,
}

fn pick_device(arg: DeviceArg) -> Device {
    match arg {
        DeviceArg::Cpu => {
            println!("[+] device: CPU (forced)");
            Device::Cpu
        }
        DeviceArg::Cuda => match try_cuda() {
            Some(d) => d,
            None => panic!("CUDA wybrane, ale niedostępne — przebuduj z --features cuda"),
        },
        DeviceArg::Auto => match try_cuda() {
            Some(d) => d,
            None => {
                println!("[+] device: CPU (CUDA niedostępne)");
                Device::Cpu
            }
        },
    }
}

#[cfg(feature = "cuda")]
fn try_cuda() -> Option<Device> {
    match Device::new_cuda(0) {
        Ok(d) => {
            println!("[+] device: CUDA(0)");
            Some(d)
        }
        Err(e) => {
            eprintln!("[!] CUDA runtime error: {e}");
            None
        }
    }
}

#[cfg(not(feature = "cuda"))]
fn try_cuda() -> Option<Device> {
    None
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("[+] wczytuję enwik8 z: {}", args.data);
    let splits = load_enwik8(&args.data)?;
    println!(
        "    train={} B | val={} B | test={} B",
        splits.train.len(),
        splits.val.len(),
        splits.test.len()
    );

    let device = pick_device(args.device);
    let model_cfg = Config::from_preset(args.preset.as_str())?;
    let train_cfg = TrainConfig {
        n_steps: args.steps,
        batch_size: args.batch_size,
        lr_max: args.lr_max,
        lr_min: args.lr_min,
        warmup_steps: args.warmup_steps,
        decay_start_frac: args.decay_start_frac,
        weight_decay: args.weight_decay,
        eval_every: args.eval_every,
        eval_batches: args.eval_batches,
        log_every: args.log_every,
        seed: args.seed,
        csv_path: args.csv_path,
        last_checkpoint_path: args.last_checkpoint,
        best_checkpoint_path: args.best_checkpoint,
    };

    println!(
        "[+] preset='{}' vocab={} seq_len={} d_model={} n_layers={} n_heads={} ffn_mult={}",
        args.preset.as_str(),
        model_cfg.vocab_size,
        model_cfg.seq_len,
        model_cfg.d_model,
        model_cfg.n_layers,
        model_cfg.n_heads,
        model_cfg.ffn_mult
    );
    println!(
        "[+] trening: steps={} batch={} lr_max={} lr_min={} warmup={} decay@{}%",
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
