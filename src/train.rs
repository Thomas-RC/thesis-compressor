use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use candle_nn::{VarBuilder, VarMap};
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use crate::config::Config;
use crate::data::BatchSampler;
use crate::model::Model;

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub n_steps: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub weight_decay: f64,
    pub eval_every: usize,
    pub eval_batches: usize,
    pub log_every: usize,
    pub seed: u64,
    pub csv_path: String,
    pub checkpoint_path: String,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            n_steps: 500,
            batch_size: 32,
            lr: 3e-4,
            weight_decay: 0.01,
            eval_every: 50,
            eval_batches: 10,
            log_every: 10,
            seed: 42,
            csv_path: "results/training.csv".into(),
            checkpoint_path: "checkpoints/model.safetensors".into(),
        }
    }
}

pub fn train(
    model_cfg: Config,
    train_data: &[u8],
    val_data: &[u8],
    train_cfg: TrainConfig,
    device: &Device,
) -> Result<()> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = Model::new(model_cfg.clone(), vb)?;
    println!(
        "[+] model zainicjalizowany (~{} parametrów, {:.2} M)",
        model_cfg.param_count_estimate(),
        model_cfg.param_count_estimate() as f64 / 1.0e6
    );

    let opt_params = ParamsAdamW {
        lr: train_cfg.lr,
        weight_decay: train_cfg.weight_decay,
        ..Default::default()
    };
    let mut opt = AdamW::new(varmap.all_vars(), opt_params)?;

    let mut train_sampler =
        BatchSampler::new(train_data, model_cfg.seq_len, train_cfg.batch_size, train_cfg.seed);
    let mut val_sampler = BatchSampler::new(
        val_data,
        model_cfg.seq_len,
        train_cfg.batch_size,
        train_cfg.seed.wrapping_add(1),
    );

    if let Some(parent) = Path::new(&train_cfg.csv_path).parent() {
        create_dir_all(parent)?;
    }
    if let Some(parent) = Path::new(&train_cfg.checkpoint_path).parent() {
        create_dir_all(parent)?;
    }
    let mut csv = BufWriter::new(File::create(&train_cfg.csv_path)?);
    writeln!(csv, "step,phase,loss_nat,bpc,wall_s")?;

    let t0 = Instant::now();
    let ln2 = 2f32.ln();

    for step in 1..=train_cfg.n_steps {
        let (input, target) = train_sampler.sample(device)?;
        let logits = model.forward(&input)?;
        let (b, t, v) = logits.dims3()?;
        let logits_flat = logits.reshape((b * t, v))?;
        let target_flat = target.reshape((b * t,))?;
        let loss = candle_nn::loss::cross_entropy(&logits_flat, &target_flat)?;
        opt.backward_step(&loss)?;

        let loss_value: f32 = loss.to_scalar()?;
        let bpc = loss_value / ln2;
        let elapsed = t0.elapsed().as_secs_f32();

        if step % train_cfg.log_every == 0 {
            println!(
                "step {step:>5}/{} | train loss {loss_value:.4} nat | BPC {bpc:.4} | {elapsed:>6.1}s",
                train_cfg.n_steps
            );
            writeln!(csv, "{step},train,{loss_value:.6},{bpc:.6},{elapsed:.3}")?;
        }

        if step % train_cfg.eval_every == 0 {
            let (val_loss, val_bpc) =
                eval_model(&model, &mut val_sampler, train_cfg.eval_batches, device)?;
            let elapsed = t0.elapsed().as_secs_f32();
            println!(
                "  -> val   loss {val_loss:.4} nat | BPC {val_bpc:.4} (avg z {} batchy)",
                train_cfg.eval_batches
            );
            writeln!(csv, "{step},val,{val_loss:.6},{val_bpc:.6},{elapsed:.3}")?;
            csv.flush()?;
        }
    }

    let (val_loss, val_bpc) =
        eval_model(&model, &mut val_sampler, train_cfg.eval_batches * 2, device)?;
    let elapsed = t0.elapsed().as_secs_f32();
    println!(
        "[+] FINAL val loss {val_loss:.4} nat | BPC {val_bpc:.4} (avg z {} batchy)",
        train_cfg.eval_batches * 2
    );
    writeln!(
        csv,
        "{},final,{val_loss:.6},{val_bpc:.6},{elapsed:.3}",
        train_cfg.n_steps
    )?;
    csv.flush()?;

    varmap.save(&train_cfg.checkpoint_path)?;
    println!("[+] zapisano checkpoint: {}", train_cfg.checkpoint_path);
    println!("[+] log treningu: {}", train_cfg.csv_path);

    Ok(())
}

fn eval_model(
    model: &Model,
    sampler: &mut BatchSampler<'_>,
    n_batches: usize,
    device: &Device,
) -> Result<(f32, f32)> {
    let mut total_loss = 0f32;
    for _ in 0..n_batches {
        let (input, target) = sampler.sample(device)?;
        let logits = model.forward(&input)?;
        let (b, t, v) = logits.dims3()?;
        let logits_flat = logits.reshape((b * t, v))?;
        let target_flat = target.reshape((b * t,))?;
        let loss = candle_nn::loss::cross_entropy(&logits_flat, &target_flat)?;
        total_loss += loss.to_scalar::<f32>()?;
    }
    let avg_loss = total_loss / n_batches as f32;
    let bpc = avg_loss / 2f32.ln();
    Ok((avg_loss, bpc))
}
