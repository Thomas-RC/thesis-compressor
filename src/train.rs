use anyhow::Result;
use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use candle_nn::{VarBuilder, VarMap};
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use crate::Arch;
use crate::config::Config;
use crate::data::BatchSampler;
use crate::model::Model;
use crate::model_llama::LlamaModel;
use crate::model_rms::Model as RmsModel;
use crate::model_rope::Model as RopeModel;
use crate::model_swiglu::Model as SwigluModel;

enum AnyModel {
    Baseline(Model),
    Llama(LlamaModel),
    LlamaRms(RmsModel),
    LlamaRope(RopeModel),
    LlamaSwiglu(SwigluModel),
}

impl AnyModel {
    fn forward(&self, x: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        match self {
            AnyModel::Baseline(m) => m.forward(x),
            AnyModel::Llama(m) => m.forward(x),
            AnyModel::LlamaRms(m) => m.forward(x),
            AnyModel::LlamaRope(m) => m.forward(x),
            AnyModel::LlamaSwiglu(m) => m.forward(x),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub arch: Arch,
    pub n_steps: usize,
    pub batch_size: usize,
    /// Micro-batches accumulated per optimizer step (effective batch = batch_size * grad_accum).
    pub grad_accum: usize,
    pub lr_max: f64,
    pub lr_min: f64,
    pub warmup_steps: usize,
    pub decay_start_frac: f64,
    pub weight_decay: f64,
    /// Max global L2 norm of gradients (0 = no clipping).
    pub grad_clip: f64,
    pub eval_every: usize,
    pub eval_batches: usize,
    pub log_every: usize,
    pub seed: u64,
    pub csv_path: String,
    pub last_checkpoint_path: String,
    pub best_checkpoint_path: String,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            arch: Arch::Baseline,
            n_steps: 500,
            batch_size: 32,
            grad_accum: 1,
            lr_max: 3e-4,
            lr_min: 3e-5,
            warmup_steps: 25,
            decay_start_frac: 0.8,
            weight_decay: 0.01,
            grad_clip: 1.0,
            eval_every: 50,
            eval_batches: 10,
            log_every: 10,
            seed: 42,
            csv_path: "results/training.csv".into(),
            last_checkpoint_path: "checkpoints/model_last.safetensors".into(),
            best_checkpoint_path: "checkpoints/model_best.safetensors".into(),
        }
    }
}

/// Global L2 norm of all gradients; rescales them in place when it exceeds
/// `max_norm` (0 disables clipping). Returns the norm before clipping.
fn clip_grad_norm(grads: &mut GradStore, vars: &[Var], max_norm: f64) -> Result<f32> {
    let mut sq_sum: Option<Tensor> = None;
    for var in vars {
        if let Some(g) = grads.get(var.as_tensor()) {
            let sq = g.sqr()?.sum_all()?;
            sq_sum = Some(match sq_sum {
                None => sq,
                Some(acc) => (acc + sq)?,
            });
        }
    }
    let norm = match sq_sum {
        None => return Ok(0.0),
        Some(t) => t.to_scalar::<f32>()?.sqrt(),
    };
    if max_norm > 0.0 && norm as f64 > max_norm {
        let scale = max_norm / (norm as f64 + 1e-6);
        for var in vars {
            if let Some(g) = grads.remove(var.as_tensor()) {
                grads.insert(var.as_tensor(), (g * scale)?);
            }
        }
    }
    Ok(norm)
}

fn lr_at_step(step: usize, cfg: &TrainConfig) -> f64 {
    let total = cfg.n_steps;
    let warmup = cfg.warmup_steps;
    let decay_start = ((total as f64) * cfg.decay_start_frac) as usize;
    if step < warmup {
        let frac = (step as f64 + 1.0) / (warmup.max(1) as f64);
        cfg.lr_min + (cfg.lr_max - cfg.lr_min) * frac
    } else if step < decay_start {
        cfg.lr_max
    } else if step < total {
        let span = (total - decay_start).max(1) as f64;
        let progress = (step - decay_start) as f64 / span;
        cfg.lr_max + (cfg.lr_min - cfg.lr_max) * progress
    } else {
        cfg.lr_min
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TrainResult {
    pub final_val_bpc: f32,
    pub best_val_bpc: f32,
    pub wall_s: f32,
    pub n_params: usize,
}

pub fn train(
    model_cfg: Config,
    train_data: &[u8],
    val_data: &[u8],
    train_cfg: TrainConfig,
    device: &Device,
) -> Result<TrainResult> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = match train_cfg.arch {
        Arch::Baseline => AnyModel::Baseline(Model::new(model_cfg.clone(), vb)?),
        Arch::Llama => AnyModel::Llama(LlamaModel::new(model_cfg.clone(), vb)?),
        Arch::LlamaRms => AnyModel::LlamaRms(RmsModel::new(model_cfg.clone(), vb)?),
        Arch::LlamaRope => AnyModel::LlamaRope(RopeModel::new(model_cfg.clone(), vb)?),
        Arch::LlamaSwiglu => AnyModel::LlamaSwiglu(SwigluModel::new(model_cfg.clone(), vb)?),
    };
    let n_params = match train_cfg.arch {
        Arch::Baseline | Arch::LlamaRms | Arch::LlamaRope => model_cfg.param_count_estimate(),
        Arch::Llama => LlamaModel::param_count_estimate(&model_cfg),
        // SwiGLU FFN ma inny rozmiar niż 4*d GELU, podajemy szacunek z model.rs jako approx
        Arch::LlamaSwiglu => model_cfg.param_count_estimate(),
    };
    println!(
        "[+] arch={} init={} model: ~{n_params} parametrów ({:.2} M), device={device:?}",
        train_cfg.arch.name(),
        model_cfg.init.name(),
        n_params as f64 / 1.0e6
    );
    let accum = train_cfg.grad_accum.max(1);
    println!(
        "[+] grad_clip={} | batch={}x{} (efektywny {}, {} tokenów/krok)",
        train_cfg.grad_clip,
        train_cfg.batch_size,
        accum,
        train_cfg.batch_size * accum,
        train_cfg.batch_size * accum * model_cfg.seq_len
    );

    let opt_params = ParamsAdamW {
        lr: train_cfg.lr_max,
        weight_decay: train_cfg.weight_decay,
        ..Default::default()
    };
    let vars = varmap.all_vars();
    let mut opt = AdamW::new(vars.clone(), opt_params)?;

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
    if let Some(parent) = Path::new(&train_cfg.last_checkpoint_path).parent() {
        create_dir_all(parent)?;
    }
    if let Some(parent) = Path::new(&train_cfg.best_checkpoint_path).parent() {
        create_dir_all(parent)?;
    }
    let mut csv = BufWriter::new(File::create(&train_cfg.csv_path)?);
    writeln!(csv, "step,phase,loss_nat,bpc,lr,wall_s,grad_norm")?;

    let t0 = Instant::now();
    let ln2 = 2f32.ln();
    let mut best_val_bpc = f32::INFINITY;

    for step in 1..=train_cfg.n_steps {
        let lr = lr_at_step(step - 1, &train_cfg);
        opt.set_learning_rate(lr);

        // Gradients of intermediate micro-batches are summed into `acc` and their
        // GradStores dropped right away; the last GradStore carries the sum.
        let mut acc: Vec<Option<Tensor>> = vec![None; vars.len()];
        let mut loss_sum = 0f32;
        let mut last_grads = None;
        for micro in 0..accum {
            let (input, target) = train_sampler.sample(device)?;
            let logits = model.forward(&input)?;
            let (b, t, v) = logits.dims3()?;
            let logits_flat = logits.reshape((b * t, v))?;
            let target_flat = target.reshape((b * t,))?;
            let loss = candle_nn::loss::cross_entropy(&logits_flat, &target_flat)?;
            loss_sum += loss.to_scalar::<f32>()?;
            let mut g = (loss / accum as f64)?.backward()?;
            if micro + 1 < accum {
                for (slot, var) in acc.iter_mut().zip(&vars) {
                    if let Some(t) = g.remove(var.as_tensor()) {
                        *slot = Some(match slot.take() {
                            None => t,
                            Some(a) => (a + t)?,
                        });
                    }
                }
            } else {
                for (slot, var) in acc.iter_mut().zip(&vars) {
                    if let Some(a) = slot.take() {
                        let sum = match g.remove(var.as_tensor()) {
                            None => a,
                            Some(t) => (a + t)?,
                        };
                        g.insert(var.as_tensor(), sum);
                    }
                }
                last_grads = Some(g);
            }
        }
        let mut grads = last_grads.expect("accum >= 1");
        let grad_norm = clip_grad_norm(&mut grads, &vars, train_cfg.grad_clip)?;
        opt.step(&grads)?;
        drop(grads);

        let loss_value = loss_sum / accum as f32;
        let bpc = loss_value / ln2;
        let elapsed = t0.elapsed().as_secs_f32();

        if step % train_cfg.log_every == 0 {
            println!(
                "step {step:>5}/{} | train {loss_value:.4} nat | BPC {bpc:.4} | lr {lr:.2e} | gnorm {grad_norm:.3} | {elapsed:>6.1}s",
                train_cfg.n_steps
            );
            writeln!(
                csv,
                "{step},train,{loss_value:.6},{bpc:.6},{lr:.6e},{elapsed:.3},{grad_norm:.6}"
            )?;
        }

        if step % train_cfg.eval_every == 0 || step == train_cfg.n_steps {
            let (val_loss, val_bpc) =
                eval_model(&model, &mut val_sampler, train_cfg.eval_batches, device)?;
            let elapsed = t0.elapsed().as_secs_f32();
            let improved = val_bpc < best_val_bpc;
            let mark = if improved { "*" } else { " " };
            println!(
                "  -> val {val_loss:.4} nat | BPC {val_bpc:.4} {mark} (best={best_val_bpc:.4}, avg z {} batchy)",
                train_cfg.eval_batches
            );
            writeln!(
                csv,
                "{step},val,{val_loss:.6},{val_bpc:.6},{lr:.6e},{elapsed:.3},"
            )?;
            if improved {
                best_val_bpc = val_bpc;
                varmap.save(&train_cfg.best_checkpoint_path)?;
            }
            csv.flush()?;
        }
    }

    let (val_loss, val_bpc) =
        eval_model(&model, &mut val_sampler, train_cfg.eval_batches * 2, device)?;
    let elapsed = t0.elapsed().as_secs_f32();
    let lr_final = lr_at_step(train_cfg.n_steps.saturating_sub(1), &train_cfg);
    println!(
        "[+] FINAL val {val_loss:.4} nat | BPC {val_bpc:.4} (avg z {} batchy) | best={best_val_bpc:.4}",
        train_cfg.eval_batches * 2
    );
    writeln!(
        csv,
        "{},final,{val_loss:.6},{val_bpc:.6},{lr_final:.6e},{elapsed:.3},",
        train_cfg.n_steps
    )?;
    csv.flush()?;

    varmap.save(&train_cfg.last_checkpoint_path)?;
    println!(
        "[+] checkpoint last: {} | best: {}",
        train_cfg.last_checkpoint_path, train_cfg.best_checkpoint_path
    );
    println!("[+] log treningu: {}", train_cfg.csv_path);

    Ok(TrainResult {
        final_val_bpc: val_bpc,
        best_val_bpc,
        wall_s: t0.elapsed().as_secs_f32(),
        n_params,
    })
}

fn eval_model(
    model: &AnyModel,
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
