// Regression test: after one backward pass every trainable parameter must
// receive a non-zero gradient. Fused candle-nn kernels built with
// `apply_op*_no_bwd` (layer_norm, rms_norm, softmax_last_dim) silently cut the
// autograd graph, which froze the whole network except the output head.

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use thesis_compressor::Arch;
use thesis_compressor::config::Config;
use thesis_compressor::init::InitScheme;
use thesis_compressor::model::Model;
use thesis_compressor::model_llama::LlamaModel;
use thesis_compressor::model_rms::Model as RmsModel;
use thesis_compressor::model_rope::Model as RopeModel;
use thesis_compressor::model_swiglu::Model as SwigluModel;

fn tiny_config(init: InitScheme) -> Config {
    Config {
        vocab_size: 256,
        seq_len: 16,
        d_model: 32,
        n_layers: 2,
        n_heads: 4,
        ffn_mult: 4,
        init,
    }
}

fn assert_all_params_get_gradient(arch: Arch) {
    for init in [InitScheme::Candle, InitScheme::Gpt2] {
        assert_all_params_get_gradient_with(arch, init);
    }
}

fn assert_all_params_get_gradient_with(arch: Arch, init: InitScheme) {
    let device = Device::Cpu;
    let cfg = tiny_config(init);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let input: Vec<u32> = (0..2 * cfg.seq_len as u32).map(|i| (i * 37 + 11) % 256).collect();
    let target: Vec<u32> = input.iter().map(|&b| (b + 1) % 256).collect();
    let input = Tensor::from_vec(input, (2, cfg.seq_len), &device).unwrap();
    let target = Tensor::from_vec(target, (2 * cfg.seq_len,), &device).unwrap();

    let logits = match arch {
        Arch::Baseline => Model::new(cfg.clone(), vb).unwrap().forward(&input),
        Arch::Llama => LlamaModel::new(cfg.clone(), vb).unwrap().forward(&input),
        Arch::LlamaRms => RmsModel::new(cfg.clone(), vb).unwrap().forward(&input),
        Arch::LlamaRope => RopeModel::new(cfg.clone(), vb).unwrap().forward(&input),
        Arch::LlamaSwiglu => SwigluModel::new(cfg.clone(), vb).unwrap().forward(&input),
    }
    .unwrap();
    let logits = logits.reshape((2 * cfg.seq_len, cfg.vocab_size)).unwrap();
    let loss = candle_nn::loss::cross_entropy(&logits, &target).unwrap();
    let grads = loss.backward().unwrap();

    let vars = varmap.data().lock().unwrap();
    let mut missing: Vec<String> = Vec::new();
    for (name, var) in vars.iter() {
        let ok = match grads.get(var.as_tensor()) {
            Some(g) => g.abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap() > 0.0,
            None => false,
        };
        if !ok {
            missing.push(name.clone());
        }
    }
    missing.sort();
    assert!(
        missing.is_empty(),
        "arch={} init={}: {} / {} parametrów bez gradientu: {:?}",
        arch.name(),
        init.name(),
        missing.len(),
        vars.len(),
        missing
    );
}

#[test]
fn baseline_all_params_get_gradient() {
    assert_all_params_get_gradient(Arch::Baseline);
}

#[test]
fn llama_all_params_get_gradient() {
    assert_all_params_get_gradient(Arch::Llama);
}

#[test]
fn llama_rms_all_params_get_gradient() {
    assert_all_params_get_gradient(Arch::LlamaRms);
}

#[test]
fn llama_rope_all_params_get_gradient() {
    assert_all_params_get_gradient(Arch::LlamaRope);
}

#[test]
fn llama_swiglu_all_params_get_gradient() {
    assert_all_params_get_gradient(Arch::LlamaSwiglu);
}
