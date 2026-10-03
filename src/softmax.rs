// Last-dim softmax with a fused forward kernel and an explicit backward pass.
//
// candle-nn's `ops::softmax_last_dim` runs a fused kernel but registers no
// backward (`apply_op1_no_bwd`), so it cuts the autograd graph. The composed
// `ops::softmax` is differentiable but keeps ~5 intermediate (B, H, T, T)
// tensors alive per attention layer, which does not fit in 16 GB for T=1024.
// This op reuses candle's CUDA softmax kernel for the forward pass and needs
// only its output y for the backward pass:
//     dL/dx = y * (g - sum(g * y, dim=-1))
// The CUDA forward is adapted from `SoftmaxLastDim` in candle-nn 0.10.2
// (MIT / Apache-2.0).

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp1, D, Layout, Result, Shape, Tensor};

struct SoftmaxLastDim;

impl CustomOp1 for SoftmaxLastDim {
    fn name(&self) -> &'static str {
        "softmax-last-dim-bwd"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let src = match storage {
            CpuStorage::F32(s) => s,
            _ => candle_core::bail!("softmax_last_dim: only f32 is supported, got {:?}", storage.dtype()),
        };
        let src = match layout.contiguous_offsets() {
            None => candle_core::bail!("softmax_last_dim: input has to be contiguous"),
            Some((o1, o2)) => &src[o1..o2],
        };
        let dims = layout.shape().dims();
        let dim_m1 = dims[dims.len() - 1];
        let mut dst = vec![0f32; src.len()];
        for (src, dst) in src.chunks(dim_m1).zip(dst.chunks_mut(dim_m1)) {
            let max = src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for (s, d) in src.iter().zip(dst.iter_mut()) {
                *d = (*s - max).exp();
                sum += *d;
            }
            for d in dst.iter_mut() {
                *d /= sum;
            }
        }
        Ok((CpuStorage::F32(dst), Shape::from_dims(dims)))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &candle_core::CudaStorage,
        layout: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle_core::cuda_backend::{Map1, WrapErr, kernel_name, kernels};
        use candle_core::{CudaDevice, WithDType};

        struct S;
        impl Map1 for S {
            fn f<T: DeviceRepr + WithDType>(
                &self,
                src: &CudaSlice<T>,
                dev: &CudaDevice,
                layout: &Layout,
            ) -> Result<CudaSlice<T>> {
                let src = match layout.contiguous_offsets() {
                    None => candle_core::bail!("softmax_last_dim: input has to be contiguous"),
                    Some((o1, o2)) => src.slice(o1..o2),
                };
                let el = layout.shape().elem_count();
                let dims = layout.shape().dims();
                let dim_m1 = dims[dims.len() - 1];
                let (n_rows, n_cols) = (el / dim_m1, dim_m1);

                let cfg = LaunchConfig {
                    grid_dim: (n_rows as u32, 1, 1),
                    block_dim: (1, 32, 1),
                    shared_mem_bytes: 0,
                };
                let func = dev.get_or_load_func(&kernel_name::<T>("softmax"), &kernels::REDUCE)?;
                // SAFETY: fully written by the kernel below.
                let dst = unsafe { dev.alloc::<T>(el)? };
                let mut builder = func.builder();
                builder.arg(&src);
                builder.arg(&dst);
                candle_core::builder_arg!(builder, n_cols as i32);
                // SAFETY: ffi.
                unsafe { builder.launch(cfg) }.w()?;
                Ok(dst)
            }
        }

        let dev = storage.device();
        let slice = S.map(&storage.slice, dev, layout)?;
        let dst = candle_core::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, layout.shape().clone()))
    }

    fn bwd(&self, _arg: &Tensor, res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        let dot = (grad_res * res)?.sum_keepdim(D::Minus1)?;
        let grad_arg = (res * grad_res.broadcast_sub(&dot)?)?;
        Ok(Some(grad_arg))
    }
}

/// Differentiable softmax over the last dimension (input must be contiguous).
pub fn softmax_last_dim(xs: &Tensor) -> Result<Tensor> {
    xs.contiguous()?.apply_op1(SoftmaxLastDim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar().unwrap()
    }

    #[test]
    fn matches_composed_softmax_forward_and_backward() {
        let device = Device::Cpu;
        let (rows, cols) = (6, 8);
        let x: Vec<f32> = (0..rows * cols).map(|i| ((i * 7919) % 97) as f32 / 13.0 - 3.0).collect();
        let mut mask = vec![0f32; rows * cols];
        for r in 0..rows {
            for c in (r + 1)..cols {
                mask[r * cols + c] = f32::NEG_INFINITY;
            }
        }
        let mask = Tensor::from_vec(mask, (rows, cols), &device).unwrap();
        let w = Tensor::from_vec(
            (0..rows * cols).map(|i| (i % 5) as f32 - 2.0).collect::<Vec<f32>>(),
            (rows, cols),
            &device,
        )
        .unwrap();

        let run = |use_custom: bool| -> (Tensor, Tensor) {
            let xv = Var::from_vec(x.clone(), (rows, cols), &device).unwrap();
            let s = xv.as_tensor().broadcast_add(&mask).unwrap();
            let y = if use_custom {
                softmax_last_dim(&s).unwrap()
            } else {
                candle_nn::ops::softmax(&s, D::Minus1).unwrap()
            };
            let loss = (&y * &w).unwrap().sum_all().unwrap();
            let grads = loss.backward().unwrap();
            (y, grads.get(xv.as_tensor()).unwrap().clone())
        };
        let (y_ref, g_ref) = run(false);
        let (y, g) = run(true);
        assert!(max_abs_diff(&y, &y_ref) < 1e-6, "forward mismatch");
        assert!(max_abs_diff(&g, &g_ref) < 1e-5, "backward mismatch");
    }
}
