use super::{GpuFailure, GPU_CONTEXT, SGD_BIND_GROUP_LAYOUT, SGD_PIPELINE};
use crate::nn::{
    tensors::{Tensor, WithGrad},
    TensorFloat,
};
use alloc::sync::Arc;
use alloc::vec::Vec;
use briny::raw::{slice_from_bytes, slice_to_bytes};
use core::sync::atomic::{AtomicBool, Ordering};
use tensor_optim::TensorOps;
use wgpu::util::DeviceExt;

/// Performs an in-place Stochastic Gradient Descent (SGD) update on the GPU.
///
/// For each weight `w[i]`, performs: `w[i] -= lr * grad[i]`.
/// Updates the tensor in-place and returns success/failure.
///
/// # Parameters
///
/// - `w`: The weight tensor (with associated gradient)
/// - `lr`: Learning rate (`TensorFloat`, cast to `f32` for GPU)
///
/// # Returns
///
/// - `true` if update succeeded
/// - `false` on GPU execution failure
#[cfg(feature = "dyntensor")]
pub fn wgpu_sgd(w: &mut WithGrad<Tensor<TensorFloat>>, lr: TensorFloat) -> bool {
    let weights_data = w.get_value().data();
    let grads_data = w.get_grad().data();

    // validate lengths
    if weights_data.is_empty()
        || weights_data.len() != grads_data.len()
        || weights_data.len() % 4 != 0
    {
        return false;
    }

    // convert f64 to f32 vectors (minimal allocations: one for weights, one for grads)
    let mut weights_f32: Vec<f32> = weights_data.iter().map(|&x| x as f32).collect();
    let mut grads_f32: Vec<f32> = grads_data.iter().map(|&x| x as f32).collect();

    // run the GPU shader (async wrapped by pollster)
    if super::block_on_gpu(run_sgd_shader(&mut weights_f32, &mut grads_f32, lr as f32)).is_err() {
        return false;
    }

    let (weights_tensor, grads_tensor) = w.split_mut();
    let weights_mut = weights_tensor.data_mut();
    let grads_mut = grads_tensor.data_mut();

    for (dst, &src) in weights_mut.iter_mut().zip(weights_f32.iter()) {
        *dst = TensorFloat::from(src);
    }
    for (dst, &src) in grads_mut.iter_mut().zip(grads_f32.iter()) {
        *dst = TensorFloat::from(src);
    }

    true
}
/// Performs an in-place Stochastic Gradient Descent (SGD) update on the GPU.
///
/// For each weight `w[i]`, performs: `w[i] -= lr * grad[i]`.
/// Updates the tensor in-place and returns success/failure.
///
/// # Parameters
///
/// - `w`: The weight tensor (with associated gradient)
/// - `lr`: Learning rate (`TensorFloat`, cast to `f32` for GPU)
///
/// # Returns
///
/// - `true` if update succeeded
/// - `false` on GPU execution failure
#[cfg(not(feature = "dyntensor"))]
pub fn wgpu_sgd<const N: usize, const D: usize>(
    w: &mut WithGrad<Tensor<TensorFloat, N, D>>,
    lr: TensorFloat,
) -> bool {
    let weights_data = w.get_value().data();
    let grads_data = w.get_grad().data();

    // validate lengths
    if weights_data.is_empty()
        || weights_data.len() != grads_data.len()
        || weights_data.len() % 4 != 0
    {
        return false;
    }

    // convert f64 to f32 vectors (minimal allocations: one for weights, one for grads)
    let mut weights_f32: Vec<f32> = weights_data.iter().map(|&x| x as f32).collect();
    let mut grads_f32: Vec<f32> = grads_data.iter().map(|&x| x as f32).collect();

    // run the GPU shader (async wrapped by pollster)
    if super::block_on_gpu(run_sgd_shader(&mut weights_f32, &mut grads_f32, lr as f32)).is_err() {
        return false;
    }

    let (weights_tensor, grads_tensor) = w.split_mut();
    let weights_mut = weights_tensor.data_mut();
    let grads_mut = grads_tensor.data_mut();

    for (dst, &src) in weights_mut.iter_mut().zip(weights_f32.iter()) {
        *dst = TensorFloat::from(src);
    }
    for (dst, &src) in grads_mut.iter_mut().zip(grads_f32.iter()) {
        *dst = TensorFloat::from(src);
    }

    true
}

async fn run_sgd_shader(weights: &mut [f32], grad: &mut [f32], lr: f32) -> Result<(), GpuFailure> {
    assert_eq!(weights.len(), grad.len());
    assert_eq!(weights.len() % 4, 0);
    let device = &GPU_CONTEXT.device;
    let queue = &GPU_CONTEXT.queue;

    // GPU buffers. NOTE: grad_buf must be COPY_SRC so we can copy its post-shader contents out.
    let weights_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("weights"),
        contents: slice_to_bytes(weights),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let grad_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("grad"),
        contents: slice_to_bytes(grad),
        // allow shader read/write AND allow copying the post-shader content back to CPU
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let lr_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("lr"),
        contents: slice_to_bytes(&[lr]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group_layout = &*SGD_BIND_GROUP_LAYOUT;

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sgd_bind_group"),
        layout: bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: weights_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: grad_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: lr_buf.as_entire_binding(),
            },
        ],
    });

    let pipeline = &*SGD_PIPELINE;

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("sgd_encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sgd_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);

        // each invocation handles 4 floats; workgroup_size = 64 -> 256 floats per group
        let num_workgroups = weights.len().div_ceil(256) as u32;
        cpass.dispatch_workgroups(num_workgroups, 1, 1);
    }

    // staging buffers to read results back
    let staging_weights = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_weights"),
        size: (weights.len() * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let staging_grad = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_grad"),
        size: (grad.len() * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // copy GPU buffers to staging buffers
    encoder.copy_buffer_to_buffer(
        &weights_buf,
        0,
        &staging_weights,
        0,
        (weights.len() * 4) as u64,
    );
    encoder.copy_buffer_to_buffer(&grad_buf, 0, &staging_grad, 0, (grad.len() * 4) as u64);

    // submit and map both staging buffers
    queue.submit(Some(encoder.finish()));

    // atomic flags to signal when mapping is ready
    let weights_ready = Arc::new(AtomicBool::new(false));
    let grad_ready = Arc::new(AtomicBool::new(false));

    // map weights buffer
    {
        let weights_ready_clone = Arc::clone(&weights_ready);
        staging_weights
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |_| {
                weights_ready_clone.store(true, Ordering::Release);
            });
    }

    // map grad buffer
    {
        let grad_ready_clone = Arc::clone(&grad_ready);
        staging_grad
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |_| {
                grad_ready_clone.store(true, Ordering::Release);
            });
    }

    let _ = GPU_CONTEXT.device.poll(wgpu::PollType::Wait);

    // now safe to get mapped ranges
    let view_w = staging_weights.slice(..).get_mapped_range();
    let updated_weights: &[f32] = slice_from_bytes(&view_w)?;
    weights.copy_from_slice(updated_weights);
    drop(view_w);
    staging_weights.unmap();

    let view_g = staging_grad.slice(..).get_mapped_range();
    let updated_grads: &[f32] = slice_from_bytes(&view_g)?;
    grad.copy_from_slice(updated_grads);
    drop(view_g);
    staging_grad.unmap();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::approx::approx_eq;
    use crate::nn::{ops::wgpu::array_from_slice, tensors::TensorOps};
    use alloc::{vec, vec::Vec};

    #[cfg(feature = "dyntensor")]
    fn make_withgrad<const N: usize>(data: &[f32]) -> WithGrad<Tensor<TensorFloat>> {
        let tensor_data = data.iter().map(|&x| x as TensorFloat).collect::<Vec<_>>();
        let tensor = Tensor::new(
            &[data.len()],
            &array_from_slice::<TensorFloat, N>(&tensor_data),
        );
        let mut wg = WithGrad::new(tensor);
        wg.set_grad(Tensor::new(
            &[data.len()],
            &array_from_slice::<TensorFloat, N>(&vec![0.0; data.len()]),
        ));
        wg
    }

    #[cfg(not(feature = "dyntensor"))]
    fn make_withgrad<const N: usize>(data: &[f32]) -> WithGrad<Tensor<TensorFloat, N, 1>> {
        let tensor_data = data.iter().map(|&x| x as TensorFloat).collect::<Vec<_>>();
        let tensor = Tensor::new(&[data.len()], &array_from_slice(&tensor_data));
        let mut wg = WithGrad::new(tensor);
        wg.set_grad(Tensor::new(
            &[data.len()],
            &array_from_slice(&vec![0.0; data.len()]),
        ));
        wg
    }

    #[test]
    fn wgpu_sgd_basic_update() {
        let mut w = make_withgrad::<4>(&[1.0, 2.0, 3.0, 4.0]); // length 4, divisible by 4
        {
            // Provide gradient matching weights (simulate grad)
            let grad_data = [0.1, 0.2, 0.3, 0.4];
            w.get_grad_mut().data_mut().copy_from_slice(
                &grad_data
                    .iter()
                    .map(|&x| x as TensorFloat)
                    .collect::<Vec<_>>(),
            );
        }

        let lr = 0.1;
        let success = wgpu_sgd(&mut w, lr);
        assert!(success);

        let updated = w.get_value().data();
        let expected: Vec<f32> = vec![
            1.0 - 0.1 * 0.1,
            2.0 - 0.1 * 0.2,
            3.0 - 0.1 * 0.3,
            4.0 - 0.1 * 0.4,
        ];

        for (&u, &e) in updated.iter().zip(expected.iter()) {
            assert!(approx_eq(u as f32, e));
        }
    }

    #[test]
    fn wgpu_sgd_zero_gradient() {
        let mut w = make_withgrad::<4>(&[5.0, 6.0, 7.0, 8.0]);
        // Grad zero: weights shouldn't change
        let grad_zero = vec![0.0, 0.0, 0.0, 0.0];
        w.get_grad_mut().data_mut().copy_from_slice(
            &grad_zero
                .iter()
                .map(|&x| x as TensorFloat)
                .collect::<Vec<_>>(),
        );

        let lr = 0.5;
        let success = wgpu_sgd(&mut w, lr);
        assert!(success);

        let updated = w.get_value().data();
        let expected = vec![5.0, 6.0, 7.0, 8.0];
        for (&u, &e) in updated.iter().zip(expected.iter()) {
            assert!(approx_eq(u as f32, e));
        }
    }

    #[test]
    fn wgpu_sgd_invalid_length_fails() {
        // Length not divisible by 4
        let mut w = make_withgrad::<3>(&[1.0, 2.0, 3.0]);
        let lr = 0.1;
        let result = wgpu_sgd(&mut w, lr);
        assert!(!result);
    }

    #[test]
    fn wgpu_sgd_empty_input() {
        let mut w = make_withgrad::<0>(&[]);
        let lr = 0.1;
        let result = wgpu_sgd(&mut w, lr);
        assert!(!result);
    }
}
