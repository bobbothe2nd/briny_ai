#![allow(clippy::cast_precision_loss, clippy::type_complexity)]

use core::sync::atomic::{AtomicBool, Ordering};

use super::{Box, GpuFailure, GPU_CONTEXT, MSE_LOSS_BIND_GROUP_LAYOUT, MSE_LOSS_PIPELINE};
use crate::nn::{
    tensors::{Tensor, TensorGrad, WithGrad},
    TensorFloat,
};
use alloc::sync::Arc;
use briny::raw::{slice_from_bytes, slice_to_bytes};
use tensor_optim::TensorOps;
use wgpu::util::DeviceExt;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Computes Mean Squared Error (MSE) loss between prediction and target on the GPU.
///
/// Uses a compute shader to calculate `(prediction[i] - target[i])^2`
/// and returns the average loss as an `TensorFloat` scalar.
///
/// # Returns
///
/// - `Some((loss, backward_fn))` on success
/// - `None` if GPU execution fails
///
/// # Backward Function
///
/// The returned closure takes a scalar gradient (usually 1.0 from upstream)
/// and computes elementwise gradients: `2 * (pred - target) / N`.
///
/// # Notes
///
/// - Loss is averaged over all elements
/// - Uses f32 internally and casts back to `TensorFloat`
#[must_use]
#[cfg(feature = "dyntensor")]
pub fn wgpu_mse_loss<'a>(
    pred: &'a WithGrad<Tensor<TensorFloat>>,
    target: &'a Tensor<TensorFloat>,
) -> Option<(
    TensorFloat,
    Box<dyn Fn(TensorFloat) -> Tensor<TensorFloat> + 'a>,
)> {
    let len = pred.get_value().len();
    if len == 0 {
        // Early return None or zero loss + noop backward, whichever makes sense
        return None;
    }

    let result =
        super::block_on_gpu(run_mse_loss_shader(pred.get_value().data(), target.data())).ok()?;

    let back = Box::new(move |grad: TensorFloat| {
        #[allow(clippy::cast_precision_loss)]
        let grad_data: Vec<TensorFloat> = pred
            .get_value()
            .data()
            .iter()
            .zip(target.data().iter())
            .map(|(&x, &y)| {
                2.0 * grad * TensorFloat::from(x - y) / pred.get_value().data().len() as TensorFloat
            })
            .collect();
        Tensor::new(pred.get_value().shape(), &grad_data)
    });

    Some((TensorFloat::from(result), Box::new(back)))
}
/// Computes Mean Squared Error (MSE) loss between prediction and target on the GPU.
///
/// Uses a compute shader to calculate `(prediction[i] - target[i])^2`
/// and returns the average loss as an `TensorFloat` scalar.
///
/// # Returns
///
/// - `Some((loss, backward_fn))` on success
/// - `None` if GPU execution fails
///
/// # Backward Function
///
/// The returned closure takes a scalar gradient (usually 1.0 from upstream)
/// and computes elementwise gradients: `2 * (pred - target) / N`.
///
/// # Notes
///
/// - Loss is averaged over all elements
/// - Uses f32 internally and casts back to `TensorFloat`
#[must_use]
#[cfg(not(feature = "dyntensor"))]
pub fn wgpu_mse_loss<'a, const N: usize, const D: usize>(
    pred: &'a WithGrad<Tensor<TensorFloat, N, D>>,
    target: &'a Tensor<TensorFloat, N, D>,
) -> Option<(
    TensorFloat,
    Box<dyn Fn(TensorFloat) -> Tensor<TensorFloat, N, D> + 'a>,
)> {
    use super::array_from_slice;

    let len = pred.get_value().len();
    if len == 0 {
        // Early return None or zero loss + noop backward, whichever makes sense
        return None;
    }

    let result =
        super::block_on_gpu(run_mse_loss_shader(pred.get_value().data(), target.data())).ok()?;

    let back = move |grad: TensorFloat| {
        use tensor_optim::ConstTensorOps;

        #[allow(clippy::cast_precision_loss)]
        let grad_data: Vec<TensorFloat> = pred
            .get_value()
            .data()
            .iter()
            .zip(target.data().iter())
            .map(|(&x, &y)| {
                2.0 * grad * TensorFloat::from(x - y) / pred.get_value().data().len() as TensorFloat
            })
            .collect();
        Tensor::new(
            pred.get_value().shape_array(),
            &array_from_slice(&grad_data),
        )
    };

    Some((TensorFloat::from(result), Box::new(back)))
}

#[allow(clippy::unused_async)]
async fn run_mse_loss_shader(prediction: &[f32], target: &[f32]) -> Result<f32, GpuFailure> {
    let device = &GPU_CONTEXT.device;
    let queue = &GPU_CONTEXT.queue;

    let len = prediction.len();
    assert_eq!(len, target.len());
    let buffer_size = (core::mem::size_of_val(prediction)) as u64;

    // === Buffers ===
    let pred_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("prediction"),
        contents: slice_to_bytes(prediction),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let target_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("target"),
        contents: slice_to_bytes(target),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let loss_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("loss"),
        size: buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bind_group_layout = &*MSE_LOSS_BIND_GROUP_LAYOUT;

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mse_loss_bind_group"),
        layout: bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: pred_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: target_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: loss_buf.as_entire_binding(),
            },
        ],
    });

    let pipeline = &*MSE_LOSS_PIPELINE;

    // === Dispatch ===
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("mse_loss_encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("mse_loss_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(u32::try_from(len).unwrap().div_ceil(64), 1, 1);
    }

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mse_staging"),
        size: buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&loss_buf, 0, &staging, 0, buffer_size);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);

    // atomic flag for mapping completion
    let ready = Arc::new(AtomicBool::new(false));
    {
        let ready_clone = Arc::clone(&ready);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            assert!(result.is_ok());
            ready_clone.store(true, Ordering::Release);
        });
    }

    // poll device until mapping completes (non-blocking)
    let _ = GPU_CONTEXT.device.poll(wgpu::PollType::Wait);

    let view = slice.get_mapped_range();
    let loss_terms: &[f32] = slice_from_bytes::<f32>(&view)?;
    let total_loss = loss_terms.iter().sum::<f32>() / len as f32;
    drop(view);
    staging.unmap();

    Ok(total_loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::approx::{approx_eq, ApproxEquality, RelativeEq};
    use crate::nn::tensors::{Tensor, WithGrad};

    #[test]
    fn wgpu_mse_loss_forward_basic() {
        let pred_data = [0.0, 0.5, 1.0];
        let target_data = [0.0, 1.0, 1.0];

        let pred = WithGrad::new(Tensor::new(&[3], &pred_data));
        let target = Tensor::new(&[3], &target_data);

        let (loss, _) = wgpu_mse_loss(&pred, &target).expect("wgpu_mse_loss failed");

        // expected MSE: mean((pred - target)^2) = ((0-0)^2 + (0.5-1)^2 + (1-1)^2) / 3 = (0 + 0.25 + 0) / 3 = 0.0833
        assert_ne!(loss.approx_eq(&0.0833333), ApproxEquality::Scarce);
    }

    #[test]
    fn wgpu_mse_loss_backward_gradient_shape_and_values() {
        let pred_data = [0.0, 0.5, 1.0];
        let target_data = [0.0, 1.0, 1.0];

        let pred = WithGrad::new(Tensor::new(&[3], &pred_data));
        let target = Tensor::new(&[3], &target_data);

        let (_, back_fn) = wgpu_mse_loss(&pred, &target).expect("wgpu_mse_loss failed");

        // Backward with grad=1.0 (typical)
        let grad_output = 1.0;

        let grad_tensor = back_fn(grad_output);

        // Gradient shape must match prediction shape
        assert_eq!(grad_tensor.shape(), pred.get_value().shape());

        // Gradients are 2*(pred - target)/N, here N=3
        let expected_grads: Vec<TensorFloat> = pred_data
            .iter()
            .zip(target_data.iter())
            .map(|(&p, &t)| 2.0 * (p - t) / 3.0)
            .collect();

        for (g, e) in grad_tensor.data().iter().zip(expected_grads.iter()) {
            assert!(approx_eq(g, e));
        }
    }

    #[test]
    fn wgpu_mse_loss_empty_input() {
        let pred = WithGrad::new(Tensor::new(&[0], &[]));
        let target = Tensor::new(&[0], &[]);

        // This might fail depending on implementation - let's check it gracefully handles or errors
        let result = wgpu_mse_loss(&pred, &target);

        // We accept None or a valid loss (0), but must not panic
        assert!(result.is_some() || result.is_none());
    }
}
