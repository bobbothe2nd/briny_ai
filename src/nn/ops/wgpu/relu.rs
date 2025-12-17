use super::{vec, GpuFailure, GPU_CONTEXT, RELU_BIND_GROUP_LAYOUT, RELU_PIPELINE};
use crate::nn::{
    tensors::{Tensor, TensorGrad, TensorOps, WithGrad},
    TensorFloat,
};
use alloc::{boxed::Box, sync::Arc};
use briny::raw::{slice_from_bytes, slice_to_bytes};
use core::sync::atomic::{AtomicBool, Ordering};
use wgpu::util::DeviceExt;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Performs elementwise `ReLU` (`max(x, 0)`) on the GPU.
///
/// Accepts an input tensor and applies `ReLU` in-place using a compute shader.
/// Returns a new Tensor and a closure for backpropagation that masks negative inputs.
///
/// # Returns
///
/// - `Some((Tensor, backward_fn))` on success
/// - `None` on GPU execution failure
///
/// # Notes
///
/// - Uses f32 precision on GPU
/// - Output and backward gradient are returned in `TensorFloat` for integration
#[must_use]
#[cfg(feature = "dyntensor")]
pub fn wgpu_relu(
    input: &WithGrad<Tensor<TensorFloat>>,
) -> Option<(
    Tensor<TensorFloat>,
    Box<dyn Fn(Tensor<TensorFloat>) -> Tensor<TensorFloat> + '_>,
)> {
    if input.get_value().len() == 0 {
        return None;
    }

    let data: Vec<f32> = input.get_value().data().iter().map(|&x| x as f32).collect();
    let mut output = vec![0.0f32; data.len()];

    let result = super::block_on_gpu(run_relu_shader(&data, &mut output));
    if result.is_err() {
        return None;
    }

    let output_tensor = Tensor::new(
        input.get_value().shape(),
        &{
            #[cfg(feature = "f64")]
            {
                output
                    .into_iter()
                    .map(TensorFloat::from)
            }
            #[cfg(not(feature = "f64"))]
            {
                output
                    .into_iter()
            }
        }.collect::<Vec<TensorFloat>>(),
    );
    let back = move |grad: Tensor<TensorFloat>| {
        let grad_data = grad
            .data()
            .iter()
            .zip(input.get_value().data().iter())
            .map(|(&g, &x)| if x > 0.0 { g } else { 0.0 })
            .collect::<Vec<TensorFloat>>();
        Tensor::new(input.get_value().shape(), &grad_data)
    };

    Some((output_tensor, Box::new(back)))
}
/// Performs elementwise `ReLU` (`max(x, 0)`) on the GPU.
///
/// Accepts an input tensor and applies `ReLU` in-place using a compute shader.
/// Returns a new Tensor and a closure for backpropagation that masks negative inputs.
///
/// # Returns
///
/// - `Some((Tensor, backward_fn))` on success
/// - `None` on GPU execution failure
///
/// # Notes
///
/// - Uses f32 precision on GPU
/// - Output and backward gradient are returned in `TensorFloat` for integration
#[must_use]
#[cfg(not(feature = "dyntensor"))]
pub fn wgpu_relu<const N: usize, const D: usize>(
    input: &WithGrad<Tensor<TensorFloat, N, D>>,
) -> Option<(
    Tensor<TensorFloat, N, D>,
    Box<dyn Fn(Tensor<TensorFloat, N, D>) -> Tensor<TensorFloat, N, D> + '_>,
)> {
    use super::array_from_slice;
    use tensor_optim::ConstTensorOps;

    if input.get_value().len() == 0 {
        return None;
    }

    let data: Vec<f32> = input.get_value().data().iter().map(|&x| x as f32).collect();
    let mut output = vec![0.0f32; data.len()];

    let result = super::block_on_gpu(run_relu_shader(&data, &mut output));
    if result.is_err() {
        return None;
    }

    let output_tensor = Tensor::new(
        input.get_value().shape_array(),
        &array_from_slice(
            &{
                #[cfg(feature = "f64")]
                {
                    output
                        .into_iter()
                        .map(TensorFloat::from)
                }
                #[cfg(not(feature = "f64"))]
                {
                    output
                        .into_iter()
                }
            }.collect::<Vec<TensorFloat>>(),
        ),
    );

    let in_val = input.get_value().clone();

    let back = move |grad: Tensor<TensorFloat, N, D>| {
        let grad_data = grad
            .data()
            .iter()
            .zip(in_val.data().iter())
            .map(|(&g, &x)| if x > 0.0 { g } else { 0.0 })
            .collect::<Vec<TensorFloat>>();
        Tensor::new(in_val.shape_array(), &array_from_slice(&grad_data))
    };

    Some((output_tensor, Box::new(back)))
}

async fn run_relu_shader(input: &[f32], output: &mut [f32]) -> Result<(), GpuFailure> {
    assert_eq!(output.len() % 4, 0);

    let device = &GPU_CONTEXT.device;
    let queue = &GPU_CONTEXT.queue;

    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input"),
        contents: slice_to_bytes(input),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: (output.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bind_layout = &*RELU_BIND_GROUP_LAYOUT;

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("relu_bind_group"),
        layout: bind_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buf.as_entire_binding(),
            },
        ],
    });

    let pipeline = &*RELU_PIPELINE;

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("relu_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("relu_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(u32::try_from(input.len()).unwrap().div_ceil(64), 1, 1);
    }

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("relu_staging"),
        size: (output.len() * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, (output.len() * 4) as u64);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);

    // atomic flag for map readiness
    let ready = Arc::new(AtomicBool::new(false));
    {
        let ready_clone = Arc::clone(&ready);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            assert!(result.is_ok());
            ready_clone.store(true, Ordering::Release);
        });
    }

    // poll device until mapping is ready (non-blocking)
    let _ = GPU_CONTEXT.device.poll(wgpu::PollType::Wait);

    let data = staging.slice(..).get_mapped_range();
    output.copy_from_slice(slice_from_bytes::<f32>(&data)?);
    drop(data);
    staging.unmap();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::approx::approx_eq;
    use crate::nn::ops::wgpu::array_from_slice;
    use crate::nn::tensors::{Tensor, WithGrad};

    #[test]
    fn wgpu_relu_basic() {
        let mut input_data = vec![-1.0, 0.0, 2.5, -3.3, 4.0];
        input_data.extend_from_slice(&[0.0, 0.0, 0.0]); // padding zeros

        let input_tensor = Tensor::new(&[input_data.len()], &array_from_slice::<_, 8>(&input_data));
        let input = WithGrad::new(input_tensor);

        let (output, _) = wgpu_relu(&input).expect("wgpu_relu failed");

        // Expected output: apply relu only on original elements
        let expected = vec![0.0, 0.0, 2.5, 0.0, 4.0, 0.0, 0.0, 0.0];

        assert_eq!(output.shape(), &[input_data.len()]);
        assert!(approx_eq(output.data(), expected.as_slice()));
    }

    #[test]
    fn wgpu_relu_backward() {
        let input_data = [-1.0, 2.0, 0.0, 3.5];
        let input_tensor = Tensor::new(&[input_data.len()], &input_data);
        let input = WithGrad::new(input_tensor);

        let (_, back_fn) = wgpu_relu(&input).expect("wgpu_relu failed");

        // Gradient from upstream is all ones
        let upstream_grad = Tensor::new(&[input_data.len()], &[1.0, 1.0, 1.0, 1.0]);

        let grad = back_fn(upstream_grad);

        // Backprop zeros out grad where input <= 0
        let expected_grad = vec![0.0, 1.0, 0.0, 1.0];
        assert_eq!(grad.shape(), &[input_data.len()]);
        assert!(approx_eq(grad.data(), expected_grad.as_slice()));
    }

    #[test]
    fn wgpu_relu_empty_input() {
        let input_tensor = Tensor::new(&[0], &[]);
        let input = WithGrad::new(input_tensor);

        let result = wgpu_relu(&input);

        // Should handle empty input gracefully, either by returning None or a tensor with zero length
        match result {
            Some((output, _)) => {
                assert_eq!(output.len(), 0);
            }
            None => {}
        }
    }
}
