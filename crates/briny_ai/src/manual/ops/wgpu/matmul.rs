use super::{GPU_CONTEXT, GpuFailure, MATMUL_BIND_GROUP_LAYOUT, MATMUL_PIPELINE};
use crate::manual::{
    TensorFloat,
    tensors::{Tensor, WithGrad},
};
use alloc::{boxed::Box, sync::Arc, vec, vec::Vec};
use briny::raw::casting::{slice_from_bytes, slice_to_bytes, to_bytes};
use core::sync::atomic::{AtomicBool, Ordering};
use tensor_optim::TensorOps;
use wgpu::util::DeviceExt;

/// Performs matrix multiplication on the GPU using a precompiled WGSL shader.
///
/// Accepts two input tensors `a` (shape `[m, k]`) and `b` (shape `[k, n]`)
/// and computes the output `c = a @ b` on the GPU, returning a Tensor of shape `[m, n]`.
///
/// Returns a closure for computing the backward gradient (on the CPU).
///
/// # Returns
///
/// - `Some((Tensor, backward_fn))` on success
/// - `None` if shape mismatch or GPU failure occurs
///
/// # Notes
///
/// - Input data is cast from `TensorFloat` → `f32` for GPU
/// - Output is cast back from `f32` → `TensorFloat`
#[must_use]
#[cfg(feature = "dyntensor")]
pub fn wgpu_matmul(
    a: &WithGrad<Tensor<TensorFloat>>,
    b: &WithGrad<Tensor<TensorFloat>>,
) -> Option<(
    Tensor<TensorFloat>,
    Box<dyn Fn(Tensor<TensorFloat>) -> (Tensor<TensorFloat>, Tensor<TensorFloat>)>,
)> {
    let (m, k) = (a.get_value().shape()[0], a.get_value().shape()[1]);
    let (k2, n) = (b.get_value().shape()[0], b.get_value().shape()[1]);
    if k != k2 {
        return None;
    }

    let a_data: Vec<f32> = a.get_value().data().iter().map(|&v| v as f32).collect();
    let b_data: Vec<f32> = b.get_value().data().iter().map(|&v| v as f32).collect();

    let output_size = m * n;
    let mut output_data = vec![0.0f32; output_size];

    super::block_on_gpu(run_matmul_shader(
        &a_data,
        &b_data,
        &mut output_data,
        m,
        k,
        n,
    ))
    .ok()?;

    let output_data_float: Vec<TensorFloat> =
        output_data.into_iter().map(TensorFloat::from).collect();
    let out_tensor = Tensor::new(&[m, n], &output_data_float);

    let a_val = a.get_value().clone();
    let b_val = b.get_value().clone();

    let back = move |grad: Tensor<TensorFloat>| {
        let grad_data: Vec<f32> = grad.data().iter().map(|&v| v as f32).collect();
        let a_data: Vec<f32> = a_val.data().iter().map(|&v| v as f32).collect();
        let b_data: Vec<f32> = b_val.data().iter().map(|&v| v as f32).collect();

        // transpose B → Bᵀ for dA = grad @ Bᵀ
        let mut b_t_data = vec![0.0f32; b_val.data().len()];
        for i in 0..n {
            for j in 0..k {
                b_t_data[i * k + j] = b_data[j * n + i];
            }
        }

        // transpose A → Aᵀ for dB = Aᵀ @ grad
        let mut a_t_data = vec![0.0f32; a_val.data().len()];
        for i in 0..k {
            for j in 0..m {
                a_t_data[i * m + j] = a_data[j * k + i];
            }
        }

        let mut da_f32 = vec![0.0f32; m * k];
        let mut db_f32 = vec![0.0f32; k * n];

        // ∂L/∂A = grad @ Bᵀ
        let _ = super::block_on_gpu(run_matmul_shader(
            &grad_data,
            &b_t_data,
            &mut da_f32,
            m,
            n,
            k, // note: Bᵀ shape = [n x k] → input is [m x n] × [n x k]
        ));

        // ∂L/∂B = Aᵀ @ grad
        let _ = super::block_on_gpu(run_matmul_shader(
            &a_t_data,
            &grad_data,
            &mut db_f32,
            k,
            m,
            n, // note: Aᵀ shape = [k x m] → input is [k x m] × [m x n]
        ));

        let da = Tensor::new(
            &[m, k],
            &da_f32
                .into_iter()
                .map(TensorFloat::from)
                .collect::<Vec<TensorFloat>>(),
        );
        let db = Tensor::new(
            &[k, n],
            &db_f32
                .into_iter()
                .map(TensorFloat::from)
                .collect::<Vec<TensorFloat>>(),
        );

        (da, db)
    };

    Some((out_tensor, Box::new(back)))
}
/// Performs matrix multiplication on the GPU using a precompiled WGSL shader.
///
/// Accepts two input tensors `a` (shape `[m, k]`) and `b` (shape `[k, n]`)
/// and computes the output `c = a @ b` on the GPU, returning a Tensor of shape `[m, n]`.
///
/// Returns a closure for computing the backward gradient (on the CPU).
///
/// # Returns
///
/// - `Some((Tensor, backward_fn))` on success
/// - `None` if shape mismatch or GPU failure occurs
///
/// # Notes
///
/// - Input data is cast from `TensorFloat` → `f32` for GPU
/// - Output is cast back from `f32` → `TensorFloat`
#[must_use]
#[cfg(not(feature = "dyntensor"))]
pub fn wgpu_matmul<'a, const A: usize, const B: usize, const OUT: usize>(
    a: &'a WithGrad<Tensor<TensorFloat, A, 2>>,
    b: &'a WithGrad<Tensor<TensorFloat, B, 2>>,
) -> Option<(
    Tensor<TensorFloat, OUT, 2>,
    Box<
        dyn Fn(
            Tensor<TensorFloat, OUT, 2>,
        ) -> (Tensor<TensorFloat, A, 2>, Tensor<TensorFloat, B, 2>),
    >,
)> {
    use super::array_from_slice;

    let (m, k) = (a.get_value().shape()[0], a.get_value().shape()[1]);
    let (k2, n) = (b.get_value().shape()[0], b.get_value().shape()[1]);
    if k != k2 {
        return None;
    }

    let a_data: Vec<f32> = a.get_value().data().iter().map(|&v| v as f32).collect();
    let b_data: Vec<f32> = b.get_value().data().iter().map(|&v| v as f32).collect();

    let output_size = m * n;
    let mut output_data = vec![0.0f32; output_size];

    super::block_on_gpu(run_matmul_shader(
        &a_data,
        &b_data,
        &mut output_data,
        m,
        k,
        n,
    ))
    .ok()?;

    let output_data_float: Vec<TensorFloat> =
        output_data.into_iter().map(TensorFloat::from).collect();
    let out_tensor = Tensor::new(&[m, n], &array_from_slice(&output_data_float));

    let a_val = a.get_value().clone();
    let b_val = b.get_value().clone();

    let back = move |grad: Tensor<TensorFloat, OUT, 2>| {
        let grad_data: Vec<f32> = grad.data().iter().map(|&v| v as f32).collect();
        let a_data: Vec<f32> = a_val.data().iter().map(|&v| v as f32).collect();
        let b_data: Vec<f32> = b_val.data().iter().map(|&v| v as f32).collect();

        // transpose B → Bᵀ for dA = grad @ Bᵀ
        let mut b_t_data = vec![0.0f32; b_val.data().len()];
        for i in 0..n {
            for j in 0..k {
                b_t_data[i * k + j] = b_data[j * n + i];
            }
        }

        // transpose A → Aᵀ for dB = Aᵀ @ grad
        let mut a_t_data = vec![0.0f32; a_val.data().len()];
        for i in 0..k {
            for j in 0..m {
                a_t_data[i * m + j] = a_data[j * k + i];
            }
        }

        let mut da_f32 = vec![0.0f32; m * k];
        let mut db_f32 = vec![0.0f32; k * n];

        // ∂L/∂A = grad @ Bᵀ
        let _ = super::block_on_gpu(run_matmul_shader(
            &grad_data,
            &b_t_data,
            &mut da_f32,
            m,
            n,
            k, // note: Bᵀ shape = [n x k] → input is [m x n] × [n x k]
        ));

        // ∂L/∂B = Aᵀ @ grad
        let _ = super::block_on_gpu(run_matmul_shader(
            &a_t_data,
            &grad_data,
            &mut db_f32,
            k,
            m,
            n, // note: Aᵀ shape = [k x m] → input is [k x m] × [m x n]
        ));

        let da = Tensor::new(
            &[m, k],
            &array_from_slice(
                &da_f32
                    .into_iter()
                    .map(TensorFloat::from)
                    .collect::<Vec<TensorFloat>>(),
            ),
        );
        let db = Tensor::new(
            &[k, n],
            &array_from_slice(
                &db_f32
                    .into_iter()
                    .map(TensorFloat::from)
                    .collect::<Vec<TensorFloat>>(),
            ),
        );

        (da, db)
    };

    Some((out_tensor, Box::new(back)))
}

async fn run_matmul_shader(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), GpuFailure> {
    let device = &GPU_CONTEXT.device;
    let queue = &GPU_CONTEXT.queue;

    let dims = [
        u32::try_from(m).unwrap(),
        u32::try_from(k).unwrap(),
        u32::try_from(n).unwrap(),
        0u32,
    ];
    let dims_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dims"),
        contents: to_bytes(&dims),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let a_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("A"),
        contents: slice_to_bytes(a),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let b_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("B"),
        contents: slice_to_bytes(b),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let c_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("C"),
        size: (out.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bind_group_layout = &*MATMUL_BIND_GROUP_LAYOUT;

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("matmul_bind_group"),
        layout: bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: dims_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: a_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: b_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: c_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline = &*MATMUL_PIPELINE;

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("matmul_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("matmul_pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(
            u32::try_from(n).unwrap().div_ceil(16),
            u32::try_from(m).unwrap().div_ceil(16),
            1,
        );
    }

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: (out.len() * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&c_buffer, 0, &staging_buffer, 0, (out.len() * 4) as u64);

    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);

    // atomic flag to track when mapping is complete
    let ready = Arc::new(AtomicBool::new(false));

    {
        let ready_clone = Arc::clone(&ready);
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            assert!(result.is_ok());
            ready_clone.store(true, Ordering::Release);
        });
    }

    // poll until the mapping is ready
    let _ = GPU_CONTEXT.device.poll(wgpu::PollType::Wait);

    let data = buffer_slice.get_mapped_range();
    out.copy_from_slice(slice_from_bytes::<f32>(&data)?);
    drop(data);
    staging_buffer.unmap();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manual::tensors::{Tensor, WithGrad};
    use approx::assert_abs_diff_eq;

    // Simple CPU matmul for checking results
    fn cpu_matmul(
        a: &[f64],
        a_shape: (usize, usize),
        b: &[f64],
        b_shape: (usize, usize),
    ) -> Vec<f64> {
        let (m, k) = a_shape;
        let (k2, n) = b_shape;
        assert_eq!(k, k2);
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for x in 0..k {
                    sum += a[i * k + x] * b[x * n + j];
                }
                out[i * n + j] = sum;
            }
        }
        out
    }

    #[test]
    fn wgpu_matmul_forward() {
        let a_data = [1.0, 2.0, 3.0, 4.0];
        let b_data = [5.0, 6.0, 7.0, 8.0];

        let a = WithGrad::new(Tensor::new(&[2, 2], &a_data));
        let b = WithGrad::new(Tensor::new(&[2, 2], &b_data));

        #[cfg(feature = "dyntensor")]
        let result = wgpu_matmul(&a, &b).expect("matmul failed");
        #[cfg(not(feature = "dyntensor"))]
        let result = wgpu_matmul::<4, 4, 4>(&a, &b).expect("matmul failed");
        let (out, _) = result;

        let expected = cpu_matmul(&a_data, (2, 2), &b_data, (2, 2));

        // Check shape
        assert_eq!(out.shape(), &[2, 2]);

        // Check values approximately equal (allow some floating error)
        for (v, e) in out.data().iter().zip(expected.iter()) {
            assert_abs_diff_eq!(v, e, epsilon = 1e-4);
        }
    }

    #[test]
    fn wgpu_matmul_backward_shapes() {
        let a_data = [1.0, 0.0, 0.0, 1.0];
        let b_data = [2.0, 3.0, 4.0, 5.0];

        let a = WithGrad::new(Tensor::new(&[2, 2], &a_data));
        let b = WithGrad::new(Tensor::new(&[2, 2], &b_data));

        let result = wgpu_matmul(&a, &b).expect("matmul failed");
        let (_, back_fn) = result;

        // Fake gradient tensor (same shape as output)
        let grad_data = [1.0, 1.0, 1.0, 1.0];
        let grad = Tensor::new(&[2, 2], &grad_data);

        let (d_a, d_b) = back_fn(grad);

        // Shapes of gradients must match inputs
        assert_eq!(d_a.shape(), a.get_value().shape());
        assert_eq!(d_b.shape(), b.get_value().shape());
    }

    #[test]
    fn wgpu_matmul_shape_mismatch() {
        let a = WithGrad::new(Tensor::new(&[2, 3], &[1.0; 6]));
        let b = WithGrad::new(Tensor::new(&[4, 2], &[1.0; 8]));

        // k dimension does not match (3 vs 4)

        #[cfg(feature = "dyntensor")]
        assert!(wgpu_matmul(&a, &b).is_none());
        #[cfg(not(feature = "dyntensor"))]
        assert!(wgpu_matmul::<6, 8, 4>(&a, &b).is_none());
    }
}
