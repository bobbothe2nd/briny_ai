use super::{
    GpuFailure, GPU_CONTEXT, MATMUL_2D_BIND_GROUP_LAYOUT, MATMUL_2D_PIPELINE,
    MATMUL_BIND_GROUP_LAYOUT, MATMUL_PIPELINE,
};
use crate::nn::{
    tensors::{Tensor, WithGrad},
    TensorFloat,
};
use alloc::{boxed::Box, vec, vec::Vec};
use briny::raw::{slice_from_bytes, slice_to_bytes, to_bytes};
use tensor_optim::TensorOps;
use wgpu::util::DeviceExt;

fn make_tensor_info_buffer(device: &wgpu::Device, shape: &[u32], strides: &[u32]) -> wgpu::Buffer {
    // WGSL std140 rules: each vec4<u32> is 16 bytes
    // We pack shape[i], stride[i], and pad to 16 bytes per 4 elements.

    // We'll allocate 8 vec4 slots (8 * 16 bytes = 128 bytes) for uniform alignment.
    // Unused entries are zeroed.

    let mut data = [0; 128];
    for i in 0..shape.len().min(8) {
        let offset = i * 16;
        let s_bytes = shape[i].to_le_bytes();
        let st_bytes = strides[i].to_le_bytes();
        data[offset..offset + 4].copy_from_slice(&s_bytes);
        data[offset + 4..offset + 8].copy_from_slice(&st_bytes);
        // remaining 8 bytes are padding
    }

    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("TensorInfo"),
        contents: &data,
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

fn make_contraction_info_buffer(device: &wgpu::Device, contract_axes: &[u32]) -> wgpu::Buffer {
    // Still padded to multiple of 16 bytes
    let mut data = [0; 16];
    data[..4].copy_from_slice(&(contract_axes.len() as u32).to_le_bytes());
    for (i, &ax) in contract_axes.iter().enumerate().take(3) {
        data[4 + i * 4..4 + (i + 1) * 4].copy_from_slice(&ax.to_le_bytes());
    }

    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ContractionInfo"),
        contents: &data,
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

fn compute_strides(shape: &[usize]) -> Vec<u32> {
    let mut strides = vec![0; shape.len()];
    let mut stride = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = stride;
        stride *= shape[i] as u32;
    }
    strides
}

fn transpose_last_two_axes(data: &[f32], shape: &[u32]) -> Vec<f32> {
    let rank = shape.len();
    assert!(
        rank >= 2,
        "Tensor must have at least 2 axes to transpose last two"
    );

    let last = shape[rank - 1] as usize;
    let second_last = shape[rank - 2] as usize;

    // compute total number of elements
    let total = data.len();
    assert_eq!(total, shape.iter().map(|&x| x as usize).product::<usize>());

    // strides for original tensor
    let mut strides = vec![1; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as usize;
    }

    // output array
    let mut out = vec![0.0f32; total];

    // iterate over all indices except last two dims
    let outer_count = total / (second_last * last);
    for outer in 0..outer_count {
        for i in 0..second_last {
            for j in 0..last {
                let in_idx = outer * second_last * last + i * last + j;
                let out_idx = outer * second_last * last + j * second_last + i;
                out[out_idx] = data[in_idx];
            }
        }
    }
    out
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
///
/// # Panics
///
/// This function will panic if the two tensors are not of equal
/// rank with at least two dimensions and no more than 8.
#[must_use]
#[cfg(feature = "dyntensor")]
pub fn wgpu_matmul(
    a: &WithGrad<Tensor<TensorFloat>>,
    b: &WithGrad<Tensor<TensorFloat>>,
) -> Option<(
    Tensor<TensorFloat>,
    Box<dyn Fn(Tensor<TensorFloat>) -> (Tensor<TensorFloat>, Tensor<TensorFloat>)>,
)> {
    assert!(2 <= a.get_value().shape().len() && a.get_value().shape().len() <= 8);
    assert_eq!(a.get_value().shape().len(), b.get_value().shape().len());

    let (m, k) = (a.get_value().shape()[0], a.get_value().shape()[1]);
    let (k2, n) = (b.get_value().shape()[0], b.get_value().shape()[1]);
    if k != k2 {
        return None;
    }

    let a_data: Vec<f32> = a.get_value().data().iter().map(|&v| v as f32).collect();
    let b_data: Vec<f32> = b.get_value().data().iter().map(|&v| v as f32).collect();

    let a_shape = a.get_value().shape(); // e.g., [2, 3]
    let b_shape = b.get_value().shape(); // e.g., [3, 4]

    assert_eq!(a_shape[a_shape.len() - 1], b_shape[b_shape.len() - 2]); // contraction axis k

    // output shape is all of A’s axes except the last, plus B’s last axis
    let mut out_shape = a_shape[..a_shape.len() - 1].to_vec(); // [..., m]
    out_shape.push(*b_shape.last().unwrap()); // [..., m, n]

    let output_size = m * n;
    let mut output_data = vec![0.0; output_size];

    // compute strides for all tensors
    let a_strides = compute_strides(a.get_value().shape());
    let b_strides = compute_strides(b.get_value().shape());
    let c_strides = compute_strides(&out_shape);

    // forward pass
    dispatch_matmul(
        &a_data,
        &b_data,
        &mut output_data,
        &a.get_value()
            .shape()
            .iter()
            .map(|&val| val as u32)
            .collect::<Vec<_>>(),
        &a_strides,
        &b.get_value()
            .shape()
            .iter()
            .map(|&val| val as u32)
            .collect::<Vec<_>>(),
        &b_strides,
        &out_shape.iter().map(|&val| val as u32).collect::<Vec<_>>(),
        &c_strides,
        &[(a.get_value().shape().len() - 1) as u32], // contract last axis
    )
    .ok()?;

    let output_data_float: Vec<TensorFloat> =
        output_data.into_iter().map(TensorFloat::from).collect();
    let out_tensor = Tensor::new(&out_shape, &output_data_float);

    let a_val = a.get_value().clone();
    let b_val = b.get_value().clone();

    let back = move |grad: Tensor<TensorFloat>| {
        let grad_data: Vec<f32> = grad.data().iter().map(|&v| v as f32).collect();
        let a_data: Vec<f32> = a_val.data().iter().map(|&v| v as f32).collect();
        let b_data: Vec<f32> = b_val.data().iter().map(|&v| v as f32).collect();

        let mut da_f32 = vec![0.0f32; m * k];
        let mut db_f32 = vec![0.0f32; k * n];

        let a_strides = compute_strides(a_val.shape());
        let b_strides = compute_strides(b_val.shape());
        let grad_strides = compute_strides(grad.shape());

        // ∂L/∂A = grad @ Bᵀ
        let b_t_data = transpose_last_two_axes(
            &b_data,
            &b_val
                .shape()
                .iter()
                .map(|&val| val as u32)
                .collect::<Vec<_>>(),
        );
        let _ = dispatch_matmul(
            &grad_data,
            &b_t_data,
            &mut da_f32,
            &grad
                .shape()
                .iter()
                .map(|&val| val as u32)
                .collect::<Vec<_>>(),
            &grad_strides,
            &b_val
                .shape()
                .iter()
                .map(|&val| val as u32)
                .collect::<Vec<_>>(),
            &b_strides,
            &a_val
                .shape()
                .iter()
                .map(|&val| val as u32)
                .collect::<Vec<_>>(),
            &a_strides,
            &[(b_val.shape().len() - 1) as u32],
        );

        // ∂L/∂B = Aᵀ @ grad
        let a_t_data = transpose_last_two_axes(
            &a_data,
            &a_val
                .shape()
                .iter()
                .map(|&val| val as u32)
                .collect::<Vec<_>>(),
        );
        let _ = dispatch_matmul(
            &a_t_data,
            &grad_data,
            &mut db_f32,
            &a_val
                .shape()
                .iter()
                .map(|&val| val as u32)
                .collect::<Vec<_>>(),
            &a_strides,
            &grad
                .shape()
                .iter()
                .map(|&val| val as u32)
                .collect::<Vec<_>>(),
            &grad_strides,
            &b_val
                .shape()
                .iter()
                .map(|&val| val as u32)
                .collect::<Vec<_>>(),
            &b_strides,
            &[(a_val.shape().len() - 1) as u32],
        );

        let da = Tensor::new(
            a_val.shape(),
            &da_f32
                .iter()
                .map(|&x| TensorFloat::from(x))
                .collect::<Vec<_>>(),
        );
        let db = Tensor::new(
            b_val.shape(),
            &db_f32
                .iter()
                .map(|&x| TensorFloat::from(x))
                .collect::<Vec<_>>(),
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
///
/// # Panics
///
/// This function will panic if the two tensors are not of equal
/// rank with at least two dimensions and no more than 8.
#[must_use]
#[cfg(not(feature = "dyntensor"))]
pub fn wgpu_matmul<'a, const A: usize, const B: usize, const OUT: usize, const D: usize>(
    a: &'a WithGrad<Tensor<TensorFloat, A, D>>,
    b: &'a WithGrad<Tensor<TensorFloat, B, D>>,
) -> Option<(
    Tensor<TensorFloat, OUT, D>,
    Box<
        dyn Fn(
            Tensor<TensorFloat, OUT, D>,
        ) -> (Tensor<TensorFloat, A, D>, Tensor<TensorFloat, B, D>),
    >,
)> {
    use super::array_from_slice;

    const {
        assert!(2 <= D && D <= 8);
    }

    let (m, k) = (a.get_value().shape()[0], a.get_value().shape()[1]);
    let (k2, n) = (b.get_value().shape()[0], b.get_value().shape()[1]);
    if k != k2 {
        return None;
    }

    let a_data: Vec<f32> = a.get_value().data().iter().map(|&v| v as f32).collect();
    let b_data: Vec<f32> = b.get_value().data().iter().map(|&v| v as f32).collect();

    let a_shape = a.get_value().shape(); // e.g., [2, 3]
    let b_shape = b.get_value().shape(); // e.g., [3, 4]

    assert_eq!(a_shape[a_shape.len() - 1], b_shape[b_shape.len() - 2]); // contraction axis k

    // output shape is all of A’s axes except the last, plus B’s last axis
    let mut out_shape = a_shape[..a_shape.len() - 1].to_vec(); // [..., m]
    out_shape.push(*b_shape.last().unwrap()); // [..., m, n]

    let output_size = m * n;
    let mut output_data = vec![0.0; output_size];

    // compute strides for all tensors
    let a_strides = compute_strides(a.get_value().shape());
    let b_strides = compute_strides(b.get_value().shape());
    let c_strides = compute_strides(&out_shape);

    // forward pass
    dispatch_matmul(
        &a_data,
        &b_data,
        &mut output_data,
        &a.get_value()
            .shape()
            .iter()
            .map(|&val| val as u32)
            .collect::<Vec<_>>(),
        &a_strides,
        &b.get_value()
            .shape()
            .iter()
            .map(|&val| val as u32)
            .collect::<Vec<_>>(),
        &b_strides,
        &out_shape.iter().map(|&val| val as u32).collect::<Vec<_>>(),
        &c_strides,
        &[(a.get_value().shape().len() - 1) as u32], // contract last axis
    )
    .ok()?;

    let output_data_float: Vec<TensorFloat> =
        output_data.into_iter().map(TensorFloat::from).collect();
    let out_tensor = Tensor::new(
        &array_from_slice(&out_shape),
        &array_from_slice(&output_data_float),
    );

    let a_val = a.get_value().clone();
    let b_val = b.get_value().clone();

    let back = move |grad: Tensor<TensorFloat, OUT, D>| {
        let grad_data: Vec<f32> = grad.data().iter().map(|&v| v as f32).collect();
        let a_data: Vec<f32> = a_val.data().iter().map(|&v| v as f32).collect();
        let b_data: Vec<f32> = b_val.data().iter().map(|&v| v as f32).collect();

        let mut da_f32 = vec![0.0f32; m * k];
        let mut db_f32 = vec![0.0f32; k * n];

        let a_strides = compute_strides(a_val.shape());
        let b_strides = compute_strides(b_val.shape());
        let grad_strides = compute_strides(grad.shape());

        // ∂L/∂A = grad @ Bᵀ
        let b_t_data = transpose_last_two_axes(
            &b_data,
            &b_val
                .shape()
                .iter()
                .map(|&val| val as u32)
                .collect::<Vec<_>>(),
        );
        let _ = dispatch_matmul(
            &grad_data,
            &b_t_data,
            &mut da_f32,
            &grad
                .shape()
                .iter()
                .map(|&val| val as u32)
                .collect::<Vec<_>>(),
            &grad_strides,
            &b_val
                .shape()
                .iter()
                .map(|&val| val as u32)
                .collect::<Vec<_>>(),
            &b_strides,
            &a_val
                .shape()
                .iter()
                .map(|&val| val as u32)
                .collect::<Vec<_>>(),
            &a_strides,
            &[(b_val.shape().len() - 1) as u32],
        );

        // ∂L/∂B = Aᵀ @ grad
        let a_t_data = transpose_last_two_axes(
            &a_data,
            &a_val
                .shape()
                .iter()
                .map(|&val| val as u32)
                .collect::<Vec<_>>(),
        );
        let _ = dispatch_matmul(
            &a_t_data,
            &grad_data,
            &mut db_f32,
            &a_val
                .shape()
                .iter()
                .map(|&val| val as u32)
                .collect::<Vec<_>>(),
            &a_strides,
            &grad
                .shape()
                .iter()
                .map(|&val| val as u32)
                .collect::<Vec<_>>(),
            &grad_strides,
            &b_val
                .shape()
                .iter()
                .map(|&val| val as u32)
                .collect::<Vec<_>>(),
            &b_strides,
            &[(a_val.shape().len() - 1) as u32],
        );

        let da = Tensor::new(
            &array_from_slice(a_val.shape()),
            &array_from_slice(
                &da_f32
                    .iter()
                    .map(|&x| TensorFloat::from(x))
                    .collect::<Vec<_>>(),
            ),
        );
        let db = Tensor::new(
            &array_from_slice(b_val.shape()),
            &array_from_slice(
                &db_f32
                    .iter()
                    .map(|&x| TensorFloat::from(x))
                    .collect::<Vec<_>>(),
            ),
        );

        (da, db)
    };

    Some((out_tensor, Box::new(back)))
}

#[allow(clippy::too_many_arguments)]
fn dispatch_matmul(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    a_shape: &[u32],
    a_strides: &[u32],
    b_shape: &[u32],
    b_strides: &[u32],
    c_shape: &[u32],
    c_strides: &[u32],
    contract_axes: &[u32],
) -> Result<(), GpuFailure> {
    if a_shape.len() == 2 && b_shape.len() == 2 {
        // 2D matmul
        super::block_on_gpu(run_matmul_shader(
            a,
            b,
            out,
            a_shape[0] as usize,
            a_shape[1] as usize,
            b_shape[1] as usize,
        ))
    } else {
        // generic matmul
        super::block_on_gpu(run_matmul_shader_generic(
            a,
            b,
            out,
            a_shape,
            a_strides,
            b_shape,
            b_strides,
            c_shape,
            c_strides,
            contract_axes,
        ))
    }
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

    let bind_group_layout = &*MATMUL_2D_BIND_GROUP_LAYOUT;

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("matmul2_bind_group"),
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

    let pipeline = &*MATMUL_2D_PIPELINE;

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("matmul2_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("matmul2_pass"),
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

    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        assert!(result.is_ok());
    });

    // poll until the mapping is ready
    let _ = GPU_CONTEXT.device.poll(wgpu::PollType::Wait);

    let data = buffer_slice.get_mapped_range();
    out.copy_from_slice(slice_from_bytes::<f32>(&data)?);
    drop(data);
    staging_buffer.unmap();

    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn run_matmul_shader_generic(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    a_shape: &[u32],
    a_strides: &[u32],
    b_shape: &[u32],
    b_strides: &[u32],
    c_shape: &[u32],
    c_strides: &[u32],
    contract_axes: &[u32],
) -> Result<(), GpuFailure> {
    use wgpu::util::DeviceExt;

    let device = &GPU_CONTEXT.device;
    let queue = &GPU_CONTEXT.queue;

    // create uniform buffers
    let a_info_buffer = make_tensor_info_buffer(device, a_shape, a_strides);
    let b_info_buffer = make_tensor_info_buffer(device, b_shape, b_strides);
    let c_info_buffer = make_tensor_info_buffer(device, c_shape, c_strides);
    let contract_buffer = make_contraction_info_buffer(device, contract_axes);

    // create storage buffers
    let a_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("A_data"),
        contents: slice_to_bytes(a),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let b_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("B_data"),
        contents: slice_to_bytes(b),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let c_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("C_data"),
        size: (out.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("matmul_bind_group"),
        layout: &MATMUL_BIND_GROUP_LAYOUT,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_info_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_info_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: c_info_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: contract_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: a_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: b_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: c_buffer.as_entire_binding(),
            },
        ],
    });

    // compute dispatch
    let pipeline = &*MATMUL_PIPELINE;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("matmul_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("matmul_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        // Compute total number of output elements and figure workgroup grid
        // For now we assume the last two dims form the matrix product output
        let rank = c_shape.len();
        assert!(rank >= 2);
        let m = c_shape[rank - 2];
        let n = c_shape[rank - 1];

        pass.dispatch_workgroups(n.div_ceil(16), m.div_ceil(16), 1);
    }

    // staging buffer
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("matmul_staging"),
        size: (out.len() * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&c_buffer, 0, &staging, 0, (out.len() * 4) as u64);
    queue.submit(Some(encoder.finish()));

    // read results
    let slice = staging.slice(..);

    slice.map_async(wgpu::MapMode::Read, move |r| {
        assert!(r.is_ok());
    });

    let _ = device.poll(wgpu::PollType::Wait);
    let data = slice.get_mapped_range();
    out.copy_from_slice(slice_from_bytes::<f32>(&data)?);
    drop(data);
    staging.unmap();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::approx::approx_eq;
    use crate::nn::tensors::{Tensor, WithGrad};

    // Simple CPU matmul for checking results
    fn cpu_matmul(
        a: &[TensorFloat],
        a_shape: (usize, usize),
        b: &[TensorFloat],
        b_shape: (usize, usize),
    ) -> Vec<TensorFloat> {
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
        let result = wgpu_matmul::<4, 4, 4, 2>(&a, &b).expect("matmul failed");
        let (out, _) = result;

        let expected = cpu_matmul(&a_data, (2, 2), &b_data, (2, 2));

        // Check shape
        assert_eq!(out.shape(), &[2, 2]);

        // Check values approximately equal (allow some floating error)
        for (&v, &e) in out.data().iter().zip(expected.iter()) {
            assert!(approx_eq(v, e));
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
        assert!(wgpu_matmul::<6, 8, 4, 2>(&a, &b).is_none());
    }
}
