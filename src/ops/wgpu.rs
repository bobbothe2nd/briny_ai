//! GPU-accelerated tensor operations using WGPU.
//!
//! This module implements high-performance compute kernels on the GPU using WGPU + WGSL.
//! It handles GPU context initialization, shader precompilation (via `lazy_static`),
//! and compute dispatch for key neural network operations:
//!
//! - `matmul` — matrix multiplication
//! - `relu` — ReLU activation
//! - `mse_loss` — mean squared error loss
//! - `sgd` — stochastic gradient descent update
//!
//! All shaders are compiled and cached once at runtime. Tensor data is copied
//! to the GPU for compute and returned as f64 to integrate with the rest of the framework.
//!
//! Most functions return both forward results and backward closures for autograd.

use crate::tensors::{Tensor, WithGrad, Ten64};
use wgpu::util::DeviceExt;
use briny::prelude::*;
use crate::ops::dispatch::{FnF64Ten64, FnTen64To, FnToDoubleTen64};

const MATMUL: &str = include_str!("shaders/matmul.wgsl");
const MSE_LOSS: &str = include_str!("shaders/mse_loss.wgsl");
const RELU: &str = include_str!("shaders/relu.wgsl");
const SGD: &str = include_str!("shaders/sgd.wgsl");

/// Basic wrapper for common GPU errors.
#[derive(Debug)]
pub enum GpuError {
    /// An error in requesting the addapter.
    Adapter(wgpu::RequestAdapterError),
    /// An error in requesting the GPU (device).
    Device(wgpu::RequestDeviceError),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::Adapter(e) => write!(f, "Adapter error: {e}"),
            GpuError::Device(e) => write!(f, "Device error: {e}")
        }
    }
}

/// Wrapper for a `GpuError` or `ValidationError` depending on how it fails.
#[derive(Debug)]
pub enum GpuFailureKind {
    /// An error resulting from the GPU.
    Gpu(GpuError),
    /// An error resulting from validating data.
    Validation(ValidationError),
}

impl std::fmt::Display for GpuFailureKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuFailureKind::Gpu(err) => write!(f, "GPU error: {err}"),
            GpuFailureKind::Validation(err) => write!(f, "Validation error: {err}"),
        }
    }
}

/// A type of error closely related to the GPU.
#[derive(Debug)]
pub struct GpuFailure {
    /// The optional type of failure that occured.
    pub kind: Option<GpuFailureKind>,
    /// The optional message explaining the failure.
    pub message: Option<String>,
}

impl From<GpuError> for GpuFailure {
    fn from(kind: GpuError) -> Self {
        Self { kind: Some(GpuFailureKind::Gpu(kind)), message: None }
    }
}

impl From<ValidationError> for GpuFailure {
    fn from(kind: ValidationError) -> Self {
        Self { kind: Some(GpuFailureKind::Validation(kind)), message: None }
    }
}

impl From<&str> for GpuFailure {
    fn from(msg: &str) -> Self {
        Self { kind: None, message: Some(msg.to_string()) }
    }
}

impl From<String> for GpuFailure {
    fn from(msg: String) -> Self {
        Self { kind: None, message: Some(msg) }
    }
}

impl std::fmt::Display for GpuFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if let Some(kind) = &self.kind {
            write!(f, "GPU failure: {kind}")
        } else if let Some(msg) = &self.message {
            write!(f, "GPU failure: {msg}")
        } else {
            write!(f, "Unknown GPU failure")
        }
    }
}

impl std::error::Error for GpuFailure {}

/// Holds the WGPU device and queue used for executing compute pipelines.
/// 
/// Initialized once globally and reused for all operations via `lazy_static`.
/// Provides the base hardware abstraction for launching compute shaders.
pub struct GpuContext {
    /// The actual GPU device.
    pub device: wgpu::Device,
    /// A queue for information related to the device.
    pub queue: wgpu::Queue,
}

impl GpuContext {
    /// Initializes a new GPU context, selecting the default adapter and creating a device + queue.
    ///
    /// This function sets up the GPU backend used for all compute operations.
    /// It wraps WGPU’s initialization logic and is called once via `lazy_static`.
    ///
    /// # Returns
    /// - `Ok(GpuContext)` if the GPU is successfully initialized
    /// - `Err(GpuError)` if adapter or device acquisition fails
    ///
    /// # Internals
    /// - Uses `pollster::block_on` to synchronously wait for async WGPU calls
    /// - Selects the default adapter with default options (typically the most performant)
    /// - Enables default limits and features for broad compatibility
    ///
    /// # Panics
    /// Only panics if called via `lazy_static!` and the initialization fails
    ///
    /// # Example
    /// ```rust
    /// use briny_ai::ops::wgpu::GpuContext;
    /// 
    /// let ctx = GpuContext::new()?;
    /// println!("Device: {:?}", ctx.device.limits());
    /// ```
    pub fn new() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::default();
        // Use block_on to await the adapter
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .map_err(GpuError::Adapter)?; // GpuError::Adapter wraps the RequestAdapterError
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::default(),
        }))
        .map_err(GpuError::Device)?; // wraps RequestDeviceError

        Ok(Self { device, queue })
    }
}

/// Secure wrapper for WGSL source code extracted from files.
pub struct WgslSource<'a>(pub &'a str);

impl<'a> Validate for WgslSource<'a> {
    fn validate(&self) -> Result<(), ValidationError> {
        let src = self.0;

        // Basic sanity checks
        if src.len() > 65536 {
            return Err(ValidationError);
        }

        if !src.contains("fn main") {
            return Err(ValidationError);
        }

        if src.contains("import") || src.contains("#include") {
            return Err(ValidationError); // Disallow source inclusion
        }

        // Disallow forbidden patterns
        let forbidden = ["asm", "unsafe", "ptr", "std::"];
        if forbidden.iter().any(|bad| src.contains(bad)) {
            return Err(ValidationError);
        }

        Ok(())
    }
}

/// Opens a WGSL shader and returns the validated, labeled contents.
/// 
/// # In Detail
/// - Opens a WGSL shader, contains it in a secure wrapper, ensures safety and validates it.
/// - Once validated, the shader is labeled and assigned to a device, unwrapped, and returned.
pub fn load_shader(
    device: &wgpu::Device,
    label: &str,
    source: &str,
) -> Result<wgpu::ShaderModule, GpuFailure> {
    WgslSource(source).validate()?; // briny-based check

    Ok(device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    }))
}

lazy_static::lazy_static! {
    static ref GPU_CONTEXT: GpuContext = GpuContext::new().expect("Failed to initialize GPU context");
    static ref MATMUL_SHADER: wgpu::ShaderModule = load_shader(
        &GPU_CONTEXT.device,
        "matmul",
        MATMUL
    ).expect("MATMUL shader failed validation or compilation");
    static ref MATMUL_BIND_GROUP_LAYOUT: wgpu::BindGroupLayout = {
        GPU_CONTEXT.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("matmul_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    };
    static ref MATMUL_PIPELINE_LAYOUT: wgpu::PipelineLayout = {
        GPU_CONTEXT.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("matmul_pipeline_layout"),
            bind_group_layouts: &[&*MATMUL_BIND_GROUP_LAYOUT],
            push_constant_ranges: &[],
        })
    };
    static ref MATMUL_PIPELINE: wgpu::ComputePipeline = {
        GPU_CONTEXT.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matmul_pipeline"),
            layout: Some(&*MATMUL_PIPELINE_LAYOUT),
            module: &MATMUL_SHADER,
            entry_point: Some("main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        })
    };
    static ref MSE_LOSS_SHADER: wgpu::ShaderModule = {
        GPU_CONTEXT.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mse_loss"),
            source: wgpu::ShaderSource::Wgsl(MSE_LOSS.into()),
        })
    };
    static ref MSE_LOSS_BIND_GROUP_LAYOUT: wgpu::BindGroupLayout = {
        GPU_CONTEXT.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mse_loss_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    };
    static ref MSE_LOSS_PIPELINE_LAYOUT: wgpu::PipelineLayout = {
        GPU_CONTEXT.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mse_loss_pipeline_layout"),
            bind_group_layouts: &[&*MSE_LOSS_BIND_GROUP_LAYOUT],
            push_constant_ranges: &[],
        })
    };
    static ref MSE_LOSS_PIPELINE: wgpu::ComputePipeline = {
        GPU_CONTEXT.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mse_loss_pipeline"),
            layout: Some(&*MSE_LOSS_PIPELINE_LAYOUT),
            module: &MSE_LOSS_SHADER,
            entry_point: Some("main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        })
    };
    static ref RELU_SHADER: wgpu::ShaderModule = {
        GPU_CONTEXT.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("relu"),
            source: wgpu::ShaderSource::Wgsl(RELU.into()),
        })
    };
    static ref RELU_BIND_GROUP_LAYOUT: wgpu::BindGroupLayout = {
        GPU_CONTEXT.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("relu_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    };
    static ref RELU_PIPELINE_LAYOUT: wgpu::PipelineLayout = {
        GPU_CONTEXT.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("relu_pipeline_layout"),
            bind_group_layouts: &[&*RELU_BIND_GROUP_LAYOUT],
            push_constant_ranges: &[],
        })
    };
    static ref RELU_PIPELINE: wgpu::ComputePipeline = {
        GPU_CONTEXT.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("relu_pipeline"),
            layout: Some(&*RELU_PIPELINE_LAYOUT),
            module: &RELU_SHADER,
            entry_point: Some("main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        })
    };
    static ref SGD_SHADER: wgpu::ShaderModule = {
        GPU_CONTEXT.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sgd"),
            source: wgpu::ShaderSource::Wgsl(SGD.into()),
        })
    };
    static ref SGD_BIND_GROUP_LAYOUT: wgpu::BindGroupLayout = {
        GPU_CONTEXT.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sgd_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    };
    static ref SGD_PIPELINE_LAYOUT: wgpu::PipelineLayout = {
        GPU_CONTEXT.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sgd_pipeline_layout"),
            bind_group_layouts: &[&*SGD_BIND_GROUP_LAYOUT],
            push_constant_ranges: &[],
        })
    };
    static ref SGD_PIPELINE: wgpu::ComputePipeline = {
        GPU_CONTEXT.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sgd_pipeline"),
            layout: Some(&*SGD_PIPELINE_LAYOUT),
            module: &SGD_SHADER,
            entry_point: Some("main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        })
    };
}

fn as_bytes<T: Copy>(data: &[T]) -> &[u8] {
    let len = std::mem::size_of_val(data);
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, len) }
}

fn bytes_to_f32_slice(data: &[u8]) -> Result<&[f32], &'static str> {
    use std::mem::{align_of, size_of};

    if data.as_ptr() as usize % align_of::<f32>() != 0 {
        return Err("unaligned buffer");
    }

    if data.len() % size_of::<f32>() != 0 {
        return Err("buffer length is not a multiple of f32");
    }

    let len = data.len() / size_of::<f32>();
    let ptr = data.as_ptr() as *const f32;
    unsafe {
        Ok(std::slice::from_raw_parts(ptr, len))
    }
}

/// Performs matrix multiplication on the GPU using a precompiled WGSL shader.
///
/// Accepts two input tensors `a` (shape `[m, k]`) and `b` (shape `[k, n]`)
/// and computes the output `c = a @ b` on the GPU, returning a Tensor of shape `[m, n]`.
///
/// Returns a closure for computing the backward gradient (on the CPU).
///
/// # Returns
/// - `Some((Tensor, backward_fn))` on success
/// - `None` if shape mismatch or GPU failure occurs
///
/// # Notes
/// - Input data is cast from f64 → f32 for GPU
/// - Output is cast back from f32 → f64
pub fn wgpu_matmul(
    a: &WithGrad<Ten64>,
    b: &WithGrad<Ten64>,
) -> Option<(Ten64, Box<FnToDoubleTen64>)> {
    let (m, k) = (a.value.shape[0], a.value.shape[1]);
    let (k2, n) = (b.value.shape[0], b.value.shape[1]);
    if k != k2 {
        return None;
    }

    let a_data: Vec<f32> = a.value.data.iter().map(|&v| v as f32).collect();
    let b_data: Vec<f32> = b.value.data.iter().map(|&v| v as f32).collect();

    let output_size = m * n;
    let mut output_data = vec![0.0f32; output_size];

    let result = pollster::block_on(run_matmul_shader(
        &a_data,
        &b_data,
        &mut output_data,
        m,
        k,
        n,
    ));

    if result.is_err() {
        return None;
    }

    let output_data_f64: Vec<f64> = output_data.into_iter().map(|v| v as f64).collect();
    let out_tensor = Tensor::new(vec![m, n], output_data_f64);

    let a_val = a.value.clone();
    let b_val = b.value.clone();

    let back = Box::new(move |grad: &Ten64| {
        let grad_data: Vec<f32> = grad.data.iter().map(|&v| v as f32).collect();
        let a_data: Vec<f32> = a_val.data.iter().map(|&v| v as f32).collect();
        let b_data: Vec<f32> = b_val.data.iter().map(|&v| v as f32).collect();

        // transpose B → Bᵀ for dA = grad @ Bᵀ
        let mut b_t_data = vec![0.0f32; b_val.data.len()];
        for i in 0..n {
            for j in 0..k {
                b_t_data[i * k + j] = b_data[j * n + i];
            }
        }

        // transpose A → Aᵀ for dB = Aᵀ @ grad
        let mut a_t_data = vec![0.0f32; a_val.data.len()];
        for i in 0..k {
            for j in 0..m {
                a_t_data[i * m + j] = a_data[j * k + i];
            }
        }

        let mut da_f32 = vec![0.0f32; m * k];
        let mut db_f32 = vec![0.0f32; k * n];

        // ∂L/∂A = grad @ Bᵀ
        pollster::block_on(run_matmul_shader(
            &grad_data,
            &b_t_data,
            &mut da_f32,
            m, n, k, // note: Bᵀ shape = [n x k] → input is [m x n] × [n x k]
        )).unwrap();

        // ∂L/∂B = Aᵀ @ grad
        pollster::block_on(run_matmul_shader(
            &a_t_data,
            &grad_data,
            &mut db_f32,
            k, m, n, // note: Aᵀ shape = [k x m] → input is [k x m] × [m x n]
        )).unwrap();

        let da = Tensor::new(vec![m, k], da_f32.into_iter().map(|v| v as f64).collect());
        let db = Tensor::new(vec![k, n], db_f32.into_iter().map(|v| v as f64).collect());

        (da, db)
    });

    Some((out_tensor, back))
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

    let dims = [m as u32, k as u32, n as u32, 0u32];
    let dims_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dims"),
        contents: as_bytes(&dims),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let a_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("A"),
        contents: as_bytes(a),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let b_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("B"),
        contents: as_bytes(b),
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
        compute_pass.dispatch_workgroups((n as u32).div_ceil(16), (m as u32).div_ceil(16), 1);
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
    buffer_slice.map_async(wgpu::MapMode::Read, |result| {
        assert!(result.is_ok());
    });
    device.poll(wgpu::PollType::Wait).unwrap();

    let data = buffer_slice.get_mapped_range();
    out.copy_from_slice(bytes_to_f32_slice(&data)?);
    drop(data);
    staging_buffer.unmap();

    Ok(())
}

/// Computes Mean Squared Error (MSE) loss between prediction and target on the GPU.
///
/// Uses a compute shader to calculate `(prediction[i] - target[i])^2`
/// and returns the average loss as an f64 scalar.
///
/// # Returns
/// - `Some((loss, backward_fn))` on success
/// - `None` if GPU execution fails
///
/// # Backward Function
/// The returned closure takes a scalar gradient (usually 1.0 from upstream)
/// and computes elementwise gradients: `2 * (pred - target) / N`.
///
/// # Notes
/// - Loss is averaged over all elements
/// - Uses f32 internally and casts back to f64
pub fn wgpu_mse_loss<'a>(
    pred: &'a WithGrad<Ten64>,
    target: &'a Ten64,
) -> Option<(f64, Box<FnF64Ten64<'a>>)> {
    let p: Vec<f32> = pred.value.data.iter().map(|&x| x as f32).collect();
    let t: Vec<f32> = target.data.iter().map(|&x| x as f32).collect();

    let result = pollster::block_on(run_mse_loss_shader(&p, &t)).ok()?;

    let back = Box::new(move |grad: f64| {
        let grad_data: Vec<f64> = p
            .iter()
            .zip(t.iter())
            .map(|(&x, &y)| 2.0 * grad * (x - y) as f64 / p.len() as f64)
            .collect();
        Tensor::new(pred.value.shape.clone(), grad_data)
    });

    Some((result as f64, back))
}

async fn run_mse_loss_shader(prediction: &[f32], target: &[f32]) -> Result<f32, GpuFailure> {
    let device = &GPU_CONTEXT.device;
    let queue = &GPU_CONTEXT.queue;

    let len = prediction.len();
    assert_eq!(len, target.len());
    let buffer_size = (std::mem::size_of_val(prediction)) as u64;

    // === Buffers ===
    let pred_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("prediction"),
        contents: as_bytes(prediction),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let target_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("target"),
        contents: as_bytes(target),
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
        cpass.dispatch_workgroups((len as u32).div_ceil(64), 1, 1);
    }

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mse_staging"),
        size: buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&loss_buf, 0, &staging, 0, buffer_size);
    queue.submit(Some(encoder.finish()));

    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::Wait);

    let view = staging.slice(..).get_mapped_range();
    let loss_terms: &[f32] = bytes_to_f32_slice(&view)?;
    let total_loss = loss_terms.iter().sum::<f32>() / len as f32;
    drop(view);
    staging.unmap();

    Ok(total_loss)
}

/// Performs elementwise ReLU (`max(x, 0)`) on the GPU.
///
/// Accepts an input tensor and applies ReLU in-place using a compute shader.
/// Returns a new Tensor and a closure for backpropagation that masks negative inputs.
///
/// # Returns
/// - `Some((Tensor, backward_fn))` on success
/// - `None` on GPU execution failure
///
/// # Notes
/// - Uses f32 precision on GPU
/// - Output and backward gradient are returned in f64 for integration
pub fn wgpu_relu(
    input: &WithGrad<Ten64>,
) -> Option<(Ten64, Box<FnTen64To>)> {
    let data: Vec<f32> = input.value.data.iter().map(|&x| x as f32).collect();
    let mut output = vec![0.0f32; data.len()];

    let result = pollster::block_on(run_relu_shader(&data, &mut output));
    if result.is_err() {
        return None;
    }

    let output_tensor = Tensor::new(
        input.value.shape.clone(),
        output.into_iter().map(|x| x as f64).collect(),
    );
    let back = move |grad: &Ten64| {
        let grad_data = grad
            .data
            .iter()
            .zip(input.value.data.iter())
            .map(|(&g, &x)| if x > 0.0 { g } else { 0.0 })
            .collect();
        Tensor::new(input.value.shape.clone(), grad_data)
    };

    Some((output_tensor, Box::new(back)))
}

async fn run_relu_shader(input: &[f32], output: &mut [f32]) -> Result<(), GpuFailure> {
    assert_eq!(output.len() % 4, 0);

    let device = &GPU_CONTEXT.device;
    let queue = &GPU_CONTEXT.queue;

    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input"),
        contents: as_bytes(input),
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
        pass.dispatch_workgroups((input.len() as u32).div_ceil(64), 1, 1);
    }

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("relu_staging"),
        size: (output.len() * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, (output.len() * 4) as u64);
    queue.submit(Some(encoder.finish()));

    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::Wait);

    let data = staging.slice(..).get_mapped_range();
    output.copy_from_slice(bytes_to_f32_slice(&data)?);
    drop(data);
    staging.unmap();

    Ok(())
}

/// Performs an in-place Stochastic Gradient Descent (SGD) update on the GPU.
///
/// For each weight `w[i]`, performs: `w[i] -= lr * grad[i]`.
/// Updates the tensor in-place and returns success/failure.
///
/// # Parameters
/// - `w`: The weight tensor (with associated gradient)
/// - `lr`: Learning rate (f64, cast to f32 for GPU)
///
/// # Returns
/// - `true` if update succeeded
/// - `false` on GPU execution failure
pub fn wgpu_sgd(w: &mut WithGrad<Ten64>, lr: f64) -> bool {
    let mut weights: Vec<f32> = w.value.data.iter().map(|&x| x as f32).collect();
    let grads: Vec<f32> = w.grad.data.iter().map(|&x| x as f32).collect();

    let result = pollster::block_on(run_sgd_shader(&mut weights, &grads, lr as f32));
    if result.is_err() {
        return false;
    }

    w.value.data = weights.into_iter().map(|x| x as f64).collect();
    true
}

async fn run_sgd_shader(weights: &mut [f32], grad: &[f32], lr: f32) -> Result<(), GpuFailure> {
    assert_eq!(weights.len(), grad.len());
    assert_eq!(weights.len() % 4, 0);
    let device = &GPU_CONTEXT.device;
    let queue = &GPU_CONTEXT.queue;

    let weights_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("weights"),
        contents: as_bytes(weights),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let grad_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("grad"),
        contents: as_bytes(grad),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let lr_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("lr"),
        contents: as_bytes(&[lr]),
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
        cpass.dispatch_workgroups((weights.len() as u32).div_ceil(64), 1, 1);
    }

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_weights"),
        size: (weights.len() * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&weights_buf, 0, &staging, 0, (weights.len() * 4) as u64);

    queue.submit(Some(encoder.finish()));
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::Wait);

    let view = staging.slice(..).get_mapped_range();
    let updated_weights: &[f32] = bytes_to_f32_slice(&view)?;
    weights.copy_from_slice(updated_weights);
    drop(view);
    staging.unmap();

    Ok(())
}
