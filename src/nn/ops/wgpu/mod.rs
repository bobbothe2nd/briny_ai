//! GPU-accelerated tensor operations using WGPU.
//!
//! This module implements high-performance compute kernels on the GPU using WGPU + WGSL.
//! It handles GPU context initialization, shader precompilation (via `lazy_static`),
//! and compute dispatch for key neural network operations:
//!
//! - `matmul` — matrix multiplication
//! - `relu` — `ReLU` activation
//! - `mse_loss` — mean squared error loss
//! - `sgd` — stochastic gradient descent update
//!
//! All shaders are compiled and cached once at runtime. Tensor data is copied
//! to the GPU for compute and returned as f64 to integrate with the rest of the framework.
//!
//! Most functions return both forward results and backward closures for autograd.

#[allow(dead_code)]
fn array_from_slice<T: Copy, const N: usize>(v: &[T]) -> [T; N] {
    v.try_into().unwrap_or_else(|_| {
        panic!(
            "expected {} elements, got {}; could not convert slice to array",
            N,
            v.len()
        )
    })
}

/// Single-threaded executor tuned for GPU futures.
fn block_on_gpu<F: core::future::Future>(f: F) -> F::Output {
    use core::{
        sync::atomic::{AtomicBool, Ordering},
        task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
    };
    use std::thread;

    #[inline]
    unsafe fn clone_raw(ptr: *const ()) -> RawWaker {
        RawWaker::new(ptr, &VTABLE)
    }

    #[inline]
    unsafe fn wake(ptr: *const ()) {
        let flag = unsafe { &*(ptr.cast::<AtomicBool>()) };
        flag.store(true, Ordering::Release);
        thread::current().unpark();
    }

    #[inline]
    const unsafe fn drop_raw(_: *const ()) {}

    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone_raw, wake, wake, drop_raw);

    let mut fut = Box::pin(f);
    let ready = AtomicBool::new(false);

    let raw = RawWaker::new((&raw const ready).cast::<()>(), &VTABLE);
    let waker = unsafe { Waker::from_raw(raw) };
    let mut cx = Context::from_waker(&waker);

    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(val) => return val,
            Poll::Pending => {
                ready.store(false, Ordering::Release);
                let _ = GPU_CONTEXT.device.poll(wgpu::PollType::Wait);
                let _ = ready.swap(false, Ordering::Acquire);
            }
        }
    }
}

use briny::BrinyError;
use wgpu::PollError;

#[cfg(feature = "alloc")]
use alloc::{boxed::Box, vec};

const MATMUL: &str = include_str!("shaders/matmul.wgsl");
const MATMUL_2D: &str = include_str!("shaders/matmul2.wgsl");
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

impl core::fmt::Display for GpuError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Adapter(e) => write!(f, "Adapter error: {e}"),
            Self::Device(e) => write!(f, "Device error: {e}"),
        }
    }
}

/// A type of error closely related to the GPU.
#[derive(Debug)]
pub struct GpuFailure {
    /// The optional type of failure that occured.
    pub kind: Option<GpuError>,
}

impl From<GpuError> for GpuFailure {
    fn from(kind: GpuError) -> Self {
        Self { kind: Some(kind) }
    }
}

impl From<BrinyError> for GpuFailure {
    fn from(_: BrinyError) -> Self {
        Self { kind: None }
    }
}

impl From<PollError> for GpuFailure {
    fn from(_: PollError) -> Self {
        Self { kind: None }
    }
}

impl core::fmt::Display for GpuFailure {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        if let Some(kind) = &self.kind {
            write!(f, "GPU failure: {kind}")
        } else {
            write!(f, "Unknown GPU failure")
        }
    }
}

impl core::error::Error for GpuFailure {}

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
    /// # Errors
    ///
    /// This method can sometimes return an error:
    ///
    /// - `Ok(GpuContext)` if the GPU is successfully initialized
    /// - `Err(GpuError)` if adapter or device acquisition fails
    ///
    /// # Internals
    ///
    /// - Uses `pollster::block_on` to synchronously wait for async WGPU calls
    /// - Selects the default adapter with default options (typically the most performant)
    /// - Enables default limits and features for broad compatibility
    ///
    /// # Panics
    ///
    /// Only panics if called via `lazy_static!` and the initialization fails
    ///
    /// # Example
    ///
    /// ```rust
    /// use briny_ai::nn::ops::wgpu::GpuContext;
    ///
    /// let ctx = GpuContext::new().unwrap();
    /// println!("Device: {:?}", ctx.device.limits());
    /// ```
    pub fn new() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::default();
        // Use block_on to await the adapter
        let adapter =
            block_on_gpu(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .map_err(GpuError::Adapter)?; // GpuError::Adapter wraps the RequestAdapterError
        let (device, queue) = block_on_gpu(adapter.request_device(&wgpu::DeviceDescriptor {
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

fn load_shader(device: &wgpu::Device, label: &str, source: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}

lazy_static::lazy_static! {
    static ref GPU_CONTEXT: GpuContext = #[allow(clippy::expect_used)] GpuContext::new().expect("failed to initialize `GPU_CONTEXT`");

    static ref MATMUL_SHADER: wgpu::ShaderModule = load_shader(
        &GPU_CONTEXT.device,
        "matmul",
        MATMUL
    );
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
                        ty: wgpu::BufferBindingType::Uniform,
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
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

    static ref MATMUL_2D_SHADER: wgpu::ShaderModule = load_shader(
        &GPU_CONTEXT.device,
        "matmul2",
        MATMUL_2D
    );
    static ref MATMUL_2D_BIND_GROUP_LAYOUT: wgpu::BindGroupLayout = {
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
    static ref MATMUL_2D_PIPELINE_LAYOUT: wgpu::PipelineLayout = {
        GPU_CONTEXT.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("matmul2_pipeline_layout"),
            bind_group_layouts: &[&*MATMUL_2D_BIND_GROUP_LAYOUT],
            push_constant_ranges: &[],
        })
    };
    static ref MATMUL_2D_PIPELINE: wgpu::ComputePipeline = {
        GPU_CONTEXT.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matmul2_pipeline"),
            layout: Some(&*MATMUL_2D_PIPELINE_LAYOUT),
            module: &MATMUL_2D_SHADER,
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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

mod matmul;
pub use self::matmul::wgpu_matmul;

mod mse;
pub use self::mse::wgpu_mse_loss;

mod relu;
pub use self::relu::wgpu_relu;

mod sgd;
pub use self::sgd::wgpu_sgd;
