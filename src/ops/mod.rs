//! # Operation Dispatch Layer
//!
//! This module defines and dispatches tensor operations across different compute backends,
//! including CPU, GPU (WGPU), and optionally CUDA.
//!
//! ## Submodules
//!
//! - [`cpu`] — Multi-threaded + SIMD CPU operations (default fallback backend)
//! - [`wgpu`] *(opt-in)* — GPU compute shader pipelines using `wgpu`
//! - [`cuda`] *(planned)* — CUDA GPU backend for NVIDIA (not yet supported)
//! - [`dispatch`] — Dynamic backend switching and unified operation interfaces
//!
//! ## Backend Selection
//!
//! All core operations are designed to be backend-agnostic from the user perspective.
//! Dispatching logic is handled internally based on compile-time features or runtime flags.
//!
//! Example:
//! ```rust
//! use briny_ai::backend::Backend;
//! use briny_ai::backprop::matmul;
//! use briny_ai::tensors::WithGrad;
//! use briny_ai::tensor;
//! 
//! let a = WithGrad::new(tensor!([[5.0, 4.0], [3.0, 2.0]]));
//! let b = WithGrad::new(tensor!([[1.0, 2.0], [3.0, 4.0]]));
//! let backend = Backend::default(); // defaults to CPU
//! let result = matmul(&a, &b); // runs on CPU
//! ```
//!
//! ## Extending the Backend
//!
//! To add a new operation:
//!
//! 1. Implement it in one or more backends (e.g. `cpu::my_op`, `wgpu::my_op`)
//! 2. Add it to the `dispatch` module for unified access
//! 3. Add shape/consistency checks in a backend-agnostic location
//!
//! ## Notes
//!
//! - SIMD and GPU acceleration are only used when their feature flags are enabled
//! - CUDA support is not implemented yet; the module dispatches to WebGPU
//! - Operations must return both forward values and backward closures
//!
//! ## Goals
//!
//! - Keep backends cleanly separated
//! - Zero-cost dispatch when statically chosen
//! - Maximal performance with minimal code duplication
//!
//! ## Feature Flags
//!
//! - `simd` — Enables AVX2-accelerated CPU paths
//! - `wgpu` — Enables `wgpu` (WebGPU) backend
//! - `cuda` — Enables placeholder CUDA module (dispatches to WGPU)

pub mod dispatch;
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(any(feature = "wgpu", feature = "cuda"))]
pub mod wgpu;