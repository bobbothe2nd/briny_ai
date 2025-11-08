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
//! - GPU acceleration is only used when the feature flag is enabled
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
//! - `wgpu` — Enables `wgpu` (WebGPU) backend
//! - `cuda` — Enables placeholder CUDA module (dispatches to WGPU)

// dispatch layer...
pub mod dispatch;

// ... across these backends:
pub mod cpu;

#[cfg(feature = "wgpu")]
pub mod wgpu;

#[cfg(feature = "cuda")]
pub mod cuda;
