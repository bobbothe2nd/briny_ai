//! # briny_ai
//!
//! This crate provides a fast, minimal, and modular deep learning backend
//! built in Rust. It features basic tensor operations, automatic differentiation,
//! and support for CPU acceleration via Rayon and AVX2 SIMD. GPU support via WGPU
//! is optional and designed for portability across Intel, AMD, and NVIDIA hardware.
//!
//! ## Features
//!
//! - **Tensors**: N-dimensional arrays with shape tracking and gradient support
//! - **Autograd**: Functional-style forward and backward passes
//! - **Operators**: Efficient implementations of matrix multiplication, ReLU,
//!   mean squared error, and stochastic gradient descent
//! - **Parallelism**: Leveraging `rayon` for CPU-side data parallelism
//! - **SIMD**: Optional AVX2 SIMD acceleration for key operators
//! - **GPU Acceleration**: Optional `wgpu`-powered compute shaders for matrix ops
//!
//! ## Features
//!
//! Enable with Cargo features:
//!
//! - `simd` — Enables AVX2 SIMD acceleration on supported x86_64 targets
//! - `wgpu` — Enables GPU compute shaders via WGPU
//!
//! ## Safety Notes
//!
//! - SIMD operations are gated behind feature flags and use `unsafe` internally.
//! - GPU support assumes buffers are correctly sized and aligned — validated at runtime.

#![forbid(missing_docs)]

pub mod prelude {
    //! Common re-exports for ease of use.
    
    pub use crate::tensors::{Tensor, WithGrad};
    pub use crate::backprop::*;
    pub use crate::modelio::{save_model, load_model};
}

pub mod tensors;
pub mod backprop;
pub mod modelio;
pub mod ops;
pub mod backend;
