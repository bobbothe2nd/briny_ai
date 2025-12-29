//! # `briny_ai`
//!
//! This crate provides a fast, minimal, and modular deep learning backend
//! built in Rust. It features basic tensor operations, automatic differentiation,
//! and support for CPU acceleration via Lazy SIMD. GPU support via WGPU is optional
//! and designed for portability across Intel, AMD, and NVIDIA hardware.
//!
//! ## Features
//!
//! - **Tensors**: N-dimensional arrays with shape tracking and gradient support
//! - **Autograd**: Functional-style forward and backward passes
//! - **Operators**: Efficient implementations of matrix multiplication, `ReLU`,
//!   mean squared error, and stochastic gradient descent
//! - **GPU Acceleration**: Optional `wgpu`-powered compute shaders for matrix ops

#![warn(clippy::all, clippy::pedantic)]
#![allow(
    clippy::type_complexity,
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::unused_async,
    clippy::unnecessary_cast,
    clippy::too_many_lines,
    clippy::manual_slice_size_calculation,
    clippy::struct_field_names,
    clippy::cast_lossless,
    clippy::excessive_precision,
    clippy::approx_constant
)]
#![deny(clippy::nursery, unused_must_use)]
#![forbid(unsafe_op_in_unsafe_fn)]
#![forbid(missing_docs)]
#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

// not meant to be acccessed outside the macro
pub mod macros;

pub mod approx;
pub mod backend;
pub mod nn;

pub mod prelude {
    //! Common re-exports at a central location.

    pub use crate::{
        backend::{get_backend, set_backend, Backend},
        nn::tensors::{Flatten, TensorOps},
        macros::{Dataset, test::TestEval, Context, adapt_lr, decay_lr},
        static_model,
    };
}
