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

#![allow(clippy::type_complexity)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::unused_async)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::manual_slice_size_calculation)]
#![allow(clippy::struct_field_names)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::approx_constant)]
#![cfg_attr(feature = "dyntensor", allow(clippy::useless_conversion))]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![deny(clippy::nursery)]
#![deny(unused_must_use)]
#![forbid(unsafe_op_in_unsafe_fn)]
#![forbid(missing_docs)]
#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

// not meant to be acccessed outside the macro
#[cfg(feature = "alloc")]
pub mod macros;

pub mod approx;
pub mod backend;
pub mod nn;

pub mod prelude {
    //! Common re-exports at a central location.

    pub use crate::{
        backend::{get_backend, set_backend, Backend},
        nn::io::BpatHeader,
        nn::tensors::{Flatten, TensorOps},
    };

    #[cfg(feature = "alloc")]
    pub use crate::{
        macros::{adapt_lr, decay_lr, test::TestEval, Context, Dataset},
        static_model,
    };
}
