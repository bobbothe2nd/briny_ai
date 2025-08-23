//! # `briny_ai`
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
    clippy::unnecessary_cast
)]
#![deny(unsafe_code)]
#![forbid(clippy::nursery, unused_must_use)]
#![forbid(missing_docs)]
#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

pub mod backend;
pub mod manual;

// eventually public
// once its stable
#[allow(dead_code, unused_imports)]
mod abstracted;
