//! Differentiable operations and autograd utilities.
//!
//! # Backpropagation and Optimization Primitives
//!
//! Provides core operations with built-in autograd support for training neural networks and other models.
//!
//! **Key Features:**
//!
//! - **Elementwise Activation (ReLU):** Zero-out negatives and propagate gradients accordingly.
//! - **Matrix Multiplication:** Naïve m×k · k×n implementation with gradient closures.
//! - **Loss Computation (MSE):** Mean Squared Error with gradient generator.
//! - **Optimizer (SGD):** In-place parameter update with gradient reset.
//!
//! ## Autograd Pattern
//!
//! Each operation follows a simple pattern:
//!
//! 1. **Inputs** are references to `WithGrad<Tensor<T>>` for tensor ops.
//! 2. **Forward Pass** computes an output `Tensor<T>`.
//! 3. **Backward Pass** returns a closure capturing minimal cloned data to compute gradients.
//! 4. **Gradient Application** uses these results to update `WithGrad` wrappers.

pub use super::ops::dispatch::*;
