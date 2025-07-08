//! briny_ai: A lightweight, efficient AI/ML framework in Rust.
//!
//! Designed for high-performance numerical computation, tensor manipulation,
//! and deep learning with a focus on minimal dependencies and maximal clarity.
//!
//! # Features
//!
//! - Multi-dimensional tensor management with gradient support.
//! - Core deep learning operations with manual backpropagation closures.
//! - Model serialization and deserialization with compression and optional encryption.
//!
//! # Goals
//!
//! - Enable easy experimentation with neural networks and ML algorithms in Rust.
//! - Prioritize correctness, explicitness, and extensibility over black-box abstraction.
//! - Provide a solid base for embedded and resource-constrained ML workloads.
//!
//! # Modules
//!
//! - [`tensors`] — Core tensor data structures and operations.
//! - [`backprop`] — Differentiable operations and autograd utilities.
//! - [`modelio`] — Robust saving/loading of model weights with integrity checks and optional security.
//!
//! # Future Directions
//!
//! - Extend autograd to support dynamic computation graphs and more complex architectures.
//! - Optimize tensor math with SIMD, BLAS, or GPU acceleration backends.
//! - Add higher-level model abstractions (layers, optimizers, datasets).
//!
//! # Example
//!
//! ```rust
//! use briny_ai::tensors::Tensor;
//! // Create a simple tensor and manipulate it...
//! ```
//!
pub mod tensors;
pub mod backprop;
pub mod modelio;
