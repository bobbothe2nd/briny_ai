//! Optimizer functions for macro.

mod adam;
pub use adam::Adam;

mod sgd;
pub use sgd::Sgd;

use crate::{
    macros::Tensor,
    nn::{tensors::WithGrad, TensorFloat},
};

/// Optimizer trait abstraction for types.
pub trait Optim<const RANK: usize, const SIZE: usize> {
    /// Construct a new optimizer.
    fn new(shape: &[usize; RANK]) -> Self;

    /// Run the function on the given weights.
    fn step(&mut self, w: &mut WithGrad<Tensor<RANK, SIZE>>, lr: TensorFloat);
}
