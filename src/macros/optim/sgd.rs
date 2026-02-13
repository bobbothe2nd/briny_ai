use crate::{
    macros::{optim::Optim, Tensor},
    nn::{tensors::WithGrad, TensorFloat},
};

/// Performs SGD function.
pub struct Sgd<const RANK: usize, const SIZE: usize> {}

impl<const RANK: usize, const W_SIZE: usize, const IN_SIZE: usize> Optim<RANK, IN_SIZE>
    for Sgd<RANK, W_SIZE>
{
    fn new(_: &[usize; RANK]) -> Self {
        Self {}
    }

    fn step(&mut self, w: &mut WithGrad<Tensor<RANK, IN_SIZE>>, lr: TensorFloat) {
        crate::nn::ops::dispatch::sgd(w, lr);
    }
}
