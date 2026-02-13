use crate::{
    macros::{optim::Optim, Tensor},
    nn::{ops::dispatch::adam, tensors::WithGrad, TensorFloat},
};

/// Performs Adam function.
pub struct Adam<const RANK: usize, const SIZE: usize> {
    m: Tensor<RANK, SIZE>,
    v: Tensor<RANK, SIZE>,
}

impl<const RANK: usize, const SIZE: usize> Optim<RANK, SIZE> for Adam<RANK, SIZE> {
    fn new(shape: &[usize; RANK]) -> Self {
        Self {
            m: Tensor::zeros(shape),
            v: Tensor::zeros(shape),
        }
    }

    fn step(&mut self, w: &mut WithGrad<Tensor<RANK, SIZE>>, lr: TensorFloat) {
        adam(w, &mut self.m, &mut self.v, 0.0, lr);
    }
}
