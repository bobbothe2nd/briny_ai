use crate::{
    abstracted::internal::TensorLike,
    manual::{
        TensorFloat,
        tensors::{Tensor, WithGrad},
    },
};

pub trait Optimizer {
    type GradientTensor: TensorLike<TensorFloat>;

    fn with_lr(lr: f64) -> Self;
    fn step(&mut self, params: &mut [WithGrad<Self::GradientTensor>]);
}

#[cfg(feature = "dyntensor")]
pub struct Sgd {
    lr: f64,
}
#[cfg(not(feature = "dyntensor"))]
pub struct Sgd<const N: usize, const D: usize> {
    lr: f64,
}

#[cfg(feature = "dyntensor")]
impl Optimizer for Sgd {
    type GradientTensor = Tensor<TensorFloat>;

    fn with_lr(lr: f64) -> Self {
        Self { lr }
    }

    fn step(&mut self, params: &mut [WithGrad<Self::GradientTensor>]) {
        for p in params {
            crate::manual::backprop::sgd(p, self.lr);
        }
    }
}
#[cfg(not(feature = "dyntensor"))]
impl<const N: usize, const D: usize> Optimizer for Sgd<N, D> {
    type GradientTensor = Tensor<TensorFloat, N, D>;

    fn with_lr(lr: f64) -> Self {
        Self { lr }
    }

    fn step(&mut self, params: &mut [WithGrad<Self::GradientTensor>]) {
        for p in params {
            crate::manual::backprop::sgd(p, self.lr);
        }
    }
}
