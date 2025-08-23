use crate::abstracted::internal::Closure;
use crate::abstracted::internal::TensorLike;
use crate::manual::TensorFloat;
use crate::manual::backprop::relu;
use crate::manual::tensors::WithGrad;

#[cfg(feature = "alloc")]
use alloc::boxed::Box;
#[cfg(not(feature = "alloc"))]
use box_closure::{Align32, OpaqueFn};

pub trait Activation {
    type Tensor: TensorLike<TensorFloat>;
    type BackFn<'a, T: 'a, U: 'a>: Closure<T, U>
    where
        Self: 'a;

    fn apply<'a>(
        &'a self,
        x: &'a WithGrad<Self::Tensor>,
    ) -> (Self::Tensor, Self::BackFn<'a, Self::Tensor, Self::Tensor>);
}

pub struct Relu<const N: usize, const D: usize>;

impl<const N: usize, const D: usize> Activation for Relu<N, D> {
    #[cfg(feature = "dyntensor")]
    type Tensor = crate::manual::tensors::Tensor<TensorFloat>;
    #[cfg(not(feature = "dyntensor"))]
    type Tensor = crate::manual::tensors::Tensor<TensorFloat, N, D>;
    #[cfg(feature = "alloc")]
    type BackFn<'a, T: 'a, U: 'a> = Box<dyn Fn(T) -> U + 'a>;
    #[cfg(not(feature = "alloc"))]
    type BackFn<'a, T: 'a, U: 'a> = OpaqueFn<'a, T, U, Align32<256>>;

    fn apply<'a>(
        &'a self,
        x: &'a WithGrad<Self::Tensor>,
    ) -> (Self::Tensor, Self::BackFn<'a, Self::Tensor, Self::Tensor>) {
        relu(x)
    }
}
