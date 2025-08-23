use crate::abstracted::internal::Closure;
use crate::{
    abstracted::internal::TensorLike,
    manual::{
        TensorFloat,
        tensors::{Tensor, WithGrad},
    },
};

pub trait LossEval {
    type Data;
    type Tensor: TensorLike<TensorFloat>;
    type BackFn<'a>: Closure<TensorFloat, Self::Tensor>;

    fn evaluate<'a>(
        &self,
        pred: &'a WithGrad<Self::Data>,
        actual: &'a Self::Data,
    ) -> (f64, Self::BackFn<'a>);
}

#[cfg(feature = "dyntensor")]
pub struct MseLoss;
#[cfg(not(feature = "dyntensor"))]
pub struct MseLoss<const N: usize, const D: usize>;

#[cfg(feature = "dyntensor")]
impl LossEval for MseLoss {
    type Data = Tensor<TensorFloat>;
    type Tensor = Tensor<TensorFloat>;

    type BackFn<'a> =
        alloc::boxed::Box<dyn Fn(crate::manual::TensorFloat) -> Tensor<TensorFloat> + 'a>;

    fn evaluate<'a>(
        &self,
        pred: &'a WithGrad<Self::Data>,
        actual: &'a Self::Data,
    ) -> (
        f64,
        alloc::boxed::Box<dyn Fn(TensorFloat) -> Tensor<TensorFloat> + 'a>,
    ) {
        let (loss, back) = crate::manual::backprop::mse_loss(pred, actual);
        (loss as f64, back)
    }
}
#[cfg(not(feature = "dyntensor"))]
impl<const N: usize, const D: usize> LossEval for MseLoss<N, D> {
    type Data = Tensor<TensorFloat, N, D>;
    type Tensor = Tensor<TensorFloat, N, D>;

    #[cfg(feature = "alloc")]
    type BackFn<'a> =
        alloc::boxed::Box<dyn Fn(crate::manual::TensorFloat) -> Tensor<TensorFloat, N, D> + 'a>;
    #[cfg(not(feature = "alloc"))]
    type BackFn<'a> = box_closure::OpaqueFn<
        'a,
        TensorFloat,
        Tensor<TensorFloat, N, D>,
        box_closure::Align32<256>,
    >;

    fn evaluate<'a>(
        &self,
        pred: &'a WithGrad<Self::Data>,
        actual: &'a Self::Data,
    ) -> (f64, Self::BackFn<'a>) {
        let (loss, back) = crate::manual::backprop::mse_loss(pred, actual);
        (loss as f64, back)
    }
}
