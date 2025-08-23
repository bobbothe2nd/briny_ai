use crate::abstracted::internal::TensorLike;
use crate::manual::TensorFloat;
use crate::manual::tensors::WithGrad;

#[cfg(feature = "dyntensor")]
use crate::manual::tensors::{Tensor, TensorOps};

mod dense;
pub use dense::*;

#[cfg(feature = "dyntensor")]
fn reduce_rows_sum_to_bias(dz: &Tensor<TensorFloat>, out_len: usize) -> Tensor<TensorFloat> {
    let mut acc = alloc::vec![0.0; out_len]; // or a fixed array if OUT is const
    let dz_data = dz.data();
    let batch = dz_data.len() / out_len;

    for i in 0..batch {
        for j in 0..out_len {
            acc[j] += dz_data[i * out_len + j];
        }
    }

    Tensor::new(&[out_len], &acc) // however you normally construct 1D tensors
}

#[cfg(feature = "alloc")]
use alloc::boxed::Box;
#[cfg(not(feature = "alloc"))]
use box_closure::OpaqueFnOnce;

#[cfg(feature = "alloc")]
pub type BackFn<'a, In, Out> = Box<dyn FnOnce(Out) -> In + 'a>;

#[cfg(not(feature = "alloc"))]
pub type BackFn<'a, In, Out> = OpaqueFnOnce<'a, Out, In, box_closure::Align32<1024>>;

/// Internal utilities to operate on layers.
///
/// Obtained from `__LayoutMarker` at compile time when
/// using the `#[model]` proc-macro.
pub trait Layer<I, O> {
    type ParamA: TensorLike<TensorFloat>;
    type ParamB: TensorLike<TensorFloat>;

    fn run<'a>(&'a mut self, input: &'a WithGrad<I>) -> (O, BackFn<'a, I, O>);

    fn params(&mut self) -> (&mut WithGrad<Self::ParamA>, &mut WithGrad<Self::ParamB>);
}

/// Marker trait for layout dummies.
///
/// Converted to [`Layer<I, O>`] at compile time when
/// using the `#[model]` proc-macro.
/// The internal [`Layer`] trait
pub trait __LayoutMarker {}
