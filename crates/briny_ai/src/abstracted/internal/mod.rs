mod actfn;
pub use actfn::*;

mod optim;
pub use optim::*;

mod loss;
pub use loss::*;

use crate::manual::tensors::{IntoWithGrad, Tensor, TensorGrad, TensorOps};
use box_closure::{
    Align1, Align2, Align4, Align8, Align16, Align32, OpaqueFn, OpaqueFnMut, OpaqueFnOnce,
};

pub trait Closure<Arg, Ret> {
    fn invoke(&self, args: Arg) -> Ret;
}

pub trait ClosureMut<Arg, Ret> {
    fn invoke(&mut self, args: Arg) -> Ret;
}

pub trait ClosureOnce<Arg, Ret> {
    fn invoke(self, args: Arg) -> Ret;
}

macro_rules! __generate_opaque_closure_impl {
    ($struct:ident, $buf:ident) => {
        impl<'a, A, R, const B: usize> Closure<A, R> for $struct<'a, A, R, $buf<B>> {
            fn invoke(&self, args: A) -> R {
                self.call(args)
            }
        }
    };
}

macro_rules! __generate_opaque_mut_impl {
    ($struct:ident, $buf:ident) => {
        impl<'a, A, R, const B: usize> ClosureMut<A, R> for $struct<'a, A, R, $buf<B>> {
            fn invoke(&mut self, args: A) -> R {
                self.call(args)
            }
        }
    };
}

macro_rules! __generate_opaque_once_impl {
    ($struct:ident, $buf:ident) => {
        impl<'a, A, R, const B: usize> ClosureOnce<A, R> for $struct<'a, A, R, $buf<B>> {
            fn invoke(self, args: A) -> R {
                self.call(args)
            }
        }
    };
}

__generate_opaque_closure_impl!(OpaqueFn, Align1);
__generate_opaque_closure_impl!(OpaqueFn, Align2);
__generate_opaque_closure_impl!(OpaqueFn, Align4);
__generate_opaque_closure_impl!(OpaqueFn, Align8);
__generate_opaque_closure_impl!(OpaqueFn, Align16);
__generate_opaque_closure_impl!(OpaqueFn, Align32);
__generate_opaque_mut_impl!(OpaqueFnMut, Align1);
__generate_opaque_mut_impl!(OpaqueFnMut, Align2);
__generate_opaque_mut_impl!(OpaqueFnMut, Align4);
__generate_opaque_mut_impl!(OpaqueFnMut, Align8);
__generate_opaque_mut_impl!(OpaqueFnMut, Align16);
__generate_opaque_mut_impl!(OpaqueFnMut, Align32);
__generate_opaque_once_impl!(OpaqueFnOnce, Align1);
__generate_opaque_once_impl!(OpaqueFnOnce, Align2);
__generate_opaque_once_impl!(OpaqueFnOnce, Align4);
__generate_opaque_once_impl!(OpaqueFnOnce, Align8);
__generate_opaque_once_impl!(OpaqueFnOnce, Align16);
__generate_opaque_once_impl!(OpaqueFnOnce, Align32);

impl<A, R, F: ?Sized + Fn(A) -> R> Closure<A, R> for F {
    fn invoke(&self, args: A) -> R {
        self(args)
    }
}

impl<A, R, F: ?Sized + FnMut(A) -> R> ClosureMut<A, R> for F {
    fn invoke(&mut self, args: A) -> R {
        self(args)
    }
}

impl<A, R, F: FnOnce(A) -> R> ClosureOnce<A, R> for F {
    fn invoke(self, args: A) -> R {
        self(args)
    }
}

pub trait TensorLike<T>: TensorGrad + TensorOps<T> + IntoWithGrad {
    fn __new_with_data(data: &[T], shape: &[usize]) -> Self;
}
#[cfg(not(feature = "dyntensor"))]
impl<T: Copy + Default, const N: usize, const R: usize> TensorLike<T> for Tensor<T, N, R> {
    fn __new_with_data(data: &[T], shape: &[usize]) -> Self {
        Self::new(
            shape.try_into().expect("invalid shape for const rank"),
            data.try_into().expect("invalid data length for const size"),
        )
    }
}
#[cfg(feature = "dyntensor")]
impl<T: Copy + Default> TensorLike<T> for Tensor<T> {
    fn __new_with_data(data: &[T], shape: &[usize]) -> Self {
        Self::new(shape, data)
    }
}
