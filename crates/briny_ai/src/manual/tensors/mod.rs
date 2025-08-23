//! Flexible tensor backends based on features and targets.

mod optim;
#[cfg(feature = "alloc")]
mod vec;

#[cfg(not(feature = "dyntensor"))]
pub use tensor_optim::ConstTensorOps;
pub use tensor_optim::TensorOps;

pub use self::optim::{Flatten, Tensor, TensorGrad};
#[cfg(feature = "alloc")]
pub use vec::VecTensor;

/// A trait mainly for converting `Tensor`s to `WithGrad`.
pub trait IntoWithGrad: TensorGrad + Sized {
    /// Wraps the tensor with a zero-initialized gradient.
    fn with_grad(self) -> WithGrad<Self> {
        WithGrad::new(self)
    }
}

impl<T: TensorGrad> IntoWithGrad for T {}

/// A container for tracking gradients of values (used in autograd).
///
/// Typically used as `WithGrad<Tensor<f64>>` or `WithGrad<f64>`.
#[derive(Debug, Clone, Default)]
pub struct WithGrad<T> {
    value: T,
    grad: T,
}

impl<T: TensorGrad> WithGrad<T> {
    /// Creates a new `WithGrad`.
    ///
    /// The gradient is zeroed, whereas the value is initialized
    /// witht the data given.
    pub fn new(value: T) -> Self {
        let grad = value.zeros_like();
        Self { value, grad }
    }

    /// Overwrites the gradient.
    pub fn set_grad(&mut self, grad: T) {
        self.grad = grad;
    }

    /// Get immutable references to the items.
    ///
    /// The gradient and the value are both immutable.
    pub const fn split(&self) -> (&T, &T) {
        (&self.value, &self.grad)
    }

    /// Get mutable references to the items.
    ///
    /// The gradient and the value are both mutable.
    pub const fn split_mut(&mut self) -> (&mut T, &mut T) {
        (&mut self.value, &mut self.grad)
    }

    /// Immutably singles out the gradient.
    pub const fn get_grad(&self) -> &T {
        &self.grad
    }

    /// Immutably singles out the value.
    pub const fn get_value(&self) -> &T {
        &self.value
    }

    /// Mutably singles out the gradient.
    pub const fn get_grad_mut(&mut self) -> &mut T {
        &mut self.grad
    }

    /// Mutably singles out the value.
    pub const fn get_value_mut(&mut self) -> &mut T {
        &mut self.value
    }
}
