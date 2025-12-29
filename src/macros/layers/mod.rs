mod activation;
pub use self::activation::{Activation, ActivationLayer};

mod attn;
pub use self::attn::{Attention, AttentionLayer};

mod collapse;
pub use self::collapse::{Collapse, CollapseLayer};

mod dense;
pub use self::dense::{Dense, DenseLayer};

mod softmax;
pub use self::softmax::{Softmax, SoftmaxLayer};

mod temporal;
pub use self::temporal::{Temporal, TemporalLayer};

use tensor_optim::TensorOps;

use super::{
    gelu, matmul, relu, sigmoid, swish, tanh, ActivationFn, ActivationKind, Backward, Closure,
    IntoWithGrad, Tensor, TensorFloat, WithGrad,
};

use crate::nn::ops::dispatch::softmax;

#[cfg(not(feature = "dyntensor"))]
use crate::nn::tensors::ConstTensorOps;

use core::marker::PhantomData;

/// An abstraction over all layers of each required function.
pub trait Layer<const RANK: usize, const IN_SIZE: usize, const N: usize> {
    // forward/backward is layer-specific

    /// Immutably obtains a reference to the weights.
    fn weights(&self) -> [&WithGrad<Tensor<RANK, IN_SIZE>>; N];

    /// Mutably obtains a reference to the weights.
    fn weights_mut(&mut self) -> [&mut WithGrad<Tensor<RANK, IN_SIZE>>; N];

    /// Zeroes the gradients of the weights mutably.
    #[inline]
    fn zero_grad(&mut self) {
        for w in self.weights_mut() {
            #[cfg(feature = "dyntensor")]
            let shape = w.get_grad().shape();
            #[cfg(not(feature = "dyntensor"))]
            let shape = w.get_grad().shape_array();

            let tensor = Tensor::zeros(shape);

            // set gradient to zeroed tensor of current shape
            w.set_grad(tensor);
        }
    }

    /// Applies an optimizer update to the weights.
    #[inline]
    fn apply_update(
        &mut self,
        lr: TensorFloat,
        optim: fn(&mut WithGrad<Tensor<RANK, IN_SIZE>>, TensorFloat),
    ) {
        // applies an update
        for w in self.weights_mut() {
            optim(w, lr);
        }
    }
}
