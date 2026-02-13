mod activation;
pub use self::activation::{Activation, ActivationLayer};

mod attn;
pub use self::attn::{SelfAttention, SelfAttentionLayer};

mod bias;
pub use self::bias::{Bias, BiasLayer};

mod causal_attn;
pub use self::causal_attn::{CausalSelfAttention, CausalSelfAttentionLayer};

mod collapse;
pub use self::collapse::{Collapse, CollapseLayer};

mod conv;
pub use self::conv::{Conv, ConvLayer};

mod dense;
pub use self::dense::{Dense, DenseLayer};

mod ffn;
pub use self::ffn::{FeedForward, FeedForwardLayer};

mod ln;
pub use self::ln::{LayerNorm, LayerNormLayer};

mod residual;
pub use self::residual::{Residual, ResidualLayer};

mod sigmoid;
pub use self::sigmoid::{Sigmoid, SigmoidLayer};

mod softmax;
pub use self::softmax::{Softmax, SoftmaxLayer};

use super::{
    gelu, matmul, relu, sigmoid, swish, tanh, ActivationFn, ActivationKind, Backward, ClosureOnce,
    IntoWithGrad, Tensor, TensorFloat, WithGrad,
};

use crate::macros::optim::Optim;
use crate::nn::ops::dispatch::softmax;

use core::marker::PhantomData;

#[cfg(not(feature = "dyntensor"))]
use crate::nn::tensors::ConstTensorOps;

#[cfg(feature = "dyntensor")]
use tensor_optim::TensorOps;

/// Performs heavy layer operations (different input, intermediate, and output sizes).
pub trait LayerOpHeavy<const RANK: usize, const SIZE: usize, const N: usize> {
    /// Performs forward evaluation.
    #[must_use]
    fn forward<'a, const IN_SIZE: usize, const W_SIZE: usize, const OUT_SIZE: usize>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, IN_SIZE>>,
    ) -> (
        Tensor<RANK, OUT_SIZE>,
        Backward<'a, RANK, OUT_SIZE, SIZE, IN_SIZE>,
    );

    /// Executes backwards closure on gradient.
    #[must_use]
    fn backward<const IN_SIZE: usize, const OUT_SIZE: usize>(
        &self,
        grad_out: Tensor<RANK, OUT_SIZE>,
        back: Backward<'_, RANK, OUT_SIZE, SIZE, IN_SIZE>,
    ) -> (Tensor<RANK, IN_SIZE>, [Tensor<RANK, SIZE>; N]);
}

/// Expands a layer.
pub trait BuildLayer<const RANK: usize, const SIZE: usize, const N: usize> {
    /// The expanded layer.
    type Layer: Layer<RANK, SIZE, N>;

    /// Constructs the operation layer.
    #[must_use]
    fn build(self) -> Self::Layer;
}

/// An abstraction over all layers of each required function.
pub trait Layer<const RANK: usize, const SIZE: usize, const N: usize> {
    /// Splits mutable references for both the optimizer and the weights.
    #[must_use]
    fn optim_weights(
        &mut self,
    ) -> (
        &mut impl Optim<RANK, SIZE>,
        [&mut WithGrad<Tensor<RANK, SIZE>>; N],
    );

    /// Immutably obtains a reference to the weights.
    #[must_use]
    fn weights(&self) -> [&WithGrad<Tensor<RANK, SIZE>>; N];

    /// Mutably obtains a reference to the weights.
    #[must_use]
    fn weights_mut(&mut self) -> [&mut WithGrad<Tensor<RANK, SIZE>>; N];

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
    fn apply_update(&mut self, lr: TensorFloat) {
        let (optim, weights) = self.optim_weights();
        // applies an update
        for w in weights {
            optim.step(w, lr);
        }
    }
}
