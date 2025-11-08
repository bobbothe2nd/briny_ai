use super::{
    matmul, relu, ActivationFn, ActivationKind, Backward, Closure, IntoWithGrad, Tensor,
    TensorFloat, WithGrad,
};

#[cfg(feature = "dyntensor")]
use crate::nn::tensors::TensorOps;

#[cfg(not(feature = "dyntensor"))]
use crate::nn::tensors::ConstTensorOps;

use core::marker::PhantomData;

/// An abstraction over all layers of each required function.
pub trait Layer<const RANK: usize, const IN_SIZE: usize> {
    // forward/backward is layer-specific

    /// Immutably obtains a reference to the weights.
    fn weights(&self) -> &WithGrad<Tensor<RANK, IN_SIZE>>;

    /// Mutably obtains a reference to the weights.
    fn weights_mut(&mut self) -> &mut WithGrad<Tensor<RANK, IN_SIZE>>;

    /// Moves the weights out of the layer by value.
    #[must_use]
    fn into_weights(self) -> WithGrad<Tensor<RANK, IN_SIZE>>;

    /// Zeroes the gradients of the weights mutably.
    #[inline]
    fn zero_grad(&mut self) {
        #[cfg(feature = "dyntensor")]
        let shape = self.weights().get_grad().shape();
        #[cfg(not(feature = "dyntensor"))]
        let shape = self.weights().get_grad().shape_array();

        let tensor = Tensor::zeros(shape);

        // set gradient to zeroed tensor of current shape
        self.weights_mut().set_grad(tensor);
    }

    /// Applies an optimizer update to the weights.
    #[inline]
    fn apply_update(
        &mut self,
        lr: TensorFloat,
        optim: fn(&mut WithGrad<Tensor<RANK, IN_SIZE>>, TensorFloat),
    ) {
        // applies an update
        optim(self.weights_mut(), lr);
    }
}

/// A representation of a dense layer builder.
pub struct Dense<const RANK: usize, const SIZE: usize> {
    /// The shape of the layer.
    pub shape: [usize; RANK],

    /// The data of the layer.
    pub data: [TensorFloat; SIZE],
}

impl<const RANK: usize, const SIZE: usize> Dense<RANK, SIZE> {
    /// Builds the structure into a compute layer.
    #[must_use]
    pub fn build(self) -> DenseLayer<RANK, SIZE> {
        DenseLayer {
            weights: Tensor::new(&self.shape, &self.data).with_grad(),
        }
    }
}

/// An expanded representation of a dense layer.
#[repr(transparent)]
pub struct DenseLayer<const RANK: usize, const SIZE: usize> {
    weights: WithGrad<Tensor<RANK, SIZE>>,
}

impl<const RANK: usize, const SIZE: usize> DenseLayer<RANK, SIZE> {
    #[must_use]
    #[inline]
    fn __forward<'a, const OUT_SIZE: usize, const IN_SIZE: usize>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, IN_SIZE>>,
    ) -> (
        Tensor<RANK, OUT_SIZE>,
        Backward<'a, RANK, OUT_SIZE, IN_SIZE, SIZE>,
    ) {
        let (out, back) = matmul(input, &self.weights);
        (out, Backward::Binary(back))
    }

    /// Forwards the activation layer.
    #[must_use]
    #[inline]
    #[cfg(feature = "dyntensor")]
    pub fn forward<'a>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, 0>>,
    ) -> (Tensor<RANK, 0>, Backward<'a, RANK, 0, 0, SIZE>) {
        self.__forward(input)
    }
    /// Forwards the activation layer.
    #[must_use]
    #[inline]
    #[cfg(not(feature = "dyntensor"))]
    pub fn forward<'a, const OUT_SIZE: usize, const IN_SIZE: usize>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, IN_SIZE>>,
    ) -> (
        Tensor<RANK, OUT_SIZE>,
        Backward<'a, RANK, OUT_SIZE, IN_SIZE, SIZE>,
    ) {
        self.__forward(input)
    }

    #[must_use]
    #[inline]
    #[allow(clippy::unnecessary_wraps, clippy::unused_self)] // for consistent API
    fn __backward<const IN_SIZE: usize, const OUT_SIZE: usize>(
        &self,
        grad_output: Tensor<RANK, OUT_SIZE>,
        back: Backward<'_, RANK, OUT_SIZE, IN_SIZE, SIZE>,
    ) -> (Tensor<RANK, IN_SIZE>, Option<Tensor<RANK, SIZE>>) {
        // call the backward closure and return the results
        match back {
            Backward::Unary(_) => {
                unreachable!("Dense always has a binary closure");
            }
            Backward::Binary(f) => {
                let (grad_in, grad_w) = f.invoke(grad_output);

                (grad_in, Some(grad_w))
            }
        }
    }

    /// Differentiates the Dense layer with the provided closure.
    #[must_use]
    #[inline]
    #[cfg(feature = "dyntensor")]
    pub fn backward(
        &self,
        grad_output: Tensor<0, 0>,
        back: Backward<'_, RANK, 0, 0, SIZE>,
    ) -> (Tensor<RANK, 0>, Option<Tensor<RANK, 0>>) {
        self.__backward(grad_output, back)
    }
    /// Differentiates the Dense layer with the provided closure.
    #[must_use]
    #[inline]
    #[cfg(not(feature = "dyntensor"))]
    pub fn backward<const IN_SIZE: usize, const OUT_SIZE: usize>(
        &self,
        grad_output: Tensor<RANK, OUT_SIZE>,
        back: Backward<'_, RANK, OUT_SIZE, IN_SIZE, SIZE>,
    ) -> (Tensor<RANK, IN_SIZE>, Option<Tensor<RANK, SIZE>>) {
        self.__backward(grad_output, back)
    }
}

impl<const RANK: usize, const IN_SIZE: usize> Layer<RANK, IN_SIZE> for DenseLayer<RANK, IN_SIZE> {
    fn weights(&self) -> &WithGrad<Tensor<RANK, IN_SIZE>> {
        &self.weights
    }

    fn weights_mut(&mut self) -> &mut WithGrad<Tensor<RANK, IN_SIZE>> {
        &mut self.weights
    }

    fn into_weights(self) -> WithGrad<Tensor<RANK, IN_SIZE>> {
        self.weights
    }
}

/// Defines a layer that propagates gradients via an activation function.
pub struct Activation<const RANK: usize, const SIZE: usize, ActivationFn> {
    /// The shape of the layer.
    pub shape: [usize; RANK],

    /// The data of the layer.
    pub data: [TensorFloat; SIZE],

    /// The activation function used by the layer.
    pub _activation: PhantomData<ActivationFn>,
}

impl<const RANK: usize, const SIZE: usize, Activator: ActivationFn>
    Activation<RANK, SIZE, Activator>
{
    /// Builds the structure into a compute layer.
    #[must_use]
    pub fn build(self) -> ActivationLayer<RANK, SIZE> {
        ActivationLayer {
            weights: Tensor::new(&self.shape, &self.data).with_grad(),
            actfn: Activator::kind(),
        }
    }
}

/// A layer that uses the provided activation function to propagate gradients.
pub struct ActivationLayer<const RANK: usize, const SIZE: usize> {
    weights: WithGrad<Tensor<RANK, SIZE>>,
    actfn: ActivationKind,
}

impl<const RANK: usize, const SIZE: usize> ActivationLayer<RANK, SIZE> {
    /// Forwards the activation layer.
    #[must_use]
    #[inline]
    pub fn forward<'a>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, SIZE>>,
    ) -> (Tensor<RANK, SIZE>, Backward<'a, RANK, SIZE, SIZE, SIZE>) {
        match self.actfn {
            ActivationKind::ReLU => {
                let (out, back) = relu(input);
                (out, Backward::Unary(back))
            }
            ActivationKind::Sigmoid => unimplemented!("activation not implemented"),
        }
    }

    /// Differentiates the activation layer with the provided closure.
    #[must_use]
    #[allow(clippy::unnecessary_wraps)]
    #[inline]
    pub fn backward(
        &self,
        grad_output: Tensor<RANK, SIZE>,
        back: Backward<'_, RANK, SIZE, SIZE, SIZE>,
    ) -> (Tensor<RANK, SIZE>, Option<Tensor<RANK, SIZE>>) {
        // call the backward closure and return the results
        match back {
            Backward::Unary(f) => {
                let grad_in = f.invoke(grad_output);
                (grad_in, None)
            }
            Backward::Binary(_) => {
                unreachable!("activation never has a binary closure");
            }
        }
    }
}

impl<const RANK: usize, const IN_SIZE: usize> Layer<RANK, IN_SIZE>
    for ActivationLayer<RANK, IN_SIZE>
{
    fn weights(&self) -> &WithGrad<Tensor<RANK, IN_SIZE>> {
        &self.weights
    }

    fn weights_mut(&mut self) -> &mut WithGrad<Tensor<RANK, IN_SIZE>> {
        &mut self.weights
    }

    fn into_weights(self) -> WithGrad<Tensor<RANK, IN_SIZE>> {
        self.weights
    }
}
