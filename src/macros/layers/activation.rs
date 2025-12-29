use super::{
    gelu, relu, sigmoid, swish, tanh, ActivationFn, ActivationKind, Backward, Closure,
    Layer, PhantomData, Tensor, TensorFloat, WithGrad,
};

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
            actfn: Activator::kind(),
        }
    }
}

/// A layer that uses the provided activation function to propagate gradients.
pub struct ActivationLayer<const RANK: usize, const SIZE: usize> {
    actfn: ActivationKind,
}

impl<const RANK: usize, const SIZE: usize> ActivationLayer<RANK, SIZE> {
    /// Forwards the activation layer.
    #[inline]
    #[must_use]
    pub fn forward<'a>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, SIZE>>,
    ) -> (Tensor<RANK, SIZE>, Backward<'a, RANK, SIZE, SIZE, SIZE>) {
        match self.actfn {
            ActivationKind::ReLU => {
                let (out, back) = relu(input);
                (out, Backward::Unary(back))
            }
            ActivationKind::Sigmoid => {
                let (out, back) = sigmoid(input);
                (out, Backward::Unary(back))
            }
            ActivationKind::GELU => {
                let (out, back) = gelu(input);
                (out, Backward::Unary(back))
            }
            ActivationKind::Swish => {
                let (out, back) = swish(input);
                (out, Backward::Unary(back))
            }
            ActivationKind::Tanh => {
                let (out, back) = tanh(input);
                (out, Backward::Unary(back))
            }
        }
    }

    /// Differentiates the activation layer with the provided closure.
    #[inline]
    #[must_use]
    #[allow(clippy::unnecessary_wraps)]
    pub fn backward<const IN_SIZE: usize, const OUT_SIZE: usize>(
        &self,
        grad_output: Tensor<RANK, IN_SIZE>,
        back: Backward<'_, RANK, SIZE, IN_SIZE, OUT_SIZE>,
    ) -> (Tensor<RANK, OUT_SIZE>, Option<Tensor<RANK, OUT_SIZE>>) {
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

impl<const RANK: usize, const IN_SIZE: usize> Layer<RANK, IN_SIZE, 0>
    for ActivationLayer<RANK, IN_SIZE>
{
    #[inline]
    fn weights(&self) -> [&WithGrad<Tensor<RANK, IN_SIZE>>; 0] {
        []
    }

    #[inline]
    fn weights_mut(&mut self) -> [&mut WithGrad<Tensor<RANK, IN_SIZE>>; 0] {
        []
    }
}
