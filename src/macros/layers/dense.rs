use super::{matmul, Backward, Closure, IntoWithGrad, Layer, Tensor, TensorFloat, WithGrad};

/// A representation of a dense layer builder.
///
/// Performs a left-hand multiplication: `output = input * W`
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
///
/// Performs a left-hand multiplication: `output = input * W`
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
        Backward<'a, RANK, OUT_SIZE, SIZE, IN_SIZE>,
    ) {
        let (out, back) = matmul(&self.weights, input);
        (out, Backward::Binary(back))
    }

    /// Forwards the activation layer.
    #[must_use]
    #[inline]
    #[cfg(feature = "dyntensor")]
    pub fn forward<'a>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, 0>>,
    ) -> (Tensor<RANK, 0>, Backward<'a, RANK, 0, SIZE, 0>) {
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
        Backward<'a, RANK, OUT_SIZE, SIZE, IN_SIZE>,
    ) {
        self.__forward(input)
    }

    #[must_use]
    #[inline]
    #[allow(clippy::unnecessary_wraps, clippy::unused_self)] // for consistent API
    fn __backward<const IN_SIZE: usize, const OUT_SIZE: usize>(
        &self,
        grad_output: Tensor<RANK, OUT_SIZE>,
        back: Backward<'_, RANK, OUT_SIZE, SIZE, IN_SIZE>,
    ) -> (Tensor<RANK, IN_SIZE>, Option<Tensor<RANK, SIZE>>) {
        // call the backward closure and return the results
        match back {
            Backward::Unary(_) => {
                unreachable!("Dense always has a binary closure");
            }
            Backward::Binary(f) => {
                let (grad_in, grad_w) = f.invoke(grad_output);

                (grad_w, Some(grad_in))
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
        back: Backward<'_, RANK, 0, SIZE, 0>,
    ) -> (Tensor<RANK, 0>, Option<Tensor<RANK, SIZE>>) {
        self.__backward(grad_output, back)
    }
    /// Differentiates the Dense layer with the provided closure.
    #[must_use]
    #[inline]
    #[cfg(not(feature = "dyntensor"))]
    pub fn backward<const IN_SIZE: usize, const OUT_SIZE: usize>(
        &self,
        grad_output: Tensor<RANK, OUT_SIZE>,
        back: Backward<'_, RANK, OUT_SIZE, SIZE, IN_SIZE>,
    ) -> (Tensor<RANK, IN_SIZE>, Option<Tensor<RANK, SIZE>>) {
        self.__backward(grad_output, back)
    }
}

impl<const RANK: usize, const IN_SIZE: usize> Layer<RANK, IN_SIZE, 1>
    for DenseLayer<RANK, IN_SIZE>
{
    #[inline]
    fn weights(&self) -> [&WithGrad<Tensor<RANK, IN_SIZE>>; 1] {
        [&self.weights]
    }

    #[inline]
    fn weights_mut(&mut self) -> [&mut WithGrad<Tensor<RANK, IN_SIZE>>; 1] {
        [&mut self.weights]
    }
}
