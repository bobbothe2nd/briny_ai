use super::{matmul, Backward, Closure, IntoWithGrad, Layer, Tensor, TensorFloat, WithGrad};

/// A representation of a collapse layer builder.
///
/// Performs a right-hand multiplication: `output = W * input`
pub struct Collapse<const RANK: usize, const SIZE: usize> {
    /// The shape of the layer.
    pub shape: [usize; RANK],

    /// The data of the layer.
    pub data: [TensorFloat; SIZE],
}

impl<const RANK: usize, const SIZE: usize> Collapse<RANK, SIZE> {
    /// Builds the structure into a compute layer.
    #[must_use]
    pub fn build(self) -> CollapseLayer<RANK, SIZE> {
        CollapseLayer {
            weights: Tensor::new(&self.shape, &self.data).with_grad(),
        }
    }
}

/// An expanded representation of a collapse (implicitly transposed dense) layer.
///
/// Performs a right-hand multiplication: `output = W * input`
#[repr(transparent)]
pub struct CollapseLayer<const RANK: usize, const SIZE: usize> {
    weights: WithGrad<Tensor<RANK, SIZE>>,
}

impl<const RANK: usize, const SIZE: usize> CollapseLayer<RANK, SIZE> {
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
                unreachable!("Collapse always has a binary closure");
            }
            Backward::Binary(f) => {
                let (grad_in, grad_w) = f.invoke(grad_output);

                (grad_in, Some(grad_w))
            }
        }
    }

    /// Differentiates the Collapse layer with the provided closure.
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
    /// Differentiates the Collapse layer with the provided closure.
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

impl<const RANK: usize, const IN_SIZE: usize> Layer<RANK, IN_SIZE, 1>
    for CollapseLayer<RANK, IN_SIZE>
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
