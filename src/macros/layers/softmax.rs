use super::{
    softmax, Backward, Closure, Layer, Tensor, TensorFloat, WithGrad,
};

/// Defines a layer that uses softmax to propagate probability distributions.
pub struct Softmax<const RANK: usize, const SIZE: usize> {
    /// The shape of the layer.
    pub shape: [usize; RANK],

    /// The data of the layer.
    pub data: [TensorFloat; SIZE],

    /// The temperature of the softmax layer.
    pub temp: f32,
}

impl<const RANK: usize, const SIZE: usize>
    Softmax<RANK, SIZE>
{
    /// Builds the structure into a compute layer.
    #[must_use]
    pub const fn build(self) -> SoftmaxLayer<RANK, SIZE> {
        SoftmaxLayer {
            temp: self.temp,
        }
    }
}

/// A layer that uses softmax to propagate probability distributions.
pub struct SoftmaxLayer<const RANK: usize, const SIZE: usize> {
    temp: f32,
}

impl<const RANK: usize, const SIZE: usize> SoftmaxLayer<RANK, SIZE> {
    /// Gets the temperature of the layer.
    #[must_use]
    pub const fn get_temp_relative(&self, _: f32) -> f32 {
        self.temp
    }

    /// Updates the temperature of the layer.
    pub fn update_temp<F>(&mut self, f: F)
    where 
        F: Fn(f32) -> f32,
    {
        self.temp = f(self.temp);
    }
}

#[cfg(feature = "dyntensor")]
impl<const RANK: usize, const SIZE: usize> SoftmaxLayer<RANK, SIZE> {
    /// Forwards the softmax layer.
    #[inline]
    #[must_use]
    pub fn forward<'a>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, 0>>,
    ) -> (Tensor<RANK, 0>, Backward<'a, RANK, 0, 0, 0>) {
        let (out, f) = softmax(input);

        (out, Backward::Unary(f))
    }

    /// Differentiates the softmax layer with the provided closure.
    #[inline]
    #[must_use]
    #[allow(clippy::unnecessary_wraps)]
    pub fn backward(
        &self,
        grad_output: Tensor<RANK, 0>,
        back: Backward<'_, RANK, 0, 0, 0>,
    ) -> (Tensor<RANK, 0>, Option<Tensor<RANK, 0>>) {
        // call the backward closure and return the results
        match back {
            Backward::Unary(f) => {
                let grad_in = f.invoke(grad_output);
                (grad_in, None)
            }
            Backward::Binary(_) => {
                unreachable!("Softmax never has a binary closure");
            }
        }
    }
}

#[cfg(not(feature = "dyntensor"))]
impl<const RANK: usize, const SIZE: usize> SoftmaxLayer<RANK, SIZE> {
    /// Forwards the softmax layer.
    #[inline]
    #[must_use]
    pub fn forward<'a, const OUT_SIZE: usize>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, SIZE>>,
    ) -> (Tensor<RANK, OUT_SIZE>, Backward<'a, RANK, OUT_SIZE, OUT_SIZE, SIZE>) {
        let (out, f) = softmax(input);

        (out, Backward::Unary(f))
    }

    /// Differentiates the softmax layer with the provided closure.
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
                unreachable!("Softmax never has a binary closure");
            }
        }
    }
}

impl<const RANK: usize, const IN_SIZE: usize> Layer<RANK, IN_SIZE, 0>
    for SoftmaxLayer<RANK, IN_SIZE>
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
