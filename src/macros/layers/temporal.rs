use super::{
    matmul, softmax, Backward, Closure, IntoWithGrad, Layer, Tensor, TensorFloat, TensorOps,
    WithGrad,
};

/// A representation of a temporal (time series) layer builder.
pub struct Temporal<const RANK: usize, const SIZE: usize> {
    /// The shape of the layer.
    pub shape: [usize; RANK],

    /// The data of the layer.
    pub data: [TensorFloat; SIZE],
}

impl<const RANK: usize, const SIZE: usize> Temporal<RANK, SIZE> {
    /// Builds the structure into a compute layer.
    #[must_use]
    pub fn build(self) -> TemporalLayer<RANK, SIZE> {
        TemporalLayer {
            attn: Tensor::new(&self.shape, &self.data).with_grad(),
        }
    }
}

/// An expanded representation of a temporal (time series) layer.
///
/// Implements scaled dot-product self-attention mechanism.
/// The weights are organized as: [`Q | K | V | Output`] where each is `d_model x d_model`.
/// `PROJ_SIZE` must equal `SIZE / 4` (each projection is `PROJ_SIZE` elements).
pub struct TemporalLayer<const RANK: usize, const SIZE: usize> {
    attn: WithGrad<Tensor<RANK, SIZE>>,
}

impl<const RANK: usize, const SIZE: usize> TemporalLayer<RANK, SIZE> {
    /// Forwards the temporal layer.
    #[must_use]
    #[cfg(feature = "dyntensor")]
    pub fn forward<'a>(
        &'a self,
        input: &'a WithGrad<Tensor<0, 0>>,
    ) -> (Tensor<0, 0>, Backward<'a, 0, 0, 0, 0>) {
        let scale = 1.0 / libm::sqrtf(self.attn.get_value().shape()[RANK - 1] as f32) as TensorFloat;
        let scaled = self.attn.get_value().map(|&x| x * scale).with_grad();

        let (attn_sm, _) = softmax(&scaled);

        let (out, _) = matmul(input, &attn_sm.clone().with_grad());

        let back = move |grad_output: Tensor<0, 0>| {
            let attn_t = attn_sm.transpose();
            grad_output.matmul(&attn_t)
        };

        (out, Backward::Unary(alloc::boxed::Box::new(back)))
    }

    /// Forwards the temporal layer.
    #[must_use]
    #[cfg(not(feature = "dyntensor"))]
    pub fn forward<'a, const IN_SIZE: usize, const OUT_SIZE: usize>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, IN_SIZE>>,
    ) -> (Tensor<RANK, OUT_SIZE>, Backward<'a, RANK, SIZE, OUT_SIZE, IN_SIZE>) {
        let scale = 1.0 / libm::sqrtf(self.attn.get_value().shape()[RANK - 1] as f32) as TensorFloat;
        let scaled = self.attn.get_value().map(|x| x * scale).with_grad();

        let (attn_sm, _) = softmax::<SIZE, SIZE, RANK>(&scaled);

        let (out, _) = matmul(input, &attn_sm.clone().with_grad());

        let back = move |grad_output: Tensor<RANK, OUT_SIZE>| {
            let attn_t = attn_sm.transpose();
            grad_output.matmul(&attn_t)
        };

        #[cfg(feature = "alloc")]
        {
            (out, Backward::Unary(alloc::boxed::Box::new(back)))
        }
        #[cfg(not(feature = "alloc"))]
        {
            (out, Backward::Unary(box_closure::OpaqueFn::new(back)))
        }
    }

    /// Differentiates the temporal layer.
    #[must_use]
    #[cfg(feature = "dyntensor")]
    pub fn backward(
        &self,
        grad_output: Tensor<0, 0>,
        back: Backward<'_, 0, 0, 0, 0>,
    ) -> (Tensor<0, 0>, Option<Tensor<0, 0>>) {
        let d_input = match back {
            Backward::Binary(_) => unreachable!("Temporal never has a binary closure"),
            Backward::Unary(f) => f.invoke(grad_output),
        };
        (d_input, None)
    }

    /// Differentiates the temporal layer.
    #[must_use]
    #[cfg(not(feature = "dyntensor"))]
    pub fn backward<const IN_SIZE: usize, const OUT_SIZE: usize>(
        &self,
        grad_output: Tensor<RANK, IN_SIZE>,
        back: Backward<'_, RANK, SIZE, IN_SIZE, OUT_SIZE>,
    ) -> (Tensor<RANK, OUT_SIZE>, Option<Tensor<RANK, SIZE>>) {
        let d_input = match back {
            Backward::Binary(_) => unreachable!("Temporal never has a binary closure"),
            Backward::Unary(f) => f.invoke(grad_output),
        };
        (d_input, None)
    }
}

impl<const RANK: usize, const IN_SIZE: usize> Layer<RANK, IN_SIZE, 1>
    for TemporalLayer<RANK, IN_SIZE>
{
    #[inline]
    fn weights(&self) -> [&WithGrad<Tensor<RANK, IN_SIZE>>; 1] {
        [&self.attn]
    }

    #[inline]
    fn weights_mut(&mut self) -> [&mut WithGrad<Tensor<RANK, IN_SIZE>>; 1] {
        [&mut self.attn]
    }
}
