use super::{
    matmul, softmax, Backward, Closure, IntoWithGrad, Layer, Tensor, TensorFloat, TensorOps,
    WithGrad,
};

#[cfg(feature = "dyntensor")]
use crate::nn::tensors::TensorGrad;

#[cfg(not(feature = "dyntensor"))]
use super::ConstTensorOps;

/// A representation of a transformer layer builder.
///
/// The transformer implements scaled dot-product self-attention.
/// For input shape [`seq_len, d_model`], the weights shape should be [`d_model, 4 * d_model`]
/// where the weights are organized as: [`Q_weights | K_weights | V_weights | Output_weights`]
/// Each weight matrix is `d_model x d_model`, so `PROJ_SIZE = d_model^2` and `SIZE = 4 * PROJ_SIZE`.
pub struct Attention<const RANK: usize, const SIZE: usize> {
    /// The shape of the layer.
    pub shape: [usize; RANK],

    /// The data of the layer.
    pub data: [TensorFloat; SIZE],
}

impl<const RANK: usize, const SIZE: usize> Attention<RANK, SIZE> {
    /// Builds the structure into a compute layer.
    #[must_use]
    pub fn build(self) -> AttentionLayer<RANK, SIZE> {
        AttentionLayer {
            w_q: Tensor::new(&self.shape, &self.data).with_grad(),
            w_k: Tensor::new(&self.shape, &self.data).with_grad(),
            w_v: Tensor::new(&self.shape, &self.data).with_grad(),
        }
    }
}

/// An expanded representation of a transformer layer.
///
/// Implements scaled dot-product self-attention mechanism.
/// The weights are organized as: [`Q | K | V | Output`] where each is `d_model x d_model`.
/// `PROJ_SIZE` must equal `SIZE / 4` (each projection is `PROJ_SIZE` elements).
pub struct AttentionLayer<const RANK: usize, const SIZE: usize> {
    w_q: WithGrad<Tensor<RANK, SIZE>>,
    w_k: WithGrad<Tensor<RANK, SIZE>>,
    w_v: WithGrad<Tensor<RANK, SIZE>>,
}

impl<const RANK: usize, const SIZE: usize> AttentionLayer<RANK, SIZE> {
    /// Forwards the attention layer.
    #[must_use]
    #[cfg(feature = "dyntensor")]
    pub fn forward<'a>(
        &'a self,
        input: &'a WithGrad<Tensor<0, 0>>,
    ) -> (Tensor<0, 0>, Backward<'a, 0, 0, 0, 0>) {
        let (q, _) = matmul(input, &self.w_q);
        let (k, _) = matmul(input, &self.w_k);
        let (v, _) = matmul(input, &self.w_v);

        let k_t = k.transpose();
        let (scores, _) = matmul(&q.clone().with_grad(), &k_t.with_grad());

        let scale = (1.0 / libm::sqrtf(input.get_value().len() as f32)) as TensorFloat;

        let scaled_scores = scores.map(|x| x * scale);
        let scaled_scores_wg = scaled_scores.with_grad();

        let (attn, _) = softmax(&scaled_scores_wg);
        let (out, _) = matmul(&attn.clone().with_grad(), &v.with_grad());

        let back = move |grad_output: Tensor<0, 0>| {
            let attn_t = attn.transpose();
            let d_v = attn_t.matmul(&grad_output);

            let grad_data = grad_output.data();
            let mut grad = alloc::vec![0.0; input.get_value().len()];

            let outer_size = attn.shape()[input.get_value().shape().len() - 2];
            let last_dim = attn.shape()[input.get_value().shape().len() - 1];

            for i in 0..outer_size {
                let offset = i * last_dim;
                let y = &scaled_scores_wg.get_value().data()[offset..offset + last_dim];
                let dy = &grad_data[offset..offset + last_dim];

                let dot: TensorFloat = y.iter().zip(dy.iter()).map(|(&yi, &dyi)| yi * dyi).sum();

                for j in 0..last_dim {
                    grad[offset + j] = y[j] * (dy[j] - dot);
                }
            }

            let d_scores = Tensor::new(attn.shape(), &grad);

            let k_t = k.transpose();
            let d_q = d_scores.matmul(&k_t);
            let d_k = d_scores.transpose().matmul(&q);

            d_q + d_k + d_v
        };

        (out, Backward::Unary(alloc::boxed::Box::new(back)))
    }

    /// Forwards the attention layer.
    #[must_use]
    #[cfg(not(feature = "dyntensor"))]
    pub fn forward<'a>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, SIZE>>,
    ) -> (Tensor<RANK, SIZE>, Backward<'a, RANK, SIZE, SIZE, SIZE>) {
        let (q, _) = matmul::<SIZE, SIZE, SIZE, RANK>(input, &self.w_q);
        let (k, _) = matmul::<SIZE, SIZE, SIZE, RANK>(input, &self.w_k);
        let (v, _) = matmul::<SIZE, SIZE, SIZE, RANK>(input, &self.w_v);

        let k_t = k.transpose();
        let (scores, _): (Tensor<RANK, SIZE>, _) = matmul(&q.clone().with_grad(), &k_t.with_grad());

        let scale = (1.0 / libm::sqrtf(SIZE as f32)) as TensorFloat;

        let scaled_scores = scores.map(|x| x * scale);
        let scaled_scores_wg = scaled_scores.with_grad();

        let (attn, _): (Tensor<RANK, SIZE>, _) = softmax(&scaled_scores_wg);
        let (out, _) = matmul(&attn.clone().with_grad(), &v.with_grad());

        let back = move |grad_output: Tensor<RANK, SIZE>| {
            let attn_t = attn.transpose();
            let d_v: Tensor<RANK, SIZE> = attn_t.matmul(&grad_output);

            let grad_data = grad_output.data();
            let mut grad = [0.0; SIZE];

            let outer_size = attn.shape()[RANK - 2];
            let last_dim = attn.shape()[RANK - 1];

            for i in 0..outer_size {
                let offset = i * last_dim;
                let y = &attn.data()[offset..offset + last_dim];
                let dy = &grad_data[offset..offset + last_dim];

                let dot: TensorFloat = y.iter().zip(dy.iter()).map(|(&yi, &dyi)| yi * dyi).sum();

                for j in 0..last_dim {
                    grad[offset + j] = y[j] * (dy[j] - dot);
                }
            }

            let d_scores = Tensor::new(attn.shape_array(), &grad);

            let k_t = k.transpose();
            let d_q = d_scores.matmul(&k_t);
            let d_k: Tensor<RANK, SIZE> = d_scores.transpose().matmul(&q);

            d_q + d_k + d_v
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

    /// Differentiates the attention layer.
    #[must_use]
    #[cfg(feature = "dyntensor")]
    pub fn backward(
        &self,
        grad_output: Tensor<0, 0>,
        back: Backward<'_, 0, 0, 0, 0>,
    ) -> (Tensor<0, 0>, Option<Tensor<0, 0>>) {
        let d_input = match back {
            Backward::Binary(_) => unreachable!("Attention never has a binary closure"),
            Backward::Unary(f) => f.invoke(grad_output),
        };
        (d_input, None)
    }

    /// Differentiates the attention layer.
    #[must_use]
    #[cfg(not(feature = "dyntensor"))]
    pub fn backward(
        &self,
        grad_output: Tensor<RANK, SIZE>,
        back: Backward<'_, RANK, SIZE, SIZE, SIZE>,
    ) -> (Tensor<RANK, SIZE>, Option<Tensor<RANK, SIZE>>) {
        let d_input = match back {
            Backward::Binary(_) => unreachable!("Attention never has a binary closure"),
            Backward::Unary(f) => f.invoke(grad_output),
        };
        (d_input, None)
    }
}

impl<const RANK: usize, const IN_SIZE: usize> Layer<RANK, IN_SIZE, 3>
    for AttentionLayer<RANK, IN_SIZE>
{
    #[inline]
    fn weights(&self) -> [&WithGrad<Tensor<RANK, IN_SIZE>>; 3] {
        [&self.w_q, &self.w_k, &self.w_v]
    }

    #[inline]
    fn weights_mut(&mut self) -> [&mut WithGrad<Tensor<RANK, IN_SIZE>>; 3] {
        [&mut self.w_q, &mut self.w_k, &mut self.w_v]
    }
}
