#![cfg_attr(feature = "dyntensor", allow(clippy::useless_conversion))]
#![allow(clippy::similar_names, clippy::cast_precision_loss)]

use super::{softmax, Backward, ClosureOnce, IntoWithGrad, Layer, Tensor, TensorFloat, WithGrad};
use crate::macros::{optim::Optim, BuildLayer, LayerOpHeavy};
use alloc::boxed::Box;
use core::marker::PhantomData;
use tensor_optim::TensorOps;

fn apply_causal_mask<const RANK: usize, const SIZE: usize>(scores: &mut Tensor<RANK, SIZE>) {
    let seq_len = scores.shape()[RANK - 1];
    for outer in 0..scores.shape()[..(RANK - 2)].iter().product() {
        let offset = outer * seq_len * seq_len;
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                scores.data_mut()[offset + (i * seq_len) + j] = f32::NEG_INFINITY;
            }
        }
    }
}

/// Causal self attention layer builder.
pub struct CausalSelfAttention<const RANK: usize, const SIZE: usize, O> {
    /// Shape of weights.
    pub shape: [usize; RANK],

    /// Initial content of weights.
    pub data: Box<[TensorFloat; SIZE]>,

    /// Optimizer of weights.
    pub _optim: PhantomData<O>,
}

impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>>
    CausalSelfAttention<RANK, SIZE, O>
{
    /// Count of tensors (Q, K, V).
    pub const TENSORS: usize = 3;

    /// Expands the layer.
    #[must_use]
    #[allow(clippy::explicit_auto_deref)]
    pub fn build(self) -> CausalSelfAttentionLayer<RANK, SIZE, O> {
        CausalSelfAttentionLayer {
            w_q: Tensor::new(&self.shape, &*self.data).with_grad(),
            w_k: (Tensor::new(&self.shape, &*self.data) - 0.1).with_grad(),
            w_v: (Tensor::new(&self.shape, &*self.data) + 0.1).with_grad(),
            optim: O::new(&self.shape),
        }
    }
}

impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> BuildLayer<RANK, SIZE, 3>
    for CausalSelfAttention<RANK, SIZE, O>
{
    type Layer = CausalSelfAttentionLayer<RANK, SIZE, O>;

    fn build(self) -> CausalSelfAttentionLayer<RANK, SIZE, O> {
        self.build()
    }
}

/// Causal self attention function.
pub struct CausalSelfAttentionLayer<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> {
    w_q: WithGrad<Tensor<RANK, SIZE>>,
    w_k: WithGrad<Tensor<RANK, SIZE>>,
    w_v: WithGrad<Tensor<RANK, SIZE>>,
    optim: O,
}

impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>>
    CausalSelfAttentionLayer<RANK, SIZE, O>
{
    #[inline]
    #[must_use]
    #[allow(clippy::useless_conversion)]
    fn __forward<'a, const IN_SIZE: usize, const OUT_SIZE: usize, const W_SIZE: usize>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, IN_SIZE>>,
    ) -> (
        Tensor<RANK, OUT_SIZE>,
        Backward<'a, RANK, OUT_SIZE, SIZE, IN_SIZE>,
    ) {
        let input = input.get_value();

        let in_shape = input.shape();
        let scale = 1.0 / libm::sqrtf(in_shape[RANK - 1] as f32);

        let q: Tensor<RANK, IN_SIZE> = input.matmul(self.w_q.get_value());

        let k: Tensor<RANK, IN_SIZE> = input.matmul(self.w_k.get_value());

        let v: Tensor<RANK, IN_SIZE> = input.matmul(self.w_v.get_value());

        let kt = k.transpose();

        let mut scores = q.matmul(&kt);
        apply_causal_mask::<RANK, W_SIZE>(&mut scores);

        let scores: WithGrad<Tensor<RANK, W_SIZE>> = (scores * scale).with_grad();

        let (attn, _): (Tensor<RANK, W_SIZE>, _) = softmax(&scores);

        let out = attn.matmul(&v);

        let back = move |grad_out: Tensor<RANK, OUT_SIZE>| {
            let grad_v: Tensor<RANK, IN_SIZE> = attn.transpose().matmul(&grad_out);
            let grad_attn: Tensor<RANK, W_SIZE> = grad_out.matmul(&v.transpose());

            let grad_scores: Tensor<RANK, W_SIZE> = {
                #[cfg(feature = "no_stack")]
                {
                    let attn_data = attn.data();
                    let shape = attn.shape();
                    let last_dim = shape[shape.len() - 1];
                    let outer_size: usize = shape[..shape.len() - 1].iter().product();

                    let grad_data = grad_attn.data();
                    let mut grad = alloc::vec![0.0; grad_attn.data().len()];

                    for i in 0..outer_size {
                        let offset = i * last_dim;
                        let y = &attn_data[offset..offset + last_dim];
                        let dy = &grad_data[offset..offset + last_dim];

                        let dot: TensorFloat =
                            y.iter().zip(dy.iter()).map(|(&yi, &dyi)| yi * dyi).sum();

                        for j in 0..last_dim {
                            grad[offset + j] = y[j] * (dy[j] - dot);
                        }
                    }

                    unsafe {
                        use crate::nn::tensors::VecTensor;

                        VecTensor::from_vec(shape.try_into().unwrap(), grad)
                            .into_tensor()
                            .unwrap_unchecked()
                    }
                }
                #[cfg(not(feature = "no_stack"))]
                {
                    let attn_data = attn.data();
                    let shape = attn.shape();
                    let last_dim = shape[shape.len() - 1];
                    let outer_size: usize = shape[..shape.len() - 1].iter().product();

                    let grad_data = grad_attn.data();
                    let mut grad = [0.0; W_SIZE];

                    for i in 0..outer_size {
                        let offset = i * last_dim;
                        let y = &attn_data[offset..offset + last_dim];
                        let dy = &grad_data[offset..offset + last_dim];

                        let dot: TensorFloat =
                            y.iter().zip(dy.iter()).map(|(&yi, &dyi)| yi * dyi).sum();

                        for j in 0..last_dim {
                            grad[offset + j] = y[j] * (dy[j] - dot);
                        }
                    }

                    Tensor::new(shape.try_into().unwrap(), &grad)
                }
            } * scale;

            let grad_q: Tensor<RANK, IN_SIZE> = grad_scores.matmul(&k);
            let grad_kt: Tensor<RANK, IN_SIZE> = grad_scores.transpose().matmul(&q);

            let grad_wq = input.transpose().matmul(&grad_q);
            let grad_wk = input.transpose().matmul(&grad_kt);
            let grad_wv = input.transpose().matmul(&grad_v);

            let grad_input = grad_q.matmul(&self.w_q.get_value().transpose())
                + grad_kt.matmul(&self.w_k.get_value().transpose())
                + grad_v.matmul(&self.w_v.get_value().transpose());

            (grad_wq, grad_wk, grad_wv, grad_input)
        };

        #[cfg(feature = "alloc")]
        {
            (
                out,
                Backward::Quaternary(alloc::boxed::Box::new(back)),
            )
        }
        #[cfg(not(feature = "alloc"))]
        {
            (
                out,
                Backward::Quaternary(box_closure::OpaqueFnOnce::new(back)),
            )
        }
    }

    /// Forwards the causal attention layer.
    #[cfg(feature = "dyntensor")]
    pub fn forward<'a>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, 0>>,
    ) -> (Tensor<RANK, 0>, Backward<'a, RANK, 0, SIZE, 0>, [(); 0]) {
        let (out, back) = self.__forward::<0, 0, 0>(input);
        (
            out,
            back,
            [],
        )
    }

    /// Forwards the causal attention layer.
    #[cfg(not(feature = "dyntensor"))]
    pub fn forward<'a, const IN_SIZE: usize, const W_SIZE: usize>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, IN_SIZE>>,
    ) -> (
        Tensor<RANK, IN_SIZE>,
        Backward<'a, RANK, IN_SIZE, SIZE, IN_SIZE>,
        [(); W_SIZE],
    ) {
        let (out, back) = self.__forward::<IN_SIZE, IN_SIZE, W_SIZE>(input);
        (
            out,
            back,
            [(); W_SIZE],
        )
    }

    #[must_use]
    #[inline]
    #[allow(clippy::unused_self)]
    fn __backward<const IN_SIZE: usize, const OUT_SIZE: usize>(
        &self,
        grad_out: Tensor<RANK, OUT_SIZE>,
        back: Backward<'_, RANK, OUT_SIZE, SIZE, IN_SIZE>,
    ) -> (Tensor<RANK, IN_SIZE>, [Tensor<RANK, SIZE>; 3]) {
        match back {
            Backward::Quaternary(f) => {
                let (grad_q, grad_k, grad_v, grad_w) = f.invoke_once(grad_out);
                (grad_w, [grad_q, grad_k, grad_v])
            }
            _ => unreachable!("CausalSelfAttention always has a quaternary closure"),
        }
    }

    /// Differentiates the causal attention layer.
    pub fn backward<const IN_SIZE: usize>(
        &self,
        grad_out: Tensor<RANK, IN_SIZE>,
        back: Backward<'_, RANK, IN_SIZE, SIZE, IN_SIZE>,
    ) -> (Tensor<RANK, IN_SIZE>, [Tensor<RANK, SIZE>; 3]) {
        self.__backward(grad_out, back)
    }
}

impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> Layer<RANK, SIZE, 3>
    for CausalSelfAttentionLayer<RANK, SIZE, O>
{
    fn optim_weights(
        &mut self,
    ) -> (
        &mut impl crate::macros::optim::Optim<RANK, SIZE>,
        [&mut WithGrad<Tensor<RANK, SIZE>>; 3],
    ) {
        (
            &mut self.optim,
            [&mut self.w_q, &mut self.w_k, &mut self.w_v],
        )
    }

    fn weights(&self) -> [&WithGrad<Tensor<RANK, SIZE>>; 3] {
        [&self.w_q, &self.w_k, &self.w_v]
    }

    fn weights_mut(&mut self) -> [&mut WithGrad<Tensor<RANK, SIZE>>; 3] {
        [&mut self.w_q, &mut self.w_k, &mut self.w_v]
    }
}

impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> LayerOpHeavy<RANK, SIZE, 3>
    for CausalSelfAttentionLayer<RANK, SIZE, O>
{
    #[inline]
    fn forward<'a, const IN_SIZE: usize, const W_SIZE: usize, const OUT_SIZE: usize>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, IN_SIZE>>,
    ) -> (
        Tensor<RANK, OUT_SIZE>,
        Backward<'a, RANK, OUT_SIZE, SIZE, IN_SIZE>,
    ) {
        const {
            assert!(IN_SIZE == OUT_SIZE, "data length mismatch in forward");
        }

        let (out, back) = self.__forward::<IN_SIZE, OUT_SIZE, W_SIZE>(input);
        (out, back)
    }

    #[inline]
    fn backward<const IN_SIZE: usize, const OUT_SIZE: usize>(
        &self,
        grad_out: Tensor<RANK, OUT_SIZE>,
        back: Backward<'_, RANK, OUT_SIZE, SIZE, IN_SIZE>,
    ) -> (Tensor<RANK, IN_SIZE>, [Tensor<RANK, SIZE>; 3]) {
        const {
            assert!(IN_SIZE == OUT_SIZE, "data length mismatch in backward");
        }

        self.__backward(grad_out, back)
    }
}
