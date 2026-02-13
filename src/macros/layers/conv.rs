#![cfg_attr(feature = "dyntensor", allow(clippy::useless_conversion))]

use super::{Backward, ClosureOnce, IntoWithGrad, Layer, Tensor, TensorFloat, WithGrad};
use crate::macros::{optim::Optim, BuildLayer, LayerOpHeavy};
use alloc::boxed::Box;
use core::marker::PhantomData;
use tensor_optim::TensorOps;

/// A representation of a kernel layer builder.
pub struct Conv<const RANK: usize, const SIZE: usize, O> {
    /// Shape of the kernel tensor.
    pub shape: [usize; RANK],

    /// Convolution data.
    pub data: Box<[TensorFloat; SIZE]>,

    /// Optimizer for weights.
    pub _optim: PhantomData<O>,
}

impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> Conv<RANK, SIZE, O> {
    /// The amount of tensors in the layer.
    pub const TENSORS: usize = 1;

    /// Expands the layer.
    #[must_use]
    #[allow(clippy::explicit_auto_deref)]
    pub fn build(self) -> ConvLayer<RANK, SIZE, O> {
        ConvLayer {
            kernel: Tensor::new(&self.shape, &*self.data).with_grad(),
            optim: O::new(&self.shape),
        }
    }
}

impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> BuildLayer<RANK, SIZE, 1>
    for Conv<RANK, SIZE, O>
{
    type Layer = ConvLayer<RANK, SIZE, O>;

    fn build(self) -> ConvLayer<RANK, SIZE, O> {
        self.build()
    }
}

/// Expanded kernel layer.
pub struct ConvLayer<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> {
    kernel: WithGrad<Tensor<RANK, SIZE>>,
    optim: O,
}

#[inline]
fn flatten(idx: &[usize], shape: &[usize]) -> usize {
    let mut stride = 1;
    let mut out = 0;

    for d in (0..idx.len()).rev() {
        out += idx[d] * stride;
        stride *= shape[d];
    }
    out
}

#[inline]
fn next_index(idx: &mut [usize], limits: &[usize]) -> bool {
    for d in (0..idx.len()).rev() {
        idx[d] += 1;
        if idx[d] < limits[d] {
            return true;
        }
        idx[d] = 0;
    }
    false
}

impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> ConvLayer<RANK, SIZE, O> {
    #[inline]
    fn __forward<'a, const IN_SIZE: usize, const OUT_SIZE: usize>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, IN_SIZE>>,
    ) -> (
        Tensor<RANK, OUT_SIZE>,
        Backward<'a, RANK, OUT_SIZE, SIZE, IN_SIZE>,
        [(); 1],
    ) {
        let in_shape = input.get_value().shape();
        let k_shape = self.kernel.get_value().shape();

        let mut out_shape = [0usize; RANK];
        for d in 0..RANK {
            out_shape[d] = in_shape[d] - k_shape[d] + 1;
        }

        let mut out = Tensor::<RANK, OUT_SIZE>::zeros(&out_shape);

        let mut out_idx = [0usize; RANK];
        loop {
            let mut acc = 0.0;

            let mut k_idx = [0usize; RANK];
            loop {
                let mut in_idx = [0usize; RANK];
                for d in 0..RANK {
                    in_idx[d] = out_idx[d] + k_idx[d];
                }

                acc += input.get_value().data()[flatten(&in_idx, in_shape)]
                    * self.kernel.get_value().data()[flatten(&k_idx, k_shape)];

                if !next_index(&mut k_idx, k_shape) {
                    break;
                }
            }

            out.data_mut()[flatten(&out_idx, &out_shape)] = acc;

            if !next_index(&mut out_idx, &out_shape) {
                break;
            }
        }

        let back = move |grad_out: Tensor<RANK, OUT_SIZE>| {
            let mut grad_x = Tensor::<RANK, IN_SIZE>::zeros(in_shape.try_into().unwrap());
            let mut grad_k = Tensor::<RANK, SIZE>::zeros(k_shape.try_into().unwrap());

            let mut out_idx = [0usize; RANK];
            loop {
                let go = grad_out.data()[flatten(&out_idx, &out_shape)];

                let mut k_idx = [0usize; RANK];
                loop {
                    let mut in_idx = [0usize; RANK];
                    for d in 0..RANK {
                        in_idx[d] = out_idx[d] + k_idx[d];
                    }

                    grad_x.data_mut()[flatten(&in_idx, in_shape)] +=
                        go * self.kernel.get_value().data()[flatten(&k_idx, k_shape)];

                    grad_k.data_mut()[flatten(&k_idx, k_shape)] +=
                        go * input.get_value().data()[flatten(&in_idx, in_shape)];

                    if !next_index(&mut k_idx, k_shape) {
                        break;
                    }
                }

                if !next_index(&mut out_idx, &out_shape) {
                    break;
                }
            }

            (grad_k, grad_x)
        };

        #[cfg(feature = "alloc")]
        {
            (out, Backward::Binary(alloc::boxed::Box::new(back)), [(); 1])
        }
        #[cfg(not(feature = "alloc"))]
        {
            (
                out,
                Backward::Binary(box_closure::OpaqueFnOnce::new(back)),
                [(); 1],
            )
        }
    }

    /// Forwards the convolution layer.
    #[inline]
    #[cfg(feature = "dyntensor")]
    pub fn forward<'a>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, 0>>,
    ) -> (Tensor<RANK, 0>, Backward<'a, RANK, 0, SIZE, 0>, [(); 1]) {
        self.__forward(input)
    }

    /// Forwards the convolution layer.
    #[inline]
    #[cfg(not(feature = "dyntensor"))]
    pub fn forward<'a, const IN_SIZE: usize, const OUT_SIZE: usize>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, IN_SIZE>>,
    ) -> (
        Tensor<RANK, OUT_SIZE>,
        Backward<'a, RANK, OUT_SIZE, SIZE, IN_SIZE>,
        [(); 1],
    ) {
        self.__forward(input)
    }

    #[inline]
    #[allow(clippy::unused_self)]
    fn __backward<const IN_SIZE: usize, const OUT_SIZE: usize>(
        &self,
        grad_output: Tensor<RANK, OUT_SIZE>,
        back: Backward<'_, RANK, OUT_SIZE, SIZE, IN_SIZE>,
    ) -> (Tensor<RANK, IN_SIZE>, [Tensor<RANK, SIZE>; 1]) {
        match back {
            Backward::Binary(f) => {
                let (grad_b, grad_in) = f.invoke_once(grad_output);
                (grad_in, [grad_b])
            }
            _ => unreachable!("Conv always has a binary closure"),
        }
    }

    /// Differentiates the convolution layer.
    #[inline]
    #[cfg(feature = "dyntensor")]
    pub fn backward(
        &self,
        grad_output: Tensor<RANK, 0>,
        back: Backward<'_, RANK, 0, SIZE, 0>,
    ) -> (Tensor<RANK, 0>, [Tensor<RANK, SIZE>; 1]) {
        self.__backward(grad_output, back)
    }

    /// Differentiates the convolution layer.
    #[inline]
    #[cfg(not(feature = "dyntensor"))]
    pub fn backward<const IN_SIZE: usize, const OUT_SIZE: usize>(
        &self,
        grad_output: Tensor<RANK, OUT_SIZE>,
        back: Backward<'_, RANK, OUT_SIZE, SIZE, IN_SIZE>,
    ) -> (Tensor<RANK, IN_SIZE>, [Tensor<RANK, SIZE>; 1]) {
        self.__backward(grad_output, back)
    }
}

impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> Layer<RANK, SIZE, 1>
    for ConvLayer<RANK, SIZE, O>
{
    #[inline]
    fn optim_weights(
        &mut self,
    ) -> (
        &mut impl Optim<RANK, SIZE>,
        [&mut WithGrad<Tensor<RANK, SIZE>>; 1],
    ) {
        (&mut self.optim, [&mut self.kernel])
    }

    #[inline]
    fn weights(&self) -> [&WithGrad<Tensor<RANK, SIZE>>; 1] {
        [&self.kernel]
    }

    #[inline]
    fn weights_mut(&mut self) -> [&mut WithGrad<Tensor<RANK, SIZE>>; 1] {
        [&mut self.kernel]
    }
}

impl<const RANK: usize, const SIZE: usize, O: Optim<RANK, SIZE>> LayerOpHeavy<RANK, SIZE, 1>
    for ConvLayer<RANK, SIZE, O>
{
    #[inline]
    fn backward<const IN_SIZE: usize, const OUT_SIZE: usize>(
        &self,
        grad_out: Tensor<RANK, OUT_SIZE>,
        back: Backward<'_, RANK, OUT_SIZE, SIZE, IN_SIZE>,
    ) -> (Tensor<RANK, IN_SIZE>, [Tensor<RANK, SIZE>; 1]) {
        self.__backward(grad_out, back)
    }

    #[inline]
    fn forward<'a, const IN_SIZE: usize, const W_SIZE: usize, const OUT_SIZE: usize>(
        &'a self,
        input: &'a WithGrad<Tensor<RANK, IN_SIZE>>,
    ) -> (
        Tensor<RANK, OUT_SIZE>,
        Backward<'a, RANK, OUT_SIZE, SIZE, IN_SIZE>,
    ) {
        let (out, back, _) = self.__forward(input);
        (out, back)
    }
}
