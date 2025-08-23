use crate::abstracted::internal::Activation;
use crate::abstracted::internal::Closure;
use crate::abstracted::layers::__LayoutMarker;
use crate::abstracted::layers::BackFn;
use crate::abstracted::layers::Layer;
use crate::manual::TensorFloat;
use crate::manual::backprop::matmul;
use crate::manual::tensors::{IntoWithGrad, Tensor, WithGrad};
use tensor_optim::TensorOps;

#[cfg(feature = "alloc")]
use alloc::boxed::Box;
#[cfg(not(feature = "alloc"))]
use box_closure::OpaqueFnOnce;

pub struct Dense<const IN: usize, const OUT: usize, Act: Activation> {
    _phantom: core::marker::PhantomData<Act>,
}

impl<const IN: usize, const OUT: usize, Act: Activation> __LayoutMarker for Dense<IN, OUT, Act> {}

pub struct DenseImpl<const IN: usize, const OUT: usize, const WN: usize, Act: Activation> {
    #[cfg(feature = "dyntensor")]
    weights: WithGrad<Tensor<TensorFloat>>, // shape [IN, OUT]
    #[cfg(not(feature = "dyntensor"))]
    weights: WithGrad<Tensor<TensorFloat, WN, 2>>,
    #[cfg(feature = "dyntensor")]
    bias: WithGrad<Tensor<TensorFloat>>, // shape [OUT, 1] or [1, OUT]
    #[cfg(not(feature = "dyntensor"))]
    bias: WithGrad<Tensor<TensorFloat, OUT, 2>>,
    #[cfg(feature = "dyntensor")]
    scratch: WithGrad<Tensor<TensorFloat>>,
    #[cfg(not(feature = "dyntensor"))]
    scratch: WithGrad<Tensor<TensorFloat, OUT, 2>>,
    activation: Act,
}

#[cfg(not(feature = "dyntensor"))]
impl<const IN: usize, const OUT: usize, const WN: usize, Act>
    Layer<Tensor<TensorFloat, IN, 2>, Tensor<TensorFloat, OUT, 2>> for DenseImpl<IN, OUT, WN, Act>
where
    Act: crate::abstracted::internal::Activation<Tensor = Tensor<TensorFloat, OUT, 2>>,
{
    type ParamA = Tensor<TensorFloat, WN, 2>;
    type ParamB = Tensor<TensorFloat, OUT, 2>;

    fn run<'a>(
        &'a mut self,
        input: &'a WithGrad<Tensor<TensorFloat, IN, 2>>,
    ) -> (
        Tensor<TensorFloat, OUT, 2>,
        BackFn<'a, Tensor<TensorFloat, IN, 2>, Tensor<TensorFloat, OUT, 2>>,
    ) {
        let (wg_ptr, wg_len, bg_ptr, bg_len) = {
            // weights.grad
            let wg = self.weights.get_grad_mut();
            let ws = wg.data_mut();
            let wg_ptr = ws.as_mut_ptr();
            let wg_len = ws.len();
            // bias.grad
            let bg = self.bias.get_grad_mut();
            let bs = bg.data_mut();
            let bg_ptr = bs.as_mut_ptr();
            let bg_len = bs.len();
            (wg_ptr, wg_len, bg_ptr, bg_len)
        };

        let (mut z, back_mat) = matmul(input, &self.weights);

        // add bias along the output dimension
        let batch = z.shape()[0];
        let bias_values = self.bias.get_value().data();
        for i in 0..batch {
            for (j, val) in bias_values.iter().enumerate().take(OUT) {
                z.data_mut()[i * OUT + j] += val;
            }
        }

        self.scratch = z.with_grad();

        let (a, back_act) = self.activation.apply(&self.scratch);

        // build composite back-prop closure that:
        //  - given dout: runs back_act -> gives dz
        //  - runs back_mat(dz) -> gives (dinput, dweights)
        //  - accumulates dweights into self.weights.grad
        //  - computes bias grad and accumulates into self.bias.grad
        //  - returns dinput

        #[cfg(feature = "alloc")]
        let back = {
            use crate::abstracted::layers::BackFn;

            Box::new(
                move |dout: Tensor<TensorFloat, OUT, 2>| -> Tensor<TensorFloat, IN, 2> {
                    // dL/dz
                    let dz = back_act.invoke(dout);

                    // --- compute dbias from dz BEFORE moving dz anywhere ---
                    // shape assumptions: dz is [batch, OUT] laid out row-major
                    // no alloc: use fixed-size array on stack for accumulation
                    let mut dbias_acc = [0.0 as TensorFloat; OUT];
                    {
                        let dz_data = dz.data(); // borrow ends at block end
                        debug_assert!(OUT > 0 && dz_data.len() % OUT == 0);
                        let batch = dz_data.len() / OUT;
                        // tight loops; no modulus in inner loop
                        let mut idx = 0usize;
                        for _ in 0..batch {
                            for val in dbias_acc.iter_mut().take(OUT) {
                                *val += dz_data[idx];
                                idx += 1;
                            }
                        }
                    } // dz_data borrow ends here

                    // move dz into back_mat, no borrow alive anymore.
                    let (dinput, dweights) = back_mat.invoke(dz);

                    // accumulate dweights into weights.grad via raw pointer
                    let src = dweights.data();
                    debug_assert_eq!(src.len(), wg_len, "weight grad len mismatch");
                    for (i, val) in src.iter().enumerate() {
                        #[allow(unsafe_code)]
                        let p = unsafe { wg_ptr.add(i) };
                        #[allow(unsafe_code)]
                        unsafe {
                            *p += val;
                        }
                    }

                    // accumulate dbias into bias.grad via raw pointer
                    debug_assert_eq!(bg_len, OUT, "bias length != OUT");
                    for (j, val) in dbias_acc.iter().enumerate().take(OUT) {
                        #[allow(unsafe_code)]
                        let p = unsafe { bg_ptr.add(j) };
                        #[allow(unsafe_code)]
                        unsafe {
                            *p += val;
                        }
                    }

                    dinput
                },
            ) as BackFn<'a, _, _>
        };

        #[cfg(not(feature = "alloc"))]
        let back = {
            OpaqueFnOnce::new(
                move |dout: Tensor<TensorFloat, OUT, 2>| -> Tensor<TensorFloat, IN, 2> {
                    // dL/dz
                    let dz = back_act.invoke(dout);

                    // --- compute dbias from dz BEFORE moving dz anywhere ---
                    // shape assumptions: dz is [batch, OUT] laid out row-major
                    // no alloc: use fixed-size array on stack for accumulation
                    let mut dbias_acc = [0.0 as TensorFloat; OUT];
                    {
                        let dz_data = dz.data(); // borrow ends at block end
                        debug_assert!(OUT > 0 && dz_data.len() % OUT == 0);
                        let batch = dz_data.len() / OUT;
                        // tight loops; no modulus in inner loop
                        let mut idx = 0usize;
                        for _ in 0..batch {
                            for val in dbias_acc.iter_mut().take(OUT) {
                                *val += dz_data[idx];
                                idx += 1;
                            }
                        }
                    } // dz_data borrow ends here

                    // move dz into back_mat, no borrow alive anymore.
                    let (dinput, dweights) = back_mat.invoke(dz);

                    // accumulate dweights into weights.grad via raw pointer
                    let src = dweights.data();
                    debug_assert_eq!(src.len(), wg_len, "weight grad len mismatch");
                    for (i, val) in src.iter().enumerate() {
                        #[allow(unsafe_code)]
                        let p = unsafe { wg_ptr.add(i) };
                        #[allow(unsafe_code)]
                        unsafe {
                            *p += val;
                        }
                    }

                    // accumulate dbias into bias.grad via raw pointer
                    debug_assert_eq!(bg_len, OUT, "bias length != OUT");
                    for (j, val) in dbias_acc.iter().enumerate().take(OUT) {
                        #[allow(unsafe_code)]
                        let p = unsafe { bg_ptr.add(j) };
                        #[allow(unsafe_code)]
                        unsafe {
                            *p += val;
                        }
                    }

                    dinput
                },
            ) as BackFn<'a, _, _>
        };

        (a, back)
    }

    fn params(&mut self) -> (&mut WithGrad<Self::ParamA>, &mut WithGrad<Self::ParamB>) {
        (&mut self.weights, &mut self.bias)
    }
}

#[cfg(feature = "dyntensor")]
impl<const IN: usize, const OUT: usize, const WN: usize, Act>
    Layer<Tensor<TensorFloat>, Tensor<TensorFloat>> for DenseImpl<IN, OUT, WN, Act>
where
    Act: crate::abstracted::internal::Activation<Tensor = Tensor<TensorFloat>>,
{
    type ParamA = Tensor<TensorFloat>;
    type ParamB = Tensor<TensorFloat>;

    fn run<'a>(
        &'a mut self,
        input: &'a WithGrad<Tensor<TensorFloat>>,
    ) -> (
        Tensor<TensorFloat>,
        BackFn<'a, Tensor<TensorFloat>, Tensor<TensorFloat>>,
    ) {
        let (wg_ptr, wg_len, bg_ptr, bg_len) = {
            // weights grad
            let wg = self.weights.get_grad_mut();
            let wg_slice = wg.data_mut();
            let wg_ptr = wg_slice.as_mut_ptr();
            let wg_len = wg_slice.len();

            // bias grad
            let bg = self.bias.get_grad_mut();
            let bg_slice = bg.data_mut();
            let bg_ptr = bg_slice.as_mut_ptr();
            let bg_len = bg_slice.len();

            (wg_ptr, wg_len, bg_ptr, bg_len)
        };
        // mutable borrows ended here

        let (mut z_lin, back_mat) = matmul(input, &self.weights);

        // add bias along the output dimension
        let batch = z_lin.shape()[0];
        let bias_values = self.bias.get_value().data();
        for i in 0..batch {
            for (j, val) in bias_values.iter().enumerate().take(OUT) {
                z_lin.data_mut()[i * OUT + j] += val;
            }
        }

        let z = z_lin; // if you fuse bias in activation, leave as-is
        self.scratch = z.with_grad();

        let (a, back_act) = self.activation.apply(&self.scratch);

        let back = Box::new(move |dout: Tensor<TensorFloat>| -> Tensor<TensorFloat> {
            use crate::abstracted::layers::reduce_rows_sum_to_bias;

            let dz = back_act.invoke(dout);

            // back through matmul -> (dinput, dW)
            let (dinput, dweights) = back_mat.invoke(dz.clone());

            // accumulate dW into weights.grad via raw pointer
            let src = dweights.data();
            debug_assert_eq!(src.len(), wg_len, "dW len mismatch");
            for (i, val) in src.iter().enumerate() {
                // *wg_ptr.add(i) += src[i];
                #[allow(unsafe_code)]
                let p = unsafe { wg_ptr.add(i) };
                #[allow(unsafe_code)]
                unsafe {
                    *p += val;
                }
            }

            // accumulate dbias by reducing dz along batch axis into bias shape
            // implement reduce_rows_sum() or whatever matches your shape rules
            let dbias = reduce_rows_sum_to_bias(&dz, bg_len);

            let src = dbias.data();
            debug_assert_eq!(src.len(), bg_len, "dbias len mismatch");
            for (i, val) in src.iter().enumerate() {
                #[allow(unsafe_code)]
                let p = unsafe { bg_ptr.add(i) };
                #[allow(unsafe_code)]
                unsafe {
                    *p += val;
                }
            }

            dinput
        }) as BackFn<'_, _, _>;

        (a, back)
    }

    fn params(&mut self) -> (&mut WithGrad<Self::ParamA>, &mut WithGrad<Self::ParamB>) {
        (&mut self.weights, &mut self.bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstracted::internal::ClosureOnce;
    use crate::abstracted::internal::Relu;
    use crate::manual::tensors::{IntoWithGrad, Tensor};

    #[test]
    fn dense_forward_and_backward() {
        const IN: usize = 2;
        const OUT: usize = 2;
        const WN: usize = IN * OUT;

        // construct dense layer with trivial activation
        let mut dense: DenseImpl<IN, OUT, WN, Relu<2, 2>> = DenseImpl {
            weights: Tensor::new(&[IN, OUT], &[1.0, 2.0, 3.0, 4.0]).with_grad(),
            bias: Tensor::new(&[1, OUT], &[0.5, -0.5]).with_grad(),
            scratch: Tensor::zeros(&[1, OUT]).with_grad(),
            activation: Relu,
        };

        // input: shape [1, IN]
        let input = Tensor::new(&[1, IN], &[1.0, 2.0]).with_grad();

        // run forward
        let (out, back) = dense.run(&input);

        // expected forward = input @ weights + bias
        // [1,2] @ [[1,2],[3,4]] + [0.5,-0.5] = [7.5, 9.5]
        let data = out.data();
        assert_eq!(data.len(), OUT);
        assert!((data[0] - 7.5).abs() < 1e-6);
        assert!((data[1] - 9.5).abs() < 1e-6);

        // run backward with dout = [1, OUT]
        let dout = Tensor::new(&[1, OUT], &[1.0, 1.0]);
        let dinput = back.invoke(dout);

        // check gradient accumulation happened
        let wg = dense.weights.get_grad();
        let bg = dense.bias.get_grad();
        let wgrad = wg.data();
        let bgrad = bg.data();

        assert!(
            wgrad.iter().any(|&g| g != 0.0),
            "weights.grad should have nonzero values"
        );
        assert!(
            bgrad.iter().any(|&g| g != 0.0),
            "bias.grad should have nonzero values"
        );

        // check dinput shape and values
        let di = dinput.data();
        assert_eq!(di.len(), IN);
    }
}
