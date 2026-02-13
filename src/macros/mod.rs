//! Provides the necessary means of abstraction which make advanced cases much simpler.

use crate::nn::{
    ops::dispatch::{gelu, matmul, relu, sigmoid, swish, tanh},
    tensors::{Flatten, IntoWithGrad, StaticShape, WithGrad},
    TensorFloat,
};
use box_closure::{
    Align1, Align16, Align2, Align32, Align4, Align8, OpaqueFn, OpaqueFnMut, OpaqueFnOnce,
};

#[cfg(feature = "alloc")]
use alloc::boxed::Box;

#[cfg(feature = "alloc")]
use crate::nn::tensors::FlatBox;

#[cfg(feature = "alloc")]
pub use alloc::vec::Vec;

mod model_def;

mod layers;
pub use self::layers::{
    Activation, ActivationLayer, Bias, BiasLayer, BuildLayer, CausalSelfAttention,
    CausalSelfAttentionLayer, Collapse, CollapseLayer, Conv, ConvLayer, Dense, DenseLayer,
    FeedForward, FeedForwardLayer, Layer, LayerNorm, LayerNormLayer, LayerOpHeavy, Residual,
    ResidualLayer, SelfAttention, SelfAttentionLayer, Sigmoid, SigmoidLayer, Softmax, SoftmaxLayer,
};

pub mod optim;

pub mod test;

/// Produces an asymmetrical distribution across an array.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn asym_distr<const N: usize>(variance: f32, offset: usize) -> Box<[TensorFloat; N]> {
    let mut arr = Box::<[TensorFloat; N]>::new_uninit();
    for i in 0..N {
        let val = (libm::sinf((i as f32) * variance + (offset as f32)) * variance) as TensorFloat;
        unsafe {
            arr.as_mut_ptr().cast::<TensorFloat>().add(i).write(val);
        }
    }
    unsafe { arr.assume_init() }
}

/// Adapts the learning rate accordingly to the loss.
#[must_use]
pub fn adapt_lr(lr: (f32, f32, f32), loss: (f32, f32), mut panic: bool) -> (f32, f32, f32, bool) {
    let (mut lr, mut min_lr, mut max_lr) = lr;
    let (prev, loss) = loss;
    if prev > loss {
        // panic because of greater loss
        panic = true;
    } else if panic {
        panic = false;
    } else {
        // optimize lr to bring forth lower loss
        if ((loss - prev).abs() + crate::approx::F32_MIN_ERROR) < lr {
            if lr < min_lr {
                lr *= 2.0;
            } else {
                lr /= 1.03; // small reduction to ensure learning rate is reasonable
            }
        } else if lr < max_lr {
            lr *= 1.01;
            if lr > min_lr {
                lr *= 1.01; // excessive increment to balance out low lr
            } else if min_lr < max_lr {
                max_lr /= 1.01;
            } else {
                min_lr /= 1.03;
            }
        } else {
            max_lr *= 2.0;
            min_lr /= 2.0;
            lr *= 1.01;
        }
    }
    (lr, min_lr, max_lr, panic)
}

/// Decreases the learning rate accordingly to the change in loss.
#[must_use]
pub fn decay_lr(lr: (f32, f32, f32), loss: (f32, f32), mut panic: bool) -> (f32, f32, f32, bool) {
    let (mut lr, min_lr, max_lr) = lr;
    let (prev, loss) = loss;
    let dif_loss = prev - loss;
    if dif_loss >= 0.0 {
        panic = true;
        lr /= libm::powf(1.01, dif_loss);
    } else if dif_loss > -0.001 {
        lr *= 1.0 - libm::powf(2.0, -1.0 / dif_loss);
    } else {
        lr *= 0.993;
    }
    (lr, min_lr, max_lr, panic)
}

/// Linearly interpolates the temperature at a variable rate.
#[must_use]
pub fn decay_temp(temp: f32, scale: f32) -> f32 {
    temp + (1.0 - temp) * scale
}

/// A backwards closure that conditionally uses allocation if available.
#[cfg(feature = "alloc")]
pub type BackFn<'a, In, Out> = Box<dyn FnOnce(In) -> Out + 'a>;

/// Generic tensor using `const` generics regardless of whether they are needed.
#[cfg(not(feature = "dyntensor"))]
pub type Tensor<const D: usize, const N: usize> = crate::nn::tensors::Tensor<TensorFloat, N, D>;
/// Generic tensor using `const` generics regardless of whether they are needed.
#[cfg(feature = "dyntensor")]
pub type Tensor<const D: usize, const N: usize> = crate::nn::tensors::Tensor<TensorFloat>;

/// A backwards closure representing either unary or binary output types.
pub enum Backward<'a, const R: usize, const S0: usize, const S1: usize, const S2: usize> {
    /// Represents an unary function `fn(Tensor) -> Tensor`.
    Unary(BackFn<'a, Tensor<R, S1>, Tensor<R, S2>>),

    /// Represents a binary function `fn(Tensor) -> (Tensor, Tensor)`.
    Binary(BackFn<'a, Tensor<R, S0>, (Tensor<R, S1>, Tensor<R, S2>)>),

    /// Represents a ternary function `fn(Tensor) -> (Tensor, Tensor, Tensor)`.
    Ternary(BackFn<'a, Tensor<R, S0>, (Tensor<R, S1>, Tensor<R, S1>, Tensor<R, S2>)>),

    /// Represents a quaternary function `fn(Tensor) -> (Tensor, Tensor, Tensor, Tensor)`.
    Quaternary(
        BackFn<'a, Tensor<R, S0>, (Tensor<R, S1>, Tensor<R, S1>, Tensor<R, S1>, Tensor<R, S2>)>,
    ),
}

/// A trait for types that are marked as activation functions.
pub trait ActivationFn {
    /// Enumerates type of activation function `Self` represents.
    fn kind() -> ActivationKind;
}

/// An enumeration of activation functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationKind {
    /// Dispatches to a `ReLU` activation function.
    ReLU,
    /// Dispatches to a Tanh activation function.
    Tanh,
    /// Dispatches to a Swish (`SiLU`) activation function.
    Swish,
    /// Dispatches to a GELU activation function.
    GELU,
}

/// Type that dispatches to a `ReLU` acitvation.
pub struct ReLU;

impl ActivationFn for ReLU {
    fn kind() -> ActivationKind {
        ActivationKind::ReLU
    }
}

/// Type that dispatches to a Tanh acitvation.
pub struct Tanh;

impl ActivationFn for Tanh {
    fn kind() -> ActivationKind {
        ActivationKind::Tanh
    }
}

/// Type that dispatches to a GELU acitvation.
pub struct GELU;

impl ActivationFn for GELU {
    fn kind() -> ActivationKind {
        ActivationKind::GELU
    }
}

/// Type that dispatches to a Swish (`SiLU`) acitvation.
pub struct Swish;

impl ActivationFn for Swish {
    fn kind() -> ActivationKind {
        ActivationKind::Swish
    }
}

/// A dataset of 3 tensors for input, gradient, and target.
///
/// Size: `(2 * IR + OR) * size_of::<usize>() + (2 * IS + OS) * size_of::<f32>()`
#[derive(Debug, Clone, PartialEq)]
pub struct Dataset<
    const IN_RANK: usize,
    const IN_SIZE: usize,
    const OUT_RANK: usize,
    const OUT_SIZE: usize,
> {
    inputs: WithGrad<Tensor<IN_RANK, IN_SIZE>>,
    targets: Tensor<OUT_RANK, OUT_SIZE>,
}

impl<const IR: usize, const IS: usize, const OR: usize, const OS: usize> Dataset<IR, IS, OR, OS> {
    /// Construct a new dataset by flattenning nested arrays.
    pub fn new<
        F1: Flatten<IS, Flattened = [f32; IS]> + StaticShape<IR>,
        F2: Flatten<OS, Flattened = [f32; OS]> + StaticShape<OR>,
    >(
        inputs: F1,
        targets: F2,
    ) -> Self {
        let input_shape = *inputs.sliced_shape();
        let input_data = inputs.flatten();
        let target_shape = *targets.sliced_shape();
        let target_data = targets.flatten();
        Self::from_parts(&input_shape, &input_data, &target_shape, &target_data)
    }

    /// Copy into a new dataset by flattenning nested arrays.
    pub fn new_copy<
        F1: FlatBox<IS, Flattened = [f32; IS]> + StaticShape<IR>,
        F2: FlatBox<OS, Flattened = [f32; OS]> + StaticShape<OR>,
    >(
        inputs: &F1,
        targets: &F2,
    ) -> Self {
        let input_shape = *inputs.sliced_shape();
        let input_data = inputs.flat_box();
        let target_shape = *targets.sliced_shape();
        let target_data = targets.flat_box();
        Self::from_parts(&input_shape, &input_data, &target_shape, &target_data)
    }

    /// Construct a new dataset by flattenning nested arrays.
    pub fn arr_of<
        F1: Flatten<IS, Flattened = [f32; IS]> + StaticShape<IR>,
        F2: Flatten<OS, Flattened = [f32; OS]> + StaticShape<OR>,
        const N: usize,
    >(
        inputs: [F1; N],
        targets: [F2; N],
    ) -> [Self; N] {
        let mut datasets_maybe = core::mem::MaybeUninit::<[Self; N]>::uninit();
        for (i, (input, target)) in inputs.into_iter().zip(targets).enumerate() {
            let dataset = Self::new(input, target);
            unsafe {
                datasets_maybe
                    .as_mut_ptr()
                    .cast::<Self>()
                    .add(i)
                    .write(dataset);
            }
        }
        unsafe { datasets_maybe.assume_init() }
    }

    /// Construct a new dataset by flattenning nested arrays.
    pub fn arr_copy<
        F1: FlatBox<IS, Flattened = [f32; IS]> + StaticShape<IR>,
        F2: FlatBox<OS, Flattened = [f32; OS]> + StaticShape<OR>,
        const N: usize,
    >(
        inputs: &[F1; N],
        targets: &[F2; N],
    ) -> [Self; N] {
        let mut datasets_maybe = core::mem::MaybeUninit::<[Self; N]>::uninit();
        for (i, (input, target)) in inputs.iter().zip(targets).enumerate() {
            let dataset = Self::new_copy(input, target);
            unsafe {
                datasets_maybe
                    .as_mut_ptr()
                    .cast::<Self>()
                    .add(i)
                    .write(dataset);
            }
        }
        unsafe { datasets_maybe.assume_init() }
    }

    /// Construct a dataset from raw numeric data and shapes.
    #[must_use]
    #[allow(clippy::unnecessary_cast)]
    pub fn from_parts(
        inputs_shape: &[usize; IR],
        inputs_data: &[f32; IS],
        targets_shape: &[usize; OR],
        targets_data: &[f32; OS],
    ) -> Self {
        let inputs_tensor = Tensor::<IR, IS>::new(inputs_shape, inputs_data);
        let targets_tensor = Tensor::<OR, OS>::new(targets_shape, targets_data);

        Self {
            inputs: inputs_tensor.with_grad(),
            targets: targets_tensor,
        }
    }

    /// Gets the inputs of the dataset.
    #[must_use]
    pub const fn inputs(&self) -> &WithGrad<Tensor<IR, IS>> {
        &self.inputs
    }

    /// Gets the targets of the dataset.
    #[must_use]
    pub const fn targets(&self) -> &Tensor<OR, OS> {
        &self.targets
    }

    /// Mutably obtains the targets of the dataset.
    #[must_use]
    pub const fn targets_mut(&mut self) -> &mut Tensor<OR, OS> {
        &mut self.targets
    }

    /// Mutably obtains the inputs of the dataset.
    pub const fn inputs_mut(&mut self) -> &mut WithGrad<Tensor<IR, IS>> {
        &mut self.inputs
    }

    /// Converts the dataset to the context of its inputs.
    #[must_use]
    pub const fn get_context(&self) -> &Context<IR, IS> {
        unsafe { &*(&raw const self.inputs).cast::<Context<IR, IS>>() }
    }

    /// Converts the dataset to the context of its inputs.
    #[must_use]
    pub fn clone_context(&self) -> Context<IR, IS> {
        Context {
            inputs: self.inputs.clone(),
        }
    }

    /// Converts the dataset to the context of its inputs.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn into_context(self) -> Context<IR, IS> {
        Context {
            inputs: self.inputs,
        }
    }
}

/// A dataset of 2 tensors for inputs and gradient (no target).
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq)]
pub struct Context<const IN_RANK: usize, const IN_SIZE: usize> {
    inputs: WithGrad<Tensor<IN_RANK, IN_SIZE>>,
}

impl<const IR: usize, const IS: usize> Context<IR, IS> {
    /// Construct a new dataset by flattenning nested arrays.
    pub fn new<F1: Flatten<IS, Flattened = [f32; IS]> + StaticShape<IR>>(inputs: F1) -> Self {
        Self::from_parts(*inputs.sliced_shape(), inputs.flatten())
    }

    /// Construct a dataset from raw numeric data and shapes.
    #[must_use]
    #[allow(clippy::unnecessary_cast)]
    pub fn from_parts(inputs_shape: [usize; IR], inputs_data: [f32; IS]) -> Self {
        let inputs_tensor = Tensor::<IR, IS>::new(&inputs_shape, &inputs_data);

        Self {
            inputs: inputs_tensor.with_grad(),
        }
    }

    /// Gets the inputs of the dataset.
    #[must_use]
    pub const fn raw(&self) -> &WithGrad<Tensor<IR, IS>> {
        &self.inputs
    }

    /// Mutably obtains the inputs of the dataset.
    pub const fn raw_mut(&mut self) -> &mut WithGrad<Tensor<IR, IS>> {
        &mut self.inputs
    }
}

/// An abstraction over closures to call them regardless of type.
pub trait Closure<Arg, Ret> {
    /// Invoke the closure.
    fn invoke(&self, args: Arg) -> Ret;
}

/// An abstraction over self-mutating closures.
pub trait ClosureMut<Arg, Ret> {
    /// Invoke the closure.
    fn invoke_mut(&mut self, args: Arg) -> Ret;
}

/// An abstraction over self-destroying closures.
pub trait ClosureOnce<Arg, Ret> {
    /// Invoke the closure.
    fn invoke_once(self, args: Arg) -> Ret;
}

macro_rules! __generate_opaque_closure_impl {
    ($struct:ident, $buf:ident) => {
        impl<'a, A, R, const B: usize> Closure<A, R> for $struct<'a, A, R, $buf<B>> {
            fn invoke(&self, args: A) -> R {
                self.call(args)
            }
        }
    };
}

macro_rules! __generate_opaque_mut_impl {
    ($struct:ident, $buf:ident) => {
        impl<'a, A, R, const B: usize> ClosureMut<A, R> for $struct<'a, A, R, $buf<B>> {
            fn invoke_mut(&mut self, args: A) -> R {
                self.call(args)
            }
        }
    };
}

macro_rules! __generate_opaque_once_impl {
    ($struct:ident, $buf:ident) => {
        impl<'a, A, R, const B: usize> ClosureOnce<A, R> for $struct<'a, A, R, $buf<B>> {
            fn invoke_once(self, args: A) -> R {
                self.call(args)
            }
        }
    };
}

__generate_opaque_closure_impl!(OpaqueFn, Align1);
__generate_opaque_closure_impl!(OpaqueFn, Align2);
__generate_opaque_closure_impl!(OpaqueFn, Align4);
__generate_opaque_closure_impl!(OpaqueFn, Align8);
__generate_opaque_closure_impl!(OpaqueFn, Align16);
__generate_opaque_closure_impl!(OpaqueFn, Align32);
__generate_opaque_mut_impl!(OpaqueFnMut, Align1);
__generate_opaque_mut_impl!(OpaqueFnMut, Align2);
__generate_opaque_mut_impl!(OpaqueFnMut, Align4);
__generate_opaque_mut_impl!(OpaqueFnMut, Align8);
__generate_opaque_mut_impl!(OpaqueFnMut, Align16);
__generate_opaque_mut_impl!(OpaqueFnMut, Align32);
__generate_opaque_once_impl!(OpaqueFnOnce, Align1);
__generate_opaque_once_impl!(OpaqueFnOnce, Align2);
__generate_opaque_once_impl!(OpaqueFnOnce, Align4);
__generate_opaque_once_impl!(OpaqueFnOnce, Align8);
__generate_opaque_once_impl!(OpaqueFnOnce, Align16);
__generate_opaque_once_impl!(OpaqueFnOnce, Align32);

impl<A, R, F: ?Sized + Fn(A) -> R> Closure<A, R> for F {
    fn invoke(&self, args: A) -> R {
        self(args)
    }
}

impl<A, R, F: ?Sized + FnMut(A) -> R> ClosureMut<A, R> for F {
    fn invoke_mut(&mut self, args: A) -> R {
        self(args)
    }
}

impl<A, R, F: FnOnce(A) -> R> ClosureOnce<A, R> for F {
    fn invoke_once(self, args: A) -> R {
        self(args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "alloc")]
    fn new_copy_safe() {
        let inputs = [[5.0; 10]; 2];
        let targets = [[5.0; 2]; 10];
        let dataset: Dataset<2, 20, 2, 20> = Dataset::new_copy(&inputs, &targets);
        let datasets: [Dataset<2, 20, 2, 20>; 8] = Dataset::arr_copy(&[inputs; 8], &[targets; 8]);
        for d in datasets {
            assert_eq!(dataset, d);
        }
    }

    #[test]
    fn get_context_safety() {
        let dataset: Dataset<2, 20, 1, 20> = Dataset::new([[5.0; 10]; 2], [5.0; 20]);
        let context;
        {
            context = dataset.get_context();
        }
        assert_eq!(context.raw(), dataset.inputs());
    }
}
