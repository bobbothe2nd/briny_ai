//! Provides the necessary means of abstraction which make advanced cases much simpler.

use crate::nn::{
    ops::dispatch::{matmul, relu},
    tensors::{Flatten, IntoWithGrad, StaticShape, WithGrad},
    TensorFloat,
};
use box_closure::{
    Align1, Align16, Align2, Align32, Align4, Align8, OpaqueFn, OpaqueFnMut, OpaqueFnOnce,
};

#[cfg(feature = "alloc")]
use alloc::boxed::Box;

mod layers;
pub use self::layers::{Activation, ActivationLayer, Dense, DenseLayer, Layer};

pub mod test;

/// A backwards closure that conditionally uses allocation if available.
#[cfg(feature = "alloc")]
pub type BackFn<'a, In, Out> = Box<dyn Fn(In) -> Out + 'a>;
/// A backwards closure that conditionally uses allocation if available.
#[cfg(not(feature = "alloc"))]
pub type BackFn<'a, In, Out> = OpaqueFn<'a, In, Out, Align8<64>>;

#[cfg(not(feature = "dyntensor"))]
type Tensor<const D: usize, const N: usize> = crate::nn::tensors::Tensor<TensorFloat, N, D>;
#[cfg(feature = "dyntensor")]
type Tensor<const D: usize, const N: usize> = crate::nn::tensors::Tensor<TensorFloat>;

/// A backwards closure representing either unary or binary output types.
pub enum Backward<'a, const R: usize, const S0: usize, const S1: usize, const S2: usize> {
    /// Represents an unary function `fn(Tensor) -> Tensor`.
    Unary(BackFn<'a, Tensor<R, S2>, Tensor<R, S2>>),

    /// Represents a binary function `fb(Tensor) -> (Tensor, Tensor)`.
    Binary(BackFn<'a, Tensor<R, S0>, (Tensor<R, S1>, Tensor<R, S2>)>),
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
    /// Dispatches to a Sigmoid activation function.
    Sigmoid,
}

/// Type that dispatches to a `ReLU` acitvation.
pub struct ReLU;

impl ActivationFn for ReLU {
    fn kind() -> ActivationKind {
        ActivationKind::ReLU
    }
}

/// Type that dispatches to a Sigmoid acitvation.
pub struct Sigmoid;

impl ActivationFn for Sigmoid {
    fn kind() -> ActivationKind {
        ActivationKind::Sigmoid
    }
}

/// Defines a deep learning model based off the descriptors.
///
/// # Descriptors
///
/// The macro requires the name of the model, as well as a description of each
/// layer with the optional activation function and correct dimensions of it.
///
/// It allows for defining sizes of tensors in addition to the flow of each layer
/// by specifying the operations. For layer-specific operation, see the individual
/// operations. To specify these layers, use the following descriptions:
///
/// ```text
/// InputLayer([..]) - describes the input shape of the dataset
/// {
///   [name] : [layer] ([..], [activation?]) => [expanded layer]
///   ...
/// }
/// OutputLayer([..]) - describes the output shape of the targets
/// ```
///
/// Each `..` represents a placeholder for the shape of that layer. The I/O layers
/// are compiled away after construction, and the specified shapes are assumed to
/// match that of the adjacent hidden layer.
///
/// Each hidden layer (the inner ones) must be described as a short, literal name followed
/// by the detailed description. This descriptor contains the name of the builder and the
/// actual layer type in conjunction with the shape of it and any activation function if
/// the layer includes one.
///
/// # Example
///
/// Correct usage of the macro is included below:
///
/// ```rust
/// briny_ai::static_model!(
///     @loss mse_loss
///     @model XorModel
///     {
///         InputLayer([4, 2]),
///         {
///             conv0: Dense([4, 2]) => DenseLayer,
///             act0: Activation([4, 1], ReLU) => ActivationLayer,
///         },
///         OutputLayer([4, 1]),
///     }
/// );
/// ```
///
/// # Behavior
///
/// - The model will automatically adjust the learning rate to optimize the model dynamically.
/// - The testing/training loops are fully unrolled and inlined for optimal performance
#[macro_export]
macro_rules! static_model {
    (
        @loss $loss:ident
        @model $name:ident
        {
            InputLayer([$($in_RANK:expr),+]),
            {$( $field:ident : $layer:ident ([$($RANK:expr),+] $(, $activation:ident)?) => $full_layer:ident, )*},
            OutputLayer([$($out_RANK:expr),+]),
        }
    ) => {
        pub struct $name {
            $(
                $field: $crate::internal::$full_layer<
                    { <[()]>::len(&[$( { let _ = &$RANK; () } ),+]) },
                    { 1 $(* $RANK)* },
                >,
            )*

            lr: $crate::nn::TensorFloat,
        }

        impl $name {
            const IN_RANK: usize = [$($in_RANK),*].len();
            const IN_SIZE: usize = { 1 $(* $in_RANK)* };

            const OUT_RANK: usize = [$($out_RANK),*].len();
            const OUT_SIZE: usize = { 1 $(* $out_RANK)* };

            const SERIALIZED_TENSOR_CAPACITY_NO_CHECKSUMS: usize = { (0 $(+ (1 $(* $RANK)*))* * ::core::mem::size_of::<f64>()) + (0 $(+ ([$($RANK),*].len()))* * ::core::mem::size_of::<u64>()) };
            const SERIALIZED_TENSOR_CAPACITY_WITH_CHECKSUMS: usize = { Self::SERIALIZED_TENSOR_CAPACITY_NO_CHECKSUMS + ((0 $(+ {let _ = [$($RANK),*]; 1})*) * ::core::mem::size_of::<u32>()) + ::core::mem::size_of::<u32>() };

            pub fn new() -> Self {
                pub struct Builder {
                    $(
                        $field: $crate::internal::$layer<
                            { <[()]>::len(&[$( { let _ = &$RANK; () } ),+]) },
                            { 1 $(* $RANK)* }
                            $(, $crate::internal::$activation)?,
                        >,
                    )*
                }

                impl Builder {
                    pub fn new() -> Self {
                        Self {
                            $(
                                $field: $crate::internal::$layer {
                                    shape: [$($RANK),+],
                                    data: ::core::array::from_fn(|i| (i as $crate::nn::TensorFloat * 1e-2).sin()),
                                    $( _activation: ::core::marker::PhantomData::<$crate::internal::$activation>, )?
                                },
                            )*
                        }
                    }

                    pub fn build(self) -> $name {
                        $name {
                            $(
                                $field: self.$field.build(),
                            )*

                            lr: 0.001,
                        }
                    }
                }

                const {
                    ::core::assert!(<[()]>::len(&[$( { let _ = &$in_RANK; () } ),+]) >= 2, "input layer must be at least 2 dimensions");
                    ::core::assert!(<[()]>::len(&[$( { let _ = &$out_RANK; () } ),+]) >= 2, "output layer must be at least 2 dimensions");
                    $(
                        ::core::assert!(<[()]>::len(&[$( { let _ = &$RANK; () } ),+]) >= 2, "all layers must be at least 2 dimensions");
                    )*
                }

                Builder::new().build()
            }

            #[allow(clippy::unnecessary_casts)]
            pub const fn set_lr(&mut self, lr: f64) {
                self.lr = lr as $crate::nn::TensorFloat;
            }

            pub const fn with_lr(self, lr: f64) -> Self {
                let mut copied = self;
                copied.set_lr(lr);
                copied
            }

            #[allow(clippy::unnecessary_casts)]
            pub const fn get_lr(&self) -> f64 {
                self.lr as f64
            }

            pub fn train_fit_epoch(
                &mut self,
                dataset: &$crate::Dataset<
                    { Self::IN_RANK },
                    { Self::IN_SIZE },
                    { Self::OUT_RANK },
                    { Self::OUT_SIZE },
                >
            ) -> f64 {
                // forward pass (fast)
                let input = dataset.inputs();
                let out = ::core::clone::Clone::clone(input.get_value());

                // shadowing for each hidden layer
                $(
                    let input = $crate::nn::tensors::IntoWithGrad::with_grad(out);
                    let (out, $field) = self.$field.forward(&input);
                )*

                // compute loss
                let output = $crate::nn::tensors::IntoWithGrad::with_grad(out);
                let (loss, back_loss) = $crate::nn::ops::dispatch::$loss(&output, dataset.targets());

                // backward pass (unrolled)
                let grad_output = $crate::internal::Closure::invoke(&back_loss, loss);

                // hidden layers backward (reversed order)
                $crate::static_model!(@rev $($field),+; grad_output self);

                // apply optimizer to non-IO layers
                $(
                    $crate::internal::Layer::apply_update(
                        &mut self.$field,
                        self.lr,
                        $crate::nn::ops::dispatch::sgd
                    );
                )*

                #[allow(clippy::unnecessary_casts)]
                {
                    loss as f64
                }
            }

            pub fn fit(
                &mut self,
                dataset: &$crate::Dataset<
                    { Self::IN_RANK },
                    { Self::IN_SIZE },
                    { Self::OUT_RANK },
                    { Self::OUT_SIZE },
                >,
                epochs: usize,
            ) -> f64 {
                let mut max_lr = 0.2_f64;
                let mut min_lr = 1e-3_f64;
                let mut loss = 0_f64;
                for i in 0..epochs {
                    let prev = loss;
                    loss = self.train_fit_epoch(dataset);

                    // optimize lr to bring forth lower loss
                    if ((loss - prev).abs() + $crate::approx::F64_MIN_ERROR) < (self.lr as f64) {
                        if self.lr < min_lr {
                            self.lr *= 2.0;
                        } else {
                            self.lr /= 1.03;   // small reduction to ensure learning rate is reasonable
                        }
                    } else if self.lr < max_lr {
                        self.lr *= 1.01;
                        if self.lr > min_lr {
                            self.lr *= 1.01;   // excessive increment to balance out low lr
                        } else if min_lr < max_lr {
                            max_lr /= 1.01;
                        } else {
                            min_lr /= 1.03;
                        }
                    } else {
                        max_lr *= 2.0;
                        min_lr /= 2.0;
                        self.lr *= 1.01;
                    }
                }
                loss // return the most recent loss
            }

            pub fn test_epoch(
                &self,
                dataset: &$crate::Dataset<
                    { Self::IN_RANK },
                    { Self::IN_SIZE },
                    { Self::OUT_RANK },
                    { Self::OUT_SIZE },
                >,
            ) -> $crate::internal::TestEval {
                // forward pass (fast)
                let input = dataset.inputs();
                let out = ::core::clone::Clone::clone(input.get_value());

                // shadowing for each hidden layer
                $(
                    let input = $crate::nn::tensors::IntoWithGrad::with_grad(out);
                    let (out, _) = self.$field.forward(&input);
                )*

                // compute loss
                let output = $crate::nn::tensors::IntoWithGrad::with_grad(out);
                let (loss, _) = $crate::nn::ops::dispatch::$loss(&output, dataset.targets());
                let score = $crate::internal::test::percentage_correct(output.get_value(), dataset.targets());
                let accuracy = $crate::internal::test::accuracy_of(output.get_value(), dataset.targets());

                #[allow(clippy::unnecessary_casts)]
                $crate::internal::TestEval {
                    loss: loss as f64,
                    score,
                    accuracy,
                }
            }

            pub fn test(
                &self,
                dataset: &$crate::Dataset<
                    { Self::IN_RANK },
                    { Self::IN_SIZE },
                    { Self::OUT_RANK },
                    { Self::OUT_SIZE },
                >,
                epochs: usize,
            ) -> $crate::internal::TestEval {
                let mut eval = $crate::internal::TestEval {
                    loss: 0.0,
                    score: 0.0,
                    accuracy: 0.0,
                };

                for i in 0..epochs {
                    let update = self.test_epoch(dataset);
                    eval.set_if_better(&update);
                }

                eval
            }

            // pub fn save(
            //     &self,
            //     path: &str,
            // ) -> ::core::result::Result<(), $crate::nn::io::SerialTensorError> {
            //     ::core::result::Result::Ok(())
            // }

            // pub fn load(
            //     &mut self,
            //     path: &str,
            // ) -> ::core::result::Result<(), $crate::nn::io::SerialTensorError> {
            //     ::core::result::Result::Ok(())
            // }
        }
    };

    (@rev [ $single:ident ] $grad:ident $s:ident) => {
        let ($grad, accum) = $s.$single.backward($grad, $single);
        if let ::core::option::Option::Some(grad_w) = accum {
            // accumulate gradients
            for (g, val) in ::core::iter::Iterator::zip($crate::nn::tensors::TensorOps::data_mut($crate::internal::Layer::weights_mut(&mut $s.$single).get_grad_mut()).iter_mut(), $crate::nn::tensors::TensorOps::data(&grad_w)) {
                *g += val;
            }
        }
    };

    (@rev [ $first:ident $(, $rest:ident)+ ] $grad:ident $s:ident) => {
        $crate::static_model!(@rev [ $( $rest ),+ ] $grad $s);
        let ($grad, accum) = $s.$first.backward($grad, $first);
        if let ::core::option::Option::Some(grad_w) = accum {
            // accumulate gradients
            for (g, val) in ::core::iter::Iterator::zip($crate::nn::tensors::TensorOps::data_mut($crate::internal::Layer::weights_mut(&mut $s.$first).get_grad_mut()).iter_mut(), $crate::nn::tensors::TensorOps::data(&grad_w)) {
                *g += val;
            }
        }
    };

    (@rev $( $layer:ident ),+ ; $grad:ident $s:ident ) => {
        $crate::static_model!(@rev [ $( $layer ),+ ] $grad $s);
    };
}

/// A structure containing all the information obtained from a test.
#[derive(Debug, Clone, Copy)]
pub struct TestEval {
    /// The loss earned by a model from it's chosen function. {#}
    pub loss: f64,

    /// The precise accuracy (with very small epsilon). {%}
    ///
    /// It should be expected that this is always very low, if
    /// it even rises from 0%.
    pub accuracy: f64,

    /// The imaginary "score" of a model. {%}
    ///
    /// Like accuracy, but with a dynamic epsilon that gives
    /// lower scores for higher error.
    pub score: f64,
}

impl TestEval {
    /// Determines which is "better" based on the loss.
    #[must_use]
    pub const fn is_better_than(&self, rhs: &Self) -> bool {
        self.loss > rhs.loss
    }

    /// Returns `rhs` if it is "better" than `self`, else `self`.
    #[must_use]
    pub const fn best_of<'a>(&'a self, rhs: &'a Self) -> &'a Self {
        if self.is_better_than(rhs) {
            self
        } else {
            rhs
        }
    }

    /// Sets `self` to `rhs` if and only if it is "better" than `self`.
    pub const fn set_if_better<'a>(&'a mut self, rhs: &'a Self) {
        if rhs.is_better_than(self) {
            *self = *rhs;
        }
    }
}

/// A dataset of 3 tensors (input, gradient, target).
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
        F1: Flatten<IS, Flattened = [f64; IS]> + StaticShape<IR>,
        F2: Flatten<OS, Flattened = [f64; OS]> + StaticShape<OR>,
    >(
        inputs: &F1,
        targets: &F2,
    ) -> Self {
        Self::from_parts(
            *inputs.sliced_shape(),
            inputs.flatten(),
            *targets.sliced_shape(),
            targets.flatten(),
        )
    }

    /// Construct a dataset from raw numeric data and shapes.
    #[must_use]
    #[allow(clippy::unnecessary_cast)]
    pub fn from_parts(
        inputs_shape: [usize; IR],
        inputs_data: [f64; IS],
        targets_shape: [usize; OR],
        targets_data: [f64; OS],
    ) -> Self {
        // fill inputs
        let inputs_iter = inputs_data.into_iter().map(|val| val as TensorFloat);
        let mut inputs_data = [0.0; IS];
        for (i, val) in inputs_iter.enumerate() {
            inputs_data[i] = val;
        }

        // fill targets
        let targets_iter = targets_data.into_iter().map(|val| val as TensorFloat);
        let mut targets_data = [0.0; OS];
        for (i, val) in targets_iter.enumerate() {
            targets_data[i] = val;
        }

        let inputs_tensor = Tensor::<IR, IS>::new(&inputs_shape, &inputs_data);
        let targets_tensor = Tensor::<OR, OS>::new(&targets_shape, &targets_data);

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

    /// Mutably obtains the inputs of the dataset.
    pub const fn inputs_mut(&mut self) -> &mut WithGrad<Tensor<IR, IS>> {
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
    fn invoke(&mut self, args: Arg) -> Ret;
}

/// An abstraction over self-destroying closures.
pub trait ClosureOnce<Arg, Ret> {
    /// Invoke the closure.
    fn invoke(self, args: Arg) -> Ret;
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
            fn invoke(&mut self, args: A) -> R {
                self.call(args)
            }
        }
    };
}

macro_rules! __generate_opaque_once_impl {
    ($struct:ident, $buf:ident) => {
        impl<'a, A, R, const B: usize> ClosureOnce<A, R> for $struct<'a, A, R, $buf<B>> {
            fn invoke(self, args: A) -> R {
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
    fn invoke(&mut self, args: A) -> R {
        self(args)
    }
}

impl<A, R, F: FnOnce(A) -> R> ClosureOnce<A, R> for F {
    fn invoke(self, args: A) -> R {
        self(args)
    }
}
