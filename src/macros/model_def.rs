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
/// InputLayer([...]) - describes the input shape of the dataset
/// {
///   [name] : [layer] ([...][(temp, decay rate)?], [activation?]) => [expanded layer][([...])]?
///   ...
/// }
/// OutputLayer([...]) - describes the output shape of the targets
/// ```
///
/// Each `...` represents a placeholder for the shape of that layer with the exception
/// of the expanded layer. The I/O layers are compiled away after construction, and the
/// specified shapes are assumed to match that of the adjacent hidden layer. In the case
/// of the expanded layer, the `...` represents a placeholder for the shape of that layers
/// output - this is just an optional type annotation that can fix many compile errors.
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
///     @optim sgd
///     @model XorModel
///     {
///         InputLayer([4, 2]),
///         {
///             conv0: Collapse([4, 2]) => CollapseLayer,
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
        @optim $optim:ident $(($lr:expr))?
        @model $name:ident
        {
            InputLayer([$($in_RANK:expr),+]),
            {$( $field:ident : $layer:ident ([$($RANK:expr),+] $(($temp:expr, $temp_scale:expr))? $(, $activation:ident)?$(,)?) => $full_layer:ident$(([$($layer_out_RANK:expr),*]))?, )*},
            OutputLayer([$($out_RANK:expr),+]),
        }
    ) => {
        pub struct $name {
            $(
                $field: $crate::macros::$full_layer<
                    { <[()]>::len(&[$( { let _ = &$RANK; () } ),+]) },
                    { 1 $(* $RANK)* }
                >,
            )*

            lr: $crate::nn::TensorFloat,
        }

        impl $name {
            const IN_RANK: usize = [$($in_RANK),*].len();
            const IN_SIZE: usize = { 1 $(* $in_RANK)* };

            const OUT_RANK: usize = [$($out_RANK),*].len();
            const OUT_SIZE: usize = { 1 $(* $out_RANK)* };

            #[allow(unused_variables)]
            const TENSOR_COUNT: usize = [$({let $field = ();}),*].len();

            const SERIALIZED_TENSOR_CAPACITY_BPATV0: usize = { (0 $(+ (1 $(* $RANK)*))* * ::core::mem::size_of::<f32>()) + (0 $(+ ([$($RANK),*].len()))* * ::core::mem::size_of::<u64>()) };
            const SERIALIZED_TENSOR_CAPACITY_BPATV1: usize = { Self::SERIALIZED_TENSOR_CAPACITY_BPATV0 + ((0 $(+ {let _ = [$($RANK),*]; 1})*) * ::core::mem::size_of::<u32>()) + ::core::mem::size_of::<u32>() };

            pub fn new(variance: f32) -> Self {
                struct Builder {
                    $(
                        $field: $crate::macros::$layer<
                            { <[()]>::len(&[$( { let _ = &$RANK; () } ),+]) },
                            { 1 $(* $RANK)* }
                            $(, $crate::macros::$activation)?
                        >,
                    )*
                }

                impl Builder {
                    fn new(variance: f32) -> Self {
                        Self {
                            $(
                                $field: $crate::macros::$layer {
                                    shape: [$($RANK),+],
                                    data: $crate::macros::asym_distr(variance, 0),
                                    $( temp: $temp, )?
                                    $( _activation: ::core::marker::PhantomData::<$crate::macros::$activation>, )?
                                },
                            )*
                        }
                    }

                    fn build(self) -> $name {
                        $name {
                            $(
                                $field: self.$field.build(),
                            )*

                            lr: 0.01,
                        }$(.with_lr($lr))?
                    }
                }

                const {
                    ::core::assert!(<[()]>::len(&[$( { let _ = &$in_RANK; () } ),+]) >= 2, "input layer must be at least 2 dimensions");
                    ::core::assert!(<[()]>::len(&[$( { let _ = &$out_RANK; () } ),+]) >= 2, "output layer must be at least 2 dimensions");
                    $(
                        ::core::assert!(<[()]>::len(&[$( { let _ = &$RANK; () } ),+]) >= 2, "all layers must be at least 2 dimensions");
                    )*
                }

                Builder::new(variance).build()
            }

            #[allow(clippy::unnecessary_casts)]
            pub const fn set_lr(&mut self, lr: f32) {
                self.lr = lr as $crate::nn::TensorFloat;
            }

            pub const fn with_lr(self, lr: f32) -> Self {
                let mut copied = self;
                copied.set_lr(lr);
                copied
            }

            #[allow(clippy::unnecessary_casts)]
            pub const fn get_lr(&self) -> f32 {
                self.lr as f32
            }

            pub fn fit_epoch(
                &mut self,
                dataset: &$crate::macros::Dataset<
                    { Self::IN_RANK },
                    { Self::IN_SIZE },
                    { Self::OUT_RANK },
                    { Self::OUT_SIZE },
                >
            ) -> f32 {
                // forward pass (fast)
                let input = dataset.inputs();
                let out = ::core::clone::Clone::clone(input.get_value());

                // shadowing for each hidden layer
                $(
                    $(let temp = self.$field.get_temp_relative($temp);)?
                    let input = $crate::nn::tensors::IntoWithGrad::with_grad(out $(/ self.$field.get_temp_relative($temp) as $crate::nn::TensorFloat)?);
                    $(
                        self.$field.update_temp(|x: f32| $crate::macros::decay_temp(temp, $temp_scale));
                    )?
                    let (out, $field)$(: ($crate::macros::Tensor<{ [$($layer_out_RANK),*].len() }, { 1 $(* $layer_out_RANK)* }>, _))? = self.$field.forward(&input);
                )*

                // compute loss
                let output = $crate::nn::tensors::IntoWithGrad::with_grad(out);
                let (loss, back_loss) = $crate::nn::ops::dispatch::$loss(&output, dataset.targets());

                // backward pass (unrolled)
                let grad_output = $crate::macros::Closure::invoke(&back_loss, loss);

                // hidden layers backward (reversed order)
                $crate::static_model!(@rev $($field),+; grad_output self);

                // apply optimizer to non-IO layers
                $(
                    $crate::macros::Layer::apply_update(
                        &mut self.$field,
                        self.lr,
                        $crate::nn::ops::dispatch::$optim,
                    );
                )*

                #[allow(clippy::unnecessary_casts)]
                {
                    loss as f32
                }
            }

            pub fn fit(
                &mut self,
                dataset: &$crate::macros::Dataset<
                    { Self::IN_RANK },
                    { Self::IN_SIZE },
                    { Self::OUT_RANK },
                    { Self::OUT_SIZE },
                >,
                epochs: usize,
                lr_update: impl Fn((f32, f32, f32), (f32, f32), bool) -> (f32, f32, f32, bool),
            ) -> f32 {
                let mut max_lr = 0.2;
                let mut min_lr = 1e-3;
                let mut loss = 0_f32;
                let mut prev = 0_f32;
                let mut panic = false;
                for i in 0..epochs {
                    if !panic {
                        prev = loss;
                    }

                    loss = self.fit_epoch(dataset);

                    let lr;
                    (lr, min_lr, max_lr, panic) = lr_update((self.lr as f32, min_lr, max_lr), (prev, loss), panic);
                    self.lr = lr as $crate::nn::TensorFloat;
                }
                loss // return the most recent loss
            }

            pub fn sample(
                &self,
                dataset: &$crate::macros::Context<
                    { Self::IN_RANK },
                    { Self::IN_SIZE },
                >,
            ) -> [f32; Self::OUT_SIZE] {
                // forward pass (fast)
                let input = dataset.raw();
                let out = ::core::clone::Clone::clone(input.get_value());

                // shadowing for each hidden layer
                $(
                    let input = $crate::nn::tensors::IntoWithGrad::with_grad(out);
                    let (out, _)$(: ($crate::macros::Tensor<{ [$($layer_out_RANK),*].len() }, { 1 $(* $layer_out_RANK)* }>, _))? = self.$field.forward(&input);
                )*
                let out: $crate::macros::Tensor<{ Self::OUT_RANK }, { Self::OUT_SIZE }> = out;

                let mut buf = [0_f32; Self::OUT_SIZE];
                #[allow(clippy::unnecessary_casts)]
                for (i, v) in $crate::nn::tensors::TensorOps::data(&out).iter().map(|&x| x as f32).enumerate() {
                    buf[i] = v;
                }
                buf
            }

            pub fn infer(
                &self,
                dataset: &$crate::macros::Dataset<
                    { Self::IN_RANK },
                    { Self::IN_SIZE },
                    { Self::OUT_RANK },
                    { Self::OUT_SIZE },
                >,
            ) -> $crate::macros::test::TestEval {
                // forward pass (fast)
                let input = dataset.inputs();
                let out = ::core::clone::Clone::clone(input.get_value());

                // shadowing for each hidden layer
                $(
                    let input = $crate::nn::tensors::IntoWithGrad::with_grad(out);
                    let (out, _)$(: ($crate::macros::Tensor<{ [$($layer_out_RANK),*].len() }, { 1 $(* $layer_out_RANK)* }>, _))? = self.$field.forward(&input);
                )*

                // compute loss
                let output = $crate::nn::tensors::IntoWithGrad::with_grad(out);
                let (loss, _) = $crate::nn::ops::dispatch::$loss(&output, dataset.targets());
                let score = $crate::macros::test::percentage_correct(output.get_value(), dataset.targets());
                let acc = $crate::macros::test::accuracy_of(output.get_value(), dataset.targets());

                #[allow(clippy::unnecessary_casts)]
                $crate::macros::test::TestEval {
                    loss: loss as f32,
                    score,
                    acc,
                }
            }

            // pub fn save(
            //     &self,
            //     path: &str,
            // ) -> ::core::result::Result<(), $crate::nn::io::SerialTensorError> {
            //     $crate::nn::io::save_tensors(path)
            // }

            // pub fn load(
            //     &mut self,
            //     path: &str,
            // ) -> ::core::result::Result<(), $crate::nn::io::SerialTensorError> {
            //     $crate::nn::io::load_tensors(path)
            // }
        }
    };

    (@rev [ $single:ident ] $grad:ident $s:ident) => {
        let ($grad, accum) = $s.$single.backward($grad, $single);
        if let ::core::option::Option::Some(grad_w) = accum {
            for w in $crate::macros::Layer::weights_mut(&mut $s.$single) {
                // accumulate gradients
                ::core::iter::Iterator::zip(
                    $crate::nn::tensors::TensorOps::data_mut(
                        w.get_grad_mut()
                    ).iter_mut(),
                    $crate::nn::tensors::TensorOps::data(&grad_w)
                ).for_each(|(g, val)| *g += val);
            }
        }
    };

    (@rev [ $first:ident $(, $rest:ident)+ ] $grad:ident $s:ident) => {
        $crate::static_model!(@rev [ $( $rest ),+ ] $grad $s);
        let ($grad, accum) = $s.$first.backward($grad, $first);
        if let ::core::option::Option::Some(grad_w) = accum {
            for w in $crate::macros::Layer::weights_mut(&mut $s.$first) {
                // accumulate gradients
                ::core::iter::Iterator::zip(
                    $crate::nn::tensors::TensorOps::data_mut(
                        w.get_grad_mut()
                    ).iter_mut(),
                    $crate::nn::tensors::TensorOps::data(&grad_w)
                ).for_each(|(g, val)| *g += val);
            }
        }
    };

    (@rev $( $layer:ident ),+ ; $grad:ident $s:ident ) => {
        $crate::static_model!(@rev [ $( $layer ),+ ] $grad $s);
    };
}
