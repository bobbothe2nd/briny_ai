# `briny_ai`

`briny_ai` offers a tensor system like no other, without requiring dynamic allocation. You can freely run any `briny_ai` model on whatever target you wish and it will happily compile to code with endless optimizations. GPUs, SIMD, cache locality, too much acceleration for it to be named (CUDA support still has no implementation)!

There is even model serialization/deserialization support at long last. To top that, the low level intrinsics are wrapped in high level datasets and abstracted away by a simple macro system that generates models automatically with unrolled training and testing loops.

The recent additions in v0.6.0 have enabled the creation of generative AI with tools like cross entropy loss and softmax activation. See examples like `small_lm_size` and `transformer` for use of such tools. The main goal for v0.6.1 (or v0.7.0) is to make a pre-norm transformer using the macro and maybe add experimental CUDA support.

## Example

A basic example showcasing use of this crate is below:

```rust
use briny_ai::prelude::*;

static_model!(
    @loss mse_loss
    @optim Sgd
    @model XorModel(model)
    {
        InputLayer([4, 2]),
        {
            conv0: Collapse([4, 2]) => CollapseLayer,
            act0: Activation([4, 1], ReLU) => ActivationLayer,
        },
        OutputLayer([4, 1]),
    }
);

fn main() {
    let base_inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let base_targets = [[0.0], [1.0], [1.0], [0.0]];
    let dataset = Dataset::new(&base_inputs, &base_targets);

    let mut model = XorModel::new();

    let loss = model.fit(&dataset, 100);
    println!("TRAINING: loss={:?}, lr={:?}", loss, model.get_lr());

    let eval = model.infer(&dataset);
    println!("TESTING: loss={:?}, accuracy={:?}%, score={:?}%", eval.loss, eval.accuracy, eval.score);

    model.save("path/to/model", BpatHeader::BpatV1).unwrap();
}
```

For implementations of the tensor system (as opposed to the macro system), see the `examples/m-*.rs` files in the repository or look at the documentation on `docs.rs`.

## Contributing

Contributions, bug reports, and suggestions are welcome! This project aims to help build performance-focused foundations for low-level and embedded Rust ML.

CUDA acceleration, WGPU performance boosts, and adding more layers are the 3 largest areas for improvement - consider contributing in one of those domains. Or if your up for the challenge, consiser adding support for pre-norm transformers in the model (`x += f(ln(x))`). All transformers currently use a custom `x = ln(x) + f(ln(x))` instead. If that still doesn't satiate your desires, you may want to fix the stack overflow (and heap overflow) bugs with huge tensors.

## License

`briny_ai`, as with all my projects, is under an MIT license.
