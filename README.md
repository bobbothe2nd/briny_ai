# `briny_ai`

`briny_ai` offers a tensor system like no other, without requiring dynamic allocation. You can freely run any `briny_ai` model on whatever target you wish and it will happily compile to code with endless optimizations. GPUs, SIMD, cache locality, too much acceleration for it to be named (CUDA support still has no implementation)! There is model serialization/deserialization support under an unstable API, this cannot be accessed by users of the crate though.

To top that, the low level intrinsics are wrapped in high level datasets and abstracted away by a complex macro system that generates models automatically with unrolled training and testing loops. Currently, only inference models are supported due to the limited ability of the tensor system. The four operations supported are:

- ReLU
- SGD
- Matrix Multiplication
- MSE Loss

Among these are many other BLAS primitive operations (add, mul, div, sub, transpose, etc.).

## Example

A basic example showcasing use of this crate is below:

```rust
use briny_ai::prelude::{
    Dataset, static_model,
};

static_model!(
    @loss mse_loss
    @model XorModel
    {
        InputLayer([4, 2]),
        {
            conv0: Dense([4, 2]) => DenseLayer,
            act0: Activation([4, 1], ReLU) => ActivationLayer,
        },
        OutputLayer([4, 1]),
    }
);

fn main() {
    let base_inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let base_targets = [[0.0], [1.0], [1.0], [0.0]];
    let dataset = Dataset::new(&base_inputs, &base_targets);

    let mut model = XorModel::new().with_lr(0.01);

    let loss = model.fit(&dataset, 100);
    println!("TRAINING: loss={:?}, lr={:?}", loss, model.get_lr());

    let eval = model.test(&dataset, 5);
    println!("TESTING: loss={:?}, accuracy={:?}%, score={:?}%", eval.loss, eval.accuracy, eval.score);
}
```

For implementations of the tensor system (as opposed to the macro system), see the `examples/` directory or look at the documentation on `docs.rs`.

## Contributing

Contributions, bug reports, and suggestions are welcome! This project aims to help build performance-focused foundations for low-level and embedded Rust ML.

Tensor serialization/deserialization, CUDA acceleration, WGPU performance boosts, and extending capabilities of the macro system are the 4 largest areas for improvement - consider contributing in one of those domains.

## License

`briny_ai`, as with all my projects, is under an MIT license.
