# `briny_ai`

`briny_ai` offers a tensor system like no other, without requiring dynamic allocation. You can freely run any `briny_ai` model on whatever target you wish and it will happily compile to code with endless optimizations. GPUs, SIMD, cache locality, too much acceleration for it to be named (CUDA support still has no implementation)!

There is model serialization/deserialization support under an unstable API, this cannot be accessed by users of the crate though. To top that, the low level intrinsics are wrapped in high level datasets and abstracted away by a complex macro system that generates models automatically with unrolled training and testing loops.

The recent additions in v0.5.0 have enabled the creation of generative AI with tools like cross entropy loss and softmax activation. See examples like `char_rnn` for use of such tools. The main goal for v0.5.1 is finally getting the BPATv0/BPATv1 stuff working to reintroduce saving/loading tensors, and by extension, entire models.

## Example

A basic example showcasing use of this crate is below:

```rust
use briny_ai::prelude::{
    Dataset, static_model,
};

static_model!(
    @loss mse_loss
    @optim sgd
    @model XorModel
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
}
```

For implementations of the tensor system (as opposed to the macro system), see the `examples/m-*.rs` files in the repository or look at the documentation on `docs.rs`.

## Contributing

Contributions, bug reports, and suggestions are welcome! This project aims to help build performance-focused foundations for low-level and embedded Rust ML.

Tensor serialization/deserialization, CUDA acceleration, WGPU performance boosts, and adding generic causal layers are the 4 largest areas for improvement - consider contributing in one of those domains.

## License

`briny_ai`, as with all my projects, is under an MIT license.
