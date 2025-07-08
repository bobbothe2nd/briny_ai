# briny_ai

A minimal, dependency-free deep learning core for scalar and tensor autograd, written in Rust.

This library provides low-level primitives for defining and training differentiable models on top of multi-dimensional arrays (`Tensor<T>`), supporting:

- Elementwise operations with autograd
- Matrix multiplication with gradient tracking
- Loss functions (MSE)
- Stochastic gradient descent (SGD)
- JSON and binary-based tensor serialization
- Compile-time tensor creation macros

## Features

- Pure Rust, no dependencies
- Compact and fast `.bpat` binary model format
- Forward + backward computation via closures
- Extensible tensor structure with runtime shape checking

---

## Usage

To use `briny_ai`, add the following to your `Cargo.toml`:

```toml
[dependencies]
briny_ai = "0.1.0"
```

## Example

```rust
use briny_ai::tensors::{Tensor, WithGrad};
use briny_ai::backprop::{relu, matmul, mse_loss, sgd};

fn main() {
    let x = WithGrad::from(Tensor::new(vec![1, 2], vec![1.0, 2.0]));
    let w = WithGrad::from(Tensor::new(vec![2, 1], vec![0.5, -1.0]));

    let (y, back1) = matmul(&x, &w);
    let (a, back2) = relu(&WithGrad::from(y));
    let target = Tensor::new(vec![1, 1], vec![0.0]);
    let (loss, back3) = mse_loss(&WithGrad::from(a.clone()), &target);

    let grad_a = back3(1.0);
    let grad_y = back2(&grad_a);
    let (grad_x, grad_w) = back1(&grad_y);

    println!("Loss: {:.4}", loss);
    println!("Gradients: {:?}", grad_w);
}
```

### Saving & Loading

```rust
use briny_ai::modelio::{save_model, load_model};

let tensor = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
save_model("model.bpat", &[tensor.clone()]).unwrap();

let tensors = load_model("model.bpat").unwrap();
assert_eq!(tensors[0], tensor);
```

## Limitations

- Only f64 tensors are supported
- No broadcasting or shape inference yet
- No support for convolution or GPU acceleration
- Autograd is manual via backward closures

## Roadmap

- Broadcasting + batched ops
- Drop-in replacements for BLAS (SIMD)
- CUDA / WebGPU backend support
- Graph-based autograd (reuse, optimization)
- Custom layers & high-level model struct

## Contributing

PRs welcome. This project is early-stage but cleanly structured and easy to extend. Open issues or ideas any time! It's MIT Licensed, too.
