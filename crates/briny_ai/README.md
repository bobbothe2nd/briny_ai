# briny_ai

A minimal deep learning core for scalar and tensor autograd, written in Rust.

This library provides low-level primitives for defining and training differentiable models on top of multi-dimensional arrays (`Tensor<T>`,`Tensor<T, N, D>`, `VecTensor<T>`), supporting:

- Elementwise operations with autograd
- Matrix multiplication with gradient tracking
- Loss functions (MSE)
- Stochastic gradient descent (SGD)

---

`briny_ai` may not be a popular crate, but it is a growing one.

|           | `v0.1.0` | `v0.2.0` | `v0.2.1` | `v0.2.2` |
|-----------|----------|----------|----------|----------|
| Size      | 9.92 KiB | 40.4 KiB | 42.3 KiB | 42.1 KiB |
| Downloads | ~200     | ~130     | ~180     | ~150     |

[*] The downloads section is just throughout the first few days

---

## Features

- Compact and fast `.bpat` binary model format with safe, explicit parsing
- Forward + backward computation via closures
- Extensible tensor structure with runtime shape checking and strong data validation
- CPU & GPU acceleration with structured error handling

## Usage

To use `briny_ai`, add it to your `Cargo.toml` or run the command `cargo add briny_ai` in your terminal.

To enable SIMD, pass the feature flag `simd`. Similarly, to enable GPU acceleration, pass the feature flag `wgpu` to the compiler. In order to make use of such features, you should set the backend like:

```rust
set_backend(Backend::Wgpu);
set_backend(Backend::Cpu); //default
set_backend(Backend::Cuda) // same as Wgpu
```

As of `v0.3.0`, all `std`-dependent features like saving and loading files are gated behind the `std` feature flag. Similarly, all dynamic allocations are gated behind an `alloc` feature, enabled whenever `std` is enabled automatically.

**NOTE**: SIMD only works on AVX2 compatible x86_64 devices.

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

## Why Choose `briny_ai`

- Unlike heavyweight frameworks like `tch-rs` or `burn`, `briny_ai` stays small and straightforward. It’s perfect if you want just the core building blocks without bloat or magic.
- You get tight integration with Rust’s type system and memory safety guarantees — minimal unsafe code lurking under the hood. Many other Rust ML crates compromise here.
- You control exactly when and how data is validated and trusted. This explicit trust model helps you avoid sneaky bugs and security risks common in other AI libraries.
- `briny_ai` relies on your own control flow and simple GPU acceleration via wgpu, avoiding large, complex dependencies or runtime surprises.
- If you’re building AI for environments where safety and correctness matter (IoT, secure enclaves, custom hardware), `briny_ai` is tailored for that.
- Because it’s small and clear, you can adapt it to your needs without wading through complex abstractions or C++ FFI layers.

If you want a no-nonsense, Rust-native AI core that’s lean, secure, and explicit — `briny_ai` is the right tool for you.

## Contributing

PRs welcome. This project is early-stage but cleanly structured and easy to extend. Open issues or ideas any time!

Got an Nvidia GPU or know CUDA? Your help is golden!

### License

Under the MIT License.
