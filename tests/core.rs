//! Tests for core tensor operations, autograd, and CPU ops

use briny_ai::nn::ops::cpu::{matmul, mse_loss, relu, sgd};
use briny_ai::nn::tensors::*;

#[test]
#[cfg(all(feature = "std"))]
fn bpat_save_and_load() {
    use briny_ai::nn::io::{load_tensors, save_tensors, BpatHeader};

    let a = Tensor::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Tensor::new(&[1, 4], &[7.0, 8.0, 9.0, 10.0]);
    let original = vec![a.as_vectensor(), b.as_vectensor()];

    save_tensors(
        "checkpoints/test/test_tensors.bpat",
        &original,
        BpatHeader::default(),
    )
    .unwrap();

    let loaded = load_tensors("checkpoints/test/test_tensors.bpat").unwrap();

    assert_eq!(original, loaded);
}

#[test]
fn tensor_creation() {
    let t = Tensor::new(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(t.shape(), [2, 2]);
    assert_eq!(t.data(), [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn tensor_zeros() {
    let z = Tensor::new(&[3], &[5f32; 3]).zeros_like();
    assert_eq!(z.data(), &[0.0; 3]);
}

#[test]
fn with_grad_creation() {
    let t = Tensor::new(&[2], &[1.0, 2.0]);
    let wg = WithGrad::new(t.clone());
    assert_eq!(wg.get_value(), &t);
    assert_eq!(wg.get_grad().data(), &[0.0, 0.0]);
}

#[test]
fn tensor_with_grad_conversion() {
    let t = Tensor::new(&[2], &[1.0, 2.0]);
    let wg = t.with_grad();
    assert_eq!(wg.get_value().data(), &[1.0, 2.0]);
    assert_eq!(wg.get_grad().data(), &[0.0, 0.0]);
}

#[test]
fn matmul_basic() {
    let a = Tensor::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).with_grad();
    let b = Tensor::new(&[3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).with_grad();
    #[cfg(all(feature = "alloc", feature = "dyntensor"))]
    let (out, _) = matmul(&a, &b);
    #[cfg(not(all(feature = "alloc", feature = "dyntensor")))]
    let (out, _) = matmul::<6, 6, 4, 2>(&a, &b);
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.data(), &[58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn mse_loss_basic() {
    let pred = Tensor::new(&[2], &[1.0, 2.0]).with_grad();
    let target = Tensor::new(&[2], &[2.0, 2.0]);
    let (loss, back) = mse_loss(&pred, &target);
    assert_eq!(loss, 0.5);
    #[cfg(feature = "alloc")]
    let grad = back(1.0);
    #[cfg(not(feature = "alloc"))]
    let grad = back.call(1.0);
    assert_eq!(grad.data(), &[-1.0, 0.0]);
}

#[test]
fn relu_forward_and_backward() {
    let input = Tensor::new(&[4], &[-1.0, 0.0, 1.0, 2.0]).with_grad();
    let (out, back) = relu(&input);
    assert_eq!(out.data(), &[0.0, 0.0, 1.0, 2.0]);
    #[cfg(feature = "alloc")]
    let grad_input = back(Tensor::new(&[4], &[1.0; 4]));
    #[cfg(not(feature = "alloc"))]
    let grad_input = back.call(Tensor::new(&[4], &[1.0; 4]));
    assert_eq!(grad_input.data(), &[0.0, 0.0, 1.0, 1.0]);
}

#[test]
fn sgd_updates_and_zeroing() {
    let mut param = Tensor::new(&[2], &[1.0, 2.0]).with_grad();
    *param.get_grad_mut() = Tensor::new(&[2], &[0.1, 0.5]);
    sgd(&mut param, 0.1);
    assert_eq!(param.get_value().data(), &[0.99, 1.95]);
    assert_eq!(param.get_grad().data(), &[0.0, 0.0]);
}
