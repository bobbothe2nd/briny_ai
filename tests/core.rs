//! Tests for core tensor operations, autograd, and CPU ops

use briny_ai::tensors::*;
use briny_ai::ops::cpu::{matmul, mse_loss, relu, sgd};

#[test]
fn test_bpat_save_and_load() {
    use briny_ai::modelio::{save_model, load_model};

    let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Tensor::new(vec![1, 4], vec![7.0, 8.0, 9.0, 10.0]);
    let original = vec![a.clone(), b.clone()];

    save_model("test_model.bpat", &original).unwrap();

    let loaded = load_model("test_model.bpat").unwrap();

    assert_eq!(original, loaded);
}

#[test]
fn test_tensor_creation() {
    let t = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(t.shape, vec![2, 2]);
    assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_tensor_map() {
    let t = Tensor::new(vec![2], vec![1.0, 2.0]);
    let t2 = t.map(|x| x * 2.0);
    assert_eq!(t2.data, vec![2.0, 4.0]);
}

#[test]
fn test_tensor_zip_map() {
    let a = Tensor::new(vec![2], vec![1.0, 2.0]);
    let b = Tensor::new(vec![2], vec![3.0, 4.0]);
    let c = a.zip_map(&b, |x, y| x + y);
    assert_eq!(c.data, vec![4.0, 6.0]);
}

#[test]
fn test_tensor_zeros() {
    let z = Tensor::zeros(vec![3]);
    assert_eq!(z.data, vec![0.0; 3]);
}

#[test]
fn test_with_grad_creation() {
    let t = Tensor::new(vec![2], vec![1.0, 2.0]);
    let wg = WithGrad::new(t.clone());
    assert_eq!(wg.value, t);
    assert_eq!(wg.grad.data, vec![0.0, 0.0]);
}

#[test]
fn test_tensor_with_grad_conversion() {
    let t = Tensor::new(vec![2], vec![1.0, 2.0]);
    let wg = t.with_grad();
    assert_eq!(wg.value.data, vec![1.0, 2.0]);
    assert_eq!(wg.grad.data, vec![0.0, 0.0]);
}

#[test]
fn test_tensor_macro() {
    use briny_ai::tensor;
    let a = tensor!([
        [1.0, 2.0, 3.0], // row 1
        [4.0, 5.0, 6.0], // row 2
    ]);
    let b = Tensor::new(vec![2, 3], vec![
        1.0, 2.0, 3.0,   // row 1
        4.0, 5.0, 6.0,  // row 2
    ]);

    assert_eq!(a, b);
}

#[test]
fn test_matmul_basic() {
    let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).with_grad();
    let b = Tensor::new(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).with_grad();
    let (out, _) = matmul(&a, &b);
    assert_eq!(out.shape, vec![2, 2]);
    assert_eq!(out.data, vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn test_mse_loss_basic() {
    let pred = Tensor::new(vec![2], vec![1.0, 2.0]).with_grad();
    let target = Tensor::new(vec![2], vec![2.0, 2.0]);
    let (loss, back) = mse_loss(&pred, &target);
    assert_eq!(loss, 0.5);
    let grad = back(1.0);
    assert_eq!(grad.data, vec![-1.0, 0.0]);
}

#[test]
fn test_relu_forward_and_backward() {
    let input = Tensor::new(vec![4], vec![-1.0, 0.0, 1.0, 2.0]).with_grad();
    let (out, back) = relu(&input);
    assert_eq!(out.data, vec![0.0, 0.0, 1.0, 2.0]);
    let grad_input = back(&Tensor::new(vec![4], vec![1.0, 1.0, 1.0, 1.0]));
    assert_eq!(grad_input.data, vec![0.0, 0.0, 1.0, 1.0]);
}

#[test]
fn test_sgd_updates_and_zeroing() {
    let mut param = Tensor::new(vec![2], vec![1.0, 2.0]).with_grad();
    param.grad = Tensor::new(vec![2], vec![0.1, 0.5]);
    sgd(&mut param, 0.1);
    assert_eq!(param.value.data, vec![0.99, 1.95]);
    assert_eq!(param.grad.data, vec![0.0, 0.0]);
}
