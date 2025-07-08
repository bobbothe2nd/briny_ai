use briny_ai::{tensors::{Tensor, WithGrad}, tensor};
use briny_ai::backprop::{matmul, relu, mse_loss, sgd};

#[test]
fn test_tensor_creation() {
    let t = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(t.shape, vec![2, 2]);
    assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_tensor_macro() {
    let t = tensor!([[1.0, 2.0], [3.0, 4.0]]);
    assert_eq!(t.shape, vec![2, 2]);
    assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_matmul_backprop() {
    let a = WithGrad {
        value: Tensor::new(vec![2, 3], vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        ]),
        grad: Tensor::new(vec![2, 3], vec![0.0; 6])
    };
    let b = WithGrad {
        value: Tensor::new(vec![3, 2], vec![
            7.0, 8.0,
            9.0, 10.0,
            11.0, 12.0
        ]),
        grad: Tensor::new(vec![3, 2], vec![0.0; 6])
    };
    
    let (output, backward) = matmul(&a, &b);
    assert_eq!(output.shape, vec![2, 2]);
    let grad_output = Tensor::new(vec![2, 2], vec![1.0, 1.0, 1.0, 1.0]);
    let (grad_a, grad_b) = backward(&grad_output);
    assert_eq!(grad_a.shape, vec![2, 3]);
    assert_eq!(grad_b.shape, vec![3, 2]);
}

#[test]
fn test_relu_backprop() {
    let t = WithGrad {
        value: Tensor::new(vec![3], vec![-1.0, 0.0, 1.0]),
        grad: Tensor::new(vec![3], vec![0.0; 3]),
    };
    let (output, backward) = relu(&t);
    assert_eq!(output.data, vec![0.0, 0.0, 1.0]);
    let grad_output = Tensor::new(vec![3], vec![1.0, 1.0, 1.0]);
    let grad = backward(&grad_output);
    assert_eq!(grad.data, vec![0.0, 0.0, 1.0]);
}

#[test]
fn test_mse_loss() {
    let pred = WithGrad {
        value: Tensor::new(vec![2], vec![1.0, 2.0]),
        grad: Tensor::new(vec![2], vec![0.0; 2]),
    };
    let target = Tensor::new(vec![2], vec![1.5, 2.5]);
    let (loss, backward) = mse_loss(&pred, &target);
    let grad = backward(1.0);
    assert_eq!(loss, 0.25);
    assert_eq!(grad.data, vec![-0.5, -0.5]);
}

#[test]
fn test_sgd() {
    let mut w = WithGrad {
        value: Tensor::new(vec![2], vec![1.0, 2.0]),
        grad: Tensor::new(vec![2], vec![0.1, 0.2]),
    };
    sgd(&mut w, 0.5);
    assert_eq!(w.value.data, vec![0.95, 1.9]);
    assert_eq!(w.grad.data, vec![0.0, 0.0]);
}