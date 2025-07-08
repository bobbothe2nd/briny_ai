use briny_ai::tensors::*;
use briny_ai::backprop::*;

#[test]
fn test_tensor_shape_mismatch_panics() {
    let result = std::panic::catch_unwind(|| {
        Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0]);
    });
    assert!(result.is_err());
}

#[test]
fn test_bpat_save_and_load() {
    use briny_ai::tensors::Tensor;
    use briny_ai::modelio::{save_model, load_model};

    let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Tensor::new(vec![1, 4], vec![7.0, 8.0, 9.0, 10.0]);
    let original = vec![a.clone(), b.clone()];

    save_model("test_model.bpat", &original).unwrap();

    let loaded = load_model("test_model.bpat").unwrap();

    assert_eq!(original, loaded);
}

#[test]
fn test_add_backprop_scalar() {
    let a = WithGrad { value: 2.0, grad: 0.0 };
    let b = WithGrad { value: 3.0, grad: 0.0 };
    let (y, back) = add(&a, &b);
    let (da, db) = back(1.0);
    assert_eq!(y, 5.0);
    assert_eq!(da, 1.0);
    assert_eq!(db, 1.0);
}

#[test]
fn test_mul_backprop_scalar() {
    let a = WithGrad { value: 2.0, grad: 0.0 };
    let b = WithGrad { value: 3.0, grad: 0.0 };
    let (y, back) = mul(&a, &b);
    let (da, db) = back(1.0);
    assert_eq!(y, 6.0);
    assert_eq!(da, 3.0);
    assert_eq!(db, 2.0);
}

#[test]
fn test_relu_backprop() {
    let input = WithGrad {
        value: Tensor::new(vec![3], vec![-1.0, 0.0, 2.0]),
        grad: Tensor::new(vec![3], vec![0.0; 3]),
    };
    let (out, back) = relu(&input);
    assert_eq!(out.data, vec![0.0, 0.0, 2.0]);

    let grad_in = back(&Tensor::new(vec![3], vec![1.0, 1.0, 1.0]));
    assert_eq!(grad_in.data, vec![0.0, 0.0, 1.0]);
}

#[test]
fn test_mse_loss_and_backprop() {
    let pred = WithGrad {
        value: Tensor::new(vec![2], vec![1.0, 2.0]),
        grad: Tensor::new(vec![2], vec![0.0, 0.0]),
    };
    let target = Tensor::new(vec![2], vec![0.0, 0.0]);
    let (loss, back) = mse_loss(&pred, &target);
    let grad = back(1.0);
    assert!(loss > 0.0);
    assert_eq!(grad.shape, vec![2]);
}

#[test]
fn test_matmul_backprop() {
    let a = WithGrad {
        value: Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
        grad: Tensor::new(vec![2, 2], vec![0.0; 4]),
    };
    let b = WithGrad {
        value: Tensor::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]),
        grad: Tensor::new(vec![2, 2], vec![0.0; 4]),
    };
    let (out, back) = matmul(&a, &b);
    assert_eq!(out.shape, vec![2, 2]);
    let (da, db) = back(&Tensor::new(vec![2, 2], vec![1.0; 4]));
    assert_eq!(da.shape, vec![2, 2]);
    assert_eq!(db.shape, vec![2, 2]);
}
