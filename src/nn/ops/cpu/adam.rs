use crate::nn::tensors::{Tensor, WithGrad};
use crate::nn::TensorFloat;
use tensor_optim::TensorOps;

/// Performs one step of Adam optimization on the given parameter tensor.
///
/// # Arguments
///
/// - `w`: Tensor with gradient to be updated
/// - `m`: First moment estimate (same shape as `w`)
/// - `v`: Second moment estimate (same shape as `w`)
/// - `t`: Current timestep (1-based)
/// - `lr`: Learning rate
///
/// # Hyperparameters (hardcoded)
///
/// - beta1 = 0.9
/// - beta2 = 0.999
/// - eps = 1e-8
#[cfg(feature = "dyntensor")]
pub fn adam(
    w: &mut WithGrad<Tensor<TensorFloat>>,
    m: &mut Tensor<TensorFloat>,
    v: &mut Tensor<TensorFloat>,
    t: TensorFloat,
    lr: TensorFloat,
) {
    let beta1: TensorFloat = 0.9;
    let beta2: TensorFloat = 0.999;
    let eps: TensorFloat = 1e-8;

    let (params, grads) = w.split_mut();
    let params_data = params.data_mut();
    let grads_data = grads.data();
    let m_data = m.data_mut();
    let v_data = v.data_mut();

    #[allow(clippy::suspicious_operation_groupings)]
    for ((param, grad), (m_val, v_val)) in params_data
        .iter_mut()
        .zip(grads_data.iter())
        .zip(m_data.iter_mut().zip(v_data.iter_mut()))
    {
        *m_val = beta1 * *m_val + (1.0 - beta1) * *grad;
        *v_val = beta2 * *v_val + (1.0 - beta2) * (*grad * *grad);

        let m_hat = *m_val / super::pow(1.0 - beta1, t);
        let v_hat = *v_val / super::pow(1.0 - beta2, t);

        *param -= lr * m_hat / (super::sqrt(v_hat) + eps);
    }

    for grad in grads.data_mut() {
        *grad = 0.0;
    }
}

/// Performs one step of Adam optimization on the given parameter tensor.
///
/// # Arguments
///
/// - `w`: Tensor with gradient to be updated
/// - `m`: First moment estimate (same shape as `w`)
/// - `v`: Second moment estimate (same shape as `w`)
/// - `t`: Current timestep (1-based)
/// - `lr`: Learning rate
///
/// # Hyperparameters (hardcoded)
///
/// - beta1 = 0.9
/// - beta2 = 0.999
/// - eps = 1e-8
#[cfg(not(feature = "dyntensor"))]
pub fn adam<const N: usize, const D: usize>(
    w: &mut WithGrad<Tensor<TensorFloat, N, D>>,
    m: &mut Tensor<TensorFloat, N, D>,
    v: &mut Tensor<TensorFloat, N, D>,
    t: TensorFloat,
    lr: TensorFloat,
) {
    let beta1: TensorFloat = 0.9;
    let beta2: TensorFloat = 0.999;
    let eps: TensorFloat = 1e-8;

    let (params, grads) = w.split_mut();
    let params_data = params.data_mut();
    let grads_data = grads.data();
    let m_data = m.data_mut();
    let v_data = v.data_mut();

    #[allow(clippy::suspicious_operation_groupings)]
    for ((param, grad), (m_val, v_val)) in params_data
        .iter_mut()
        .zip(grads_data.iter())
        .zip(m_data.iter_mut().zip(v_data.iter_mut()))
    {
        *m_val = beta1 * *m_val + (1.0 - beta1) * *grad;
        *v_val = beta2 * *v_val + (1.0 - beta2) * (*grad * *grad);

        let m_hat = *m_val / super::pow(1.0 - beta1, t);
        let v_hat = *v_val / super::pow(1.0 - beta2, t);

        *param -= lr * m_hat / (super::sqrt(v_hat) + eps);
    }

    for grad in grads.data_mut() {
        *grad = 0.0;
    }
}
