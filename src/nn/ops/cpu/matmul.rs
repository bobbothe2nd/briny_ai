use crate::nn::tensors::Tensor;
use crate::nn::tensors::WithGrad;
use crate::nn::TensorFloat;

#[cfg(feature = "alloc")]
use alloc::boxed::Box;
#[cfg(not(feature = "alloc"))]
use box_closure::{Align8, OpaqueFn};

/// Performs a matrix multiplication `C = A × B` on two 2D tensors (`A: m×k`, `B: k×n`),
/// returning the result tensor and a closure for backpropagation.
///
/// # Requirements
///
/// - Shapes must be compatible: `A.shape = [m, k]` and `B.shape = [k, n]`.
///
/// # Returns
///
/// - Output tensor of shape `[m, n]`
/// - Backward function computing gradients w.r.t. `A` and `B`
///
/// # Panics
///
/// - If the inner dimensions of `A` and `B` do not match.
#[must_use]
#[cfg(feature = "dyntensor")]
pub fn matmul<'a>(
    a: &'a WithGrad<Tensor<TensorFloat>>,
    b: &'a WithGrad<Tensor<TensorFloat>>,
) -> (
    Tensor<TensorFloat>,
    Box<dyn Fn(Tensor<TensorFloat>) -> (Tensor<TensorFloat>, Tensor<TensorFloat>) + 'a>,
) {
    // clone shallow metadata; data is shared or cloned shallowly per your impl
    let a_val = a.get_value();
    let b_val = b.get_value();

    // forward matmul: a [m, k] * b [k, n] -> [m, n]
    let out = a_val.matmul(b_val);

    // backward closure computing gradients explicitly using your grad functions
    let back = move |grad: Tensor<TensorFloat>| {
        // precompute transposes
        let b_t = b_val.transpose();
        let a_t = a_val.transpose();

        let grad_wrt_a = grad.matmul(&b_t);
        let grad_wrt_b = a_t.matmul(&grad);

        (grad_wrt_a, grad_wrt_b)
    };

    (out, Box::new(back))
}

/// Performs a matrix multiplication `C = A × B` on two 2D tensors (`A: m×k`, `B: k×n`),
/// returning the result tensor and a closure for backpropagation.
///
/// # Confusing Compiler Errors
///
/// Sometimes the size of the output cannot be figured out by the compiler.
/// If the compiler needs you to specify what const generic `N` is, that's
/// what's happenning. To fix this, make sure you're calling the backwards
/// closure provided correctly, this will usually fix the error.
///
/// If the error still occurs, the compiler probably just wants you to
/// explicitly tell it what the answer is.
///
/// # Requirements
///
/// - Shapes must be compatible: `A.shape = [m, k]` and `B.shape = [k, n]`.
///
/// # Returns
///
/// - Output tensor of shape `[m, n]`
/// - Backward function computing gradients w.r.t. `A` and `B`
///
/// # Panics
///
/// If the shapes of the tensors is not what is expected by the function, this
/// method panics:
///
/// - both must be at least two dimensional
/// - inner dimensions must match
/// - batch dimensions must match
///
/// These are checked twice in debug mode for sanity. Since that's unnecessary,
/// the first checks are disabled in release mode, leaving the resposibility to
/// an external crate, `tensor_optim`.
#[must_use]
#[cfg(all(feature = "alloc", not(feature = "dyntensor")))]
pub fn matmul<'a, const A: usize, const B: usize, const OUT: usize, const D: usize>(
    a: &'a WithGrad<Tensor<TensorFloat, A, D>>,
    b: &'a WithGrad<Tensor<TensorFloat, B, D>>,
) -> (
    Tensor<TensorFloat, OUT, D>,
    Box<
        dyn Fn(
                Tensor<TensorFloat, OUT, D>,
            ) -> (Tensor<TensorFloat, A, D>, Tensor<TensorFloat, B, D>)
            + 'a,
    >,
) {
    let a_val = a.get_value();
    let b_val = b.get_value();

    // forward matmul
    let out = a_val.matmul(b_val);

    // gradients closure
    let back = move |grad: Tensor<TensorFloat, OUT, D>| {
        // transpose last two axes
        let b_t = b_val.transpose();
        let a_t = a_val.transpose();

        // compute gradients
        let grad_wrt_a = grad.matmul(&b_t); // shape: (M, K)
        let grad_wrt_b = a_t.matmul(&grad); // shape: (K, N)

        (grad_wrt_a, grad_wrt_b)
    };

    (out, Box::new(back))
}

/// Performs a matrix multiplication `C = A × B` on two 2D tensors (`A: m×k`, `B: k×n`),
/// returning the result tensor and a closure for backpropagation.
///
/// # Confusing Compiler Errors
///
/// Sometimes the size of the output cannot be figured out by the compiler.
/// If the compiler needs you to specify what const generic `N` is, that's
/// what's happenning. To fix this, make sure you're calling the backwards
/// closure provided correctly, this will usually fix the error.
///
/// If the error still occurs, the compiler probably just wants you to
/// explicitly tell it what the answer is.
///
/// # Requirements
///
/// - Shapes must be compatible: `A.shape = [m, k]` and `B.shape = [k, n]`.
///
/// # Returns
///
/// - Output tensor of shape `[m, n]`
/// - Backward function computing gradients w.r.t. `A` and `B`
///
/// # Panics
///
/// If the shapes of the tensors is not what is expected by the function, this
/// method panics:
///
/// - both must be at least two dimensional
/// - inner dimensions must match
/// - batch dimensions must match
///
/// These are checked twice in debug mode for sanity. Since that's unnecessary,
/// the first checks are disabled in release mode, leaving the resposibility to
/// an external crate, `tensor_optim`.
#[must_use]
#[cfg(not(feature = "alloc"))]
pub fn matmul<'a, const A: usize, const B: usize, const OUT: usize, const D: usize>(
    a: &'a WithGrad<Tensor<TensorFloat, A, D>>,
    b: &'a WithGrad<Tensor<TensorFloat, B, D>>,
) -> (
    Tensor<TensorFloat, OUT, D>,
    OpaqueFn<
        'a,
        Tensor<TensorFloat, OUT, D>,
        (Tensor<TensorFloat, A, D>, Tensor<TensorFloat, B, D>),
        Align8<128>,
    >,
) {
    let a_val = a.get_value();
    let b_val = b.get_value();

    // forward matmul
    let out = a_val.matmul(b_val);

    // gradients closure
    let back = move |grad: Tensor<TensorFloat, OUT, D>| {
        // transpose last two axes
        let b_t = b_val.transpose();
        let a_t = a_val.transpose();

        // compute gradients
        let grad_wrt_a = grad.matmul(&b_t); // shape: (M, K)
        let grad_wrt_b = a_t.matmul(&grad); // shape: (K, N)

        (grad_wrt_a, grad_wrt_b)
    };

    (out, OpaqueFn::new(back))
}
