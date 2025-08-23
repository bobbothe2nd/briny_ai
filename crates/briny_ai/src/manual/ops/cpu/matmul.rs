use crate::manual::tensors::Tensor;
use crate::manual::tensors::WithGrad;

#[cfg(feature = "alloc")]
use alloc::boxed::Box;
#[cfg(not(feature = "alloc"))]
use box_closure::{Align32, OpaqueFn};
#[cfg(feature = "dyntensor")]
use tensor_optim::TensorOps;

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
pub fn matmul<
    'a,
    T: Clone
        + Default
        + Copy
        + core::ops::Add<Output = T>
        + core::ops::Mul<Output = T>
        + core::ops::AddAssign
        + 'a,
>(
    a: &WithGrad<Tensor<T>>,
    b: &WithGrad<Tensor<T>>,
) -> (
    Tensor<T>,
    Box<dyn Fn(Tensor<T>) -> (Tensor<T>, Tensor<T>) + 'a>,
) {
    // validate input shapes (both 2D)
    assert_eq!(a.get_value().shape().len(), 2, "`A` must be 2D for matmul");
    assert_eq!(b.get_value().shape().len(), 2, "`B` must be 2D for matmul");
    // if not 2D, checks would fail:
    assert_eq!(
        a.get_value().shape()[1],
        b.get_value().shape()[0],
        "inner dimensions must match for matmul",
    );

    // clone shallow metadata; data is shared or cloned shallowly per your impl
    let a_val = a.get_value().clone();
    let b_val = b.get_value().clone();

    // forward matmul: a [m, k] * b [k, n] -> [m, n]
    let out = a_val.matmul(&b_val);

    // backward closure computing gradients explicitly using your grad functions
    let back = move |grad: Tensor<T>| {
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
pub fn matmul<'a, T, const A: usize, const B: usize, const OUT: usize>(
    a: &'a WithGrad<Tensor<T, A, 2>>,
    b: &'a WithGrad<Tensor<T, B, 2>>,
) -> (
    Tensor<T, OUT, 2>,
    Box<dyn Fn(Tensor<T, OUT, 2>) -> (Tensor<T, A, 2>, Tensor<T, B, 2>)>,
)
where
    T: Copy
        + core::ops::Add<Output = T>
        + core::ops::Mul<Output = T>
        + Default
        + core::ops::AddAssign
        + 'static,
{
    let a_val = a.get_value().clone();
    let b_val = b.get_value().clone();

    // forward matmul
    let out = a_val.matmul(&b_val);

    // gradients closure
    let back = move |grad: Tensor<T, OUT, 2>| {
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
pub fn matmul<'a, T, const A: usize, const B: usize, const OUT: usize>(
    a: &'a WithGrad<Tensor<T, A, 2>>,
    b: &'a WithGrad<Tensor<T, B, 2>>,
) -> (
    Tensor<T, OUT, 2>,
    OpaqueFn<'a, Tensor<T, OUT, 2>, (Tensor<T, A, 2>, Tensor<T, B, 2>), Align32<256>>,
)
where
    T: Copy
        + core::ops::Add<Output = T>
        + core::ops::Mul<Output = T>
        + Default
        + core::ops::AddAssign,
{
    let a_val = a.get_value().clone();
    let b_val = b.get_value().clone();

    // forward matmul
    let out = a_val.matmul(&b_val);

    // gradients closure
    let back = move |grad: Tensor<T, OUT, 2>| {
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
