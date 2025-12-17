//! Utilities to approximate the accuracy of an inference via tests.

use super::Tensor;
use crate::{
    approx::{approx_eq, ApproxEquality, RelativeEq},
    nn::tensors::TensorGrad,
};
use tensor_optim::TensorOps;

fn __percentage_correct<const D: usize, const N: usize>(
    output: &Tensor<D, N>,
    target: &Tensor<D, N>,
) -> f32 {
    let mut correct = 0f32;
    for (a, b) in output.data().iter().zip(target.data().iter()) {
        let prec = (a - b).approx_eq(&0.0);
        if prec == ApproxEquality::Precise {
            // when roughly exactly equal, give a full point
            correct += 1.0;
        } else if prec == ApproxEquality::Partial {
            // if not precise, it should be partially equal
            correct += 0.5;
        } else if prec == ApproxEquality::Relative {
            // it really should at least be relatively close
            correct += 0.25;
        } else if (a - b).abs() < 0.1 {
            // epsilon of 0.1 to provide overly optimistic results
            correct += 0.05;
        }
    }
    (correct * 100.0) / (target.len() as f32)
}

/// Estimates the accuracy of an inference with respect to the target.
///
/// Logically, this will return more optimistic results than [`accuracy_of`] and should not be trusted
/// as the correct accuracy of the inference.
#[must_use]
#[cfg(feature = "dyntensor")]
pub fn percentage_correct(output: &Tensor<0, 0>, target: &Tensor<0, 0>) -> f32 {
    __percentage_correct::<0, 0>(output, target)
}
/// Estimates the accuracy of an inference with respect to the target.
///
/// Logically, this will return more optimistic results than [`accuracy_of`] and should not be trusted
/// as the correct accuracy of the inference.
#[must_use]
#[cfg(not(feature = "dyntensor"))]
pub fn percentage_correct<const D: usize, const N: usize>(
    output: &Tensor<D, N>,
    target: &Tensor<D, N>,
) -> f32 {
    __percentage_correct::<D, N>(output, target)
}

fn __accuracy_of<const D: usize, const N: usize>(
    output: &Tensor<D, N>,
    target: &Tensor<D, N>,
) -> f32 {
    let mut correct = 0f32;
    for (a, b) in output.data().iter().zip(target.data().iter()) {
        if approx_eq(a, b) {
            // when roughly exactly equal, give a full point
            correct += 1.0;
        }
    }
    (correct * 100.0) / (target.len().min(output.len()) as f32)
}

/// Approximates the accuracy of the output based off it's target.
///
/// The accuracy returned by this as an `f32` is based off how many elements of each tensor are
/// relatively equal in the form of percentage, with the assumption that the lengths are the same.
#[must_use]
#[cfg(feature = "dyntensor")]
pub fn accuracy_of(output: &Tensor<0, 0>, target: &Tensor<0, 0>) -> f32 {
    __accuracy_of::<0, 0>(output, target)
}
/// Approximates the accuracy of the output based off it's target.
///
/// The accuracy returned by this as an `f32` is based off how many elements of each tensor are
/// relatively equal in the form of percentage, with the assumption that the lengths are the same.
#[must_use]
#[cfg(not(feature = "dyntensor"))]
pub fn accuracy_of<const D: usize, const N: usize>(
    output: &Tensor<D, N>,
    target: &Tensor<D, N>,
) -> f32 {
    __accuracy_of::<D, N>(output, target)
}
