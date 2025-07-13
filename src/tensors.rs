//! Core tensor data structures and operations.
//! 
//! # Core Tensor Utilities
//!
//! This module defines the core logic for representing, manipulating, and computing with multi-dimensional arrays, or tensors.
//!
//! It supports:
//! - Construction of N-dimensional tensors with shape and row-major data layout
//! - Elementary operations like tensor addition
//! - Scalar arithmetic with autograd-compatible `WithGrad` wrappers
//! - SGD-style parameter updates
//! - Parsing tensor data from nested JSON or literals
//! - Compile-time tensor macros
//!
//! ## Design Highlights
//! - Tensors are strongly typed: `Tensor<T>` for any element type (usually `f64`)
//! - Shape is stored as a `Vec<usize>` and enforced at runtime
//! - `WithGrad<T>` pairs any value with its gradient for autograd
//! - The `tensor!` macro supports ergonomic tensor creation from nested arrays
//! - The `parse_tensor` function loads tensors from lightweight JSON without `serde`
//!
//! ## Limitations
//! - Row-major only
//! - No broadcasting, slicing, or shape inference
//! - `parse_tensor` is currently limited to `f64` values
//!
//! ## Example
//!
//! ```rust
//! use briny_ai::{tensor, tensors::{Tensor}};
//! 
//! let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
//! assert_eq!(t.shape, vec![2, 3]);
//! ```
//! 
//! This flow diagram shows how training should be:
//! ```text
//!                 +---------+       +--------+
//! input a ------> | matmul  |-----> | relu   |----+
//!                 +---------+       +--------+    |
//!                                                 v
//!                                              loss (mse)
//!                                                 |
//! target <----------------------------------------+
//! ```
//! 
//! Forward: x → matmul → relu → matmul → mse_loss → scalar
//! Backward: scalar grad ← mse_loss grad ← matmul grad ← relu grad ← matmul grad ← x.grad

use briny::trust::UntrustedData;

/// Represents an N-dimensional tensor with a shape and flat row-major data.
///
/// - All elements must be the same type (`T`).
/// - `shape` defines the structure, e.g., `[2, 3]` for a 2×3 matrix.
/// - `data` holds the flattened content in row-major order.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<T> {
    pub shape: Vec<usize>,
    pub data: Vec<T>,
}

/// A type alias meaning Tensor<f64>
pub type Ten64 = Tensor<f64>;

/// A minimal trait that represents a tensor-like structure supporting gradients.
pub trait TensorGrad: Clone {
    /// Returns the shape of the tensor.
    fn shape(&self) -> &[usize];

    /// Returns the number of elements in the tensor.
    fn len(&self) -> usize;

    /// Returns whether the tensor is empty or not.
    fn is_empty(&self) -> bool;

    /// Creates a zero-filled tensor of the same shape.
    fn zeros_like(&self) -> Self;
}

impl<T: Default + Clone> Tensor<T> {
    /// Creates a new tensor with the given shape and flat data.
    ///
    /// # Panics
    /// Panics if the number of elements in `data` does not match the shape product.
    pub fn new(shape: impl Into<Vec<usize>>, data: Vec<T>) -> Self {
        let shape = shape.into();
        assert_eq!(
            shape.iter().product::<usize>(),
            data.len(),
            "shape {:?} is incompatible with {} data elements",
            shape,
            data.len()
        );
        Self { shape, data }
    }

    /// Replaces this tensor's data with another tensor of the same shape.
    ///
    /// # Panics
    /// Panics if shapes do not match.
    pub fn update(&mut self, mut other: Tensor<T>) {
        assert_eq!(self.shape, other.shape, "shape mismatch");
        std::mem::swap(&mut self.data, &mut other.data);
    }
}

impl<T: Copy + Default> Tensor<T> {
    pub fn transpose(&self) -> Tensor<T> {
        assert_eq!(self.shape.len(), 2, "Transpose only supports 2D tensors");
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut transposed = vec![T::default(); rows * cols];

        for i in 0..rows {
            for j in 0..cols {
                transposed[j * rows + i] = self.data[i * cols + j];
            }
        }

        Tensor::new(vec![cols, rows], transposed)
    }
}

impl Ten64 {
    /// Applies a unary function element-wise to the tensor, returning a new tensor.
    ///
    /// # Example
    /// ```rust
    /// use briny_ai::tensors::Tensor;
    /// 
    /// let x = Tensor::new(vec![2], vec![1.0, -1.0]);
    /// let y = x.map(|v| v.abs());
    /// assert_eq!(y.data, vec![1.0, 1.0]);
    /// ```
    ///
    /// # Arguments
    /// - `f`: A function that maps `f64 -> f64`
    ///
    /// # Returns
    /// A new `Ten64` with each element transformed by `f`.
    pub fn map<F: Fn(f64) -> f64>(&self, f: F) -> Self {
        Self::new(self.shape.clone(), self.data.iter().map(|&x| f(x)).collect())
    }

    /// Applies a binary function element-wise between two tensors of the same shape,
    /// returning a new tensor. Useful for defining differentiable pairwise operations.
    ///
    /// # Panics
    /// Panics if the shapes of `self` and `other` do not match.
    ///
    /// # Example
    /// ```rust
    /// use briny_ai::tensors::Tensor;
    /// 
    /// let a = Tensor::new(vec![2], vec![1.0, 2.0]);
    /// let b = Tensor::new(vec![2], vec![3.0, 4.0]);
    /// let c = a.zip_map(&b, |x, y| x + y);
    /// assert_eq!(c.data, vec![4.0, 6.0]);
    /// ```
    ///
    /// # Arguments
    /// - `other`: The other tensor to zip with
    /// - `f`: A function mapping `(f64, f64) -> f64`
    ///
    /// # Returns
    /// A new `Ten64` resulting from applying `f` element-wise.
    pub fn zip_map<F: Fn(f64, f64) -> f64>(&self, other: &Self, f: F) -> Self {
        assert_eq!(self.shape, other.shape);
        Self::new(self.shape.clone(), self.data.iter().zip(&other.data).map(|(&a, &b)| f(a, b)).collect())
    }

    /// Constructs a new tensor filled with zeros of a specified shape.
    ///
    /// # Example
    /// ```rust
    /// use briny_ai::tensors::Tensor;
    /// 
    /// let t = Tensor::zeros(vec![3, 2]);
    /// assert_eq!(t.data, vec![0.0; 6]);
    /// assert_eq!(t.shape, vec![3, 2]);
    /// ```
    ///
    /// # Arguments
    /// - `shape`: Shape of the tensor (any iterable that converts to `Vec<usize>`)
    ///
    /// # Returns
    /// A `Ten64` of the given shape, filled with `0.0`.
    pub fn zeros(shape: impl Into<Vec<usize>>) -> Self {
        let shape = shape.into();
        let size = shape.iter().product();
        Tensor::new(shape, vec![f64::default(); size])
    }

    /// Multiplies a tensor by another tensor.
    /// 
    /// NOTE: This is performed on the CPU!
    /// Computes the gradient of a matrix multiplication `C = A @ B` with respect to A and B,
    /// given `grad_output = dL/dC`, the upstream gradient.
    ///
    /// Returns `(dA, dB)` such that:
    /// - `dA = grad_output @ B.T`
    /// - `dB = A.T @ grad_output`
    ///
    /// This assumes `self` is `grad_output` and is shaped `[m, n]`.
    pub fn matmul(&self, a: &Ten64, b: &Ten64) -> (Ten64, Ten64) {
        let (m, k) = (a.shape[0], a.shape[1]);
        let n = b.shape[1];

        assert_eq!(self.shape, vec![m, n], "grad shape mismatch");
        assert_eq!(a.shape, vec![m, k], "input A shape mismatch");
        assert_eq!(b.shape, vec![k, n], "input B shape mismatch");

        let mut da = vec![0.0; m * k];
        let mut db = vec![0.0; k * n];

        let a_data = &a.data;
        let b_data = &b.data;
        let grad = &self.data;

        for i in 0..m {
            for kk in 0..k {
                let a_ik = a_data[i * k + kk];
                for j in 0..n {
                    let g = grad[i * n + j];
                    da[i * k + kk] += g * b_data[kk * n + j];
                    db[kk * n + j] += a_ik * g;
                }
            }
        }

        (
            Tensor::new(vec![m, k], da),
            Tensor::new(vec![k, n], db),
        )
    }
}

impl TensorGrad for Ten64 {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn is_empty(&self) -> bool {
        self.data.len() == 0
    }

    fn zeros_like(&self) -> Self {
        Self::new(self.shape.clone(), vec![0.0; self.data.len()])
    }
}

/// A container for tracking gradients of values (used in autograd).
///
/// Typically used as `WithGrad<Ten64>` or `WithGrad<f64>`.
#[derive(Debug, Clone)]
pub struct WithGrad<T> {
    pub value: T,
    pub grad: T,
}

/// A trait mainly for converting `Tensor`s to `WithGrad`.
pub trait IntoWithGrad: TensorGrad + Sized {
    /// Wraps the tensor with a zero-initialized gradient.
    fn with_grad(self) -> WithGrad<Self> {
        WithGrad::new(self)
    }
}

impl<T: TensorGrad> IntoWithGrad for T {}

impl<T: TensorGrad> WithGrad<T> {
    /// Creates a new `WithGrad` wrapper with zero-initialized gradient.
    pub fn new(value: T) -> Self {
        let grad = value.zeros_like();
        Self { value, grad }
    }
}

/// Adds two tensors element-wise, returning result and backprop function.
///
/// # Panics
/// Panics if shapes do not match.
///
/// # Returns
/// - Output tensor
/// - Closure that computes gradients for both inputs given `dL/dout`
pub fn add_tensor<'a>(
    a: &'a WithGrad<Ten64>, 
    b: &'a WithGrad<Ten64>
) -> (Ten64, impl Fn(&Ten64) -> (Ten64, Ten64) + 'a) {
    assert_eq!(a.value.shape, b.value.shape);

    let out = Tensor::new(
        a.value.shape.clone(),
        a.value.data.iter().zip(&b.value.data).map(|(x, y)| x + y).collect()
    );

    let a_shape = a.value.shape.clone();
    let b_shape = b.value.shape.clone();

    let back = move |grad_output: &Ten64| {
        (
            Tensor::new(a_shape.clone(), grad_output.data.clone()),
            Tensor::new(b_shape.clone(), grad_output.data.clone()),
        )
    };

    (out, back)
}

/// Computes `a + b` for scalars, with autograd-compatible backward pass.
pub fn add(a: &WithGrad<f64>, b: &WithGrad<f64>) -> (f64, impl Fn(f64) -> (f64, f64)) {
    let y = a.value + b.value;
    let back = move |grad_output: f64| (grad_output, grad_output);
    (y, back)
}

/// Computes `a * b` for scalars, with autograd-compatible backward pass.
pub fn mul(a: &WithGrad<f64>, b: &WithGrad<f64>) -> (f64, impl Fn(f64) -> (f64, f64)) {
    let y = a.value * b.value;
    let back = move |grad_output: f64| (grad_output * b.value, grad_output * a.value);
    (y, back)
}

/// Defines a tensor from nested literal arrays.
///
/// Supports arbitrary dimensionality as long as sublists are uniform in shape.
///
/// # Example
/// ```rust
/// use briny_ai::tensor;
///
/// let t = tensor!([[1.0, 2.0], [3.0, 4.0]]);
/// assert_eq!(t.shape, vec![2, 2]);
/// ```
#[macro_export]
macro_rules! tensor {
    ($lit:literal) => {
        $crate::tensors::Tensor::new(Vec::<usize>::new(), vec![$lit])
    };
    ([ $( $inner:tt ),+ $(,)? ]) => {{
        let children = vec![ $( tensor!($inner) ),+ ];
        let first_shape = &children[0].shape;
        assert!(children.iter().all(|c| c.shape == *first_shape),
            "ragged tensor literal (rows have mismatched shapes)");
        let mut shape = vec![children.len()];
        shape.extend_from_slice(first_shape);
        let mut data = Vec::with_capacity(children.len() * children[0].data.len());
        for c in children { data.extend(c.data); }
        $crate::tensors::Tensor::new(shape, data)
    }};
}

/// Parses a JSON string containing a flat or nested array into a `Ten64`.
///
/// # Format
/// Accepts standard JSON arrays (e.g. `[1, 2]` or `[[1.0, 2.0], [3.0, 4.0]]`).
///
/// # Returns
/// A `Ten64` if parsing succeeds, else an error string.
///
/// # Limitations
/// - Input must be syntactically valid JSON arrays.
/// - Ragged arrays (non-uniform shapes) are rejected.
/// - Only `f64` values are supported.
pub fn parse_tensor(json: &str) -> Result<Tensor<f64>, &'static str> {
    let untrusted = UntrustedData::new(json);
    validate_tensor(untrusted)
}

fn validate_tensor(untrusted: UntrustedData<&str>) -> Result<Tensor<f64>, &'static str> {
    let json = untrusted.value();
    let bytes = json.as_bytes();
    let mut idx = 0;

    enum Tok { LBrack, RBrack, Comma, Num(f64) }

    fn next_token(i: &mut usize, s: &[u8]) -> Result<Option<Tok>, &'static str> {
        while *i < s.len() && s[*i].is_ascii_whitespace() { *i += 1; }
        if *i >= s.len() { return Ok(None); }
        let c = s[*i];
        *i += 1;
        Ok(Some(match c {
            b'[' => Tok::LBrack,
            b']' => Tok::RBrack,
            b',' => Tok::Comma,
            b'-' | b'0'..=b'9' => {
                let start = *i - 1;
                while *i < s.len() &&
                    (s[*i].is_ascii_digit() || matches!(s[*i], b'.' | b'e' | b'E' | b'+' | b'-')) {
                    *i += 1;
                }
                let num_str = std::str::from_utf8(&s[start..*i]).map_err(|_| "invalid utf8")?;
                let num = num_str.parse::<f64>().map_err(|_| "bad number")?;
                Tok::Num(num)
            }
            _ => return Err("invalid character"),
        }))
    }

    fn parse_value(
        idx: &mut usize,
        s: &[u8],
        out: &mut Vec<f64>,
    ) -> Result<Vec<usize>, &'static str> {
        if let Some(tok) = next_token(idx, s)? {
            match tok {
                Tok::Num(n) => {
                    out.push(n);
                    Ok(vec![]) // scalar
                }
                Tok::LBrack => {
                    let mut shapes = Vec::new();
                    let mut count = 0;
                    let mut first = true;
                    loop {
                        // peek RBrack
                        let save = *idx;
                        match next_token(idx, s)? {
                            Some(Tok::RBrack) => break,
                            Some(_) => *idx = save,
                            None => return Err("unexpected EOF"),
                        }

                        if !first {
                            match next_token(idx, s)? {
                                Some(Tok::Comma) => {}
                                _ => return Err("missing comma"),
                            }
                        }

                        let child_shape = parse_value(idx, s, out)?;
                        if let Some(prev_shape) = shapes.first() {
                            if child_shape != *prev_shape {
                                return Err("ragged tensor");
                            }
                        }
                        shapes.push(child_shape);
                        count += 1;
                        first = false;
                    }

                    Ok([vec![count], shapes.first().unwrap_or(&vec![]).clone()].concat())
                }
                _ => Err("expected number or array"),
            }
        } else {
            Err("unexpected EOF")
        }
    }

    let mut flat_data = Vec::new();
    let shape = parse_value(&mut idx, bytes, &mut flat_data)?;

    while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
        idx += 1;
    }

    if idx != bytes.len() {
        return Err("trailing characters after tensor");
    }

    let expected_len: usize = shape.iter().product();
    if expected_len != flat_data.len() {
        return Err("data shape mismatch");
    }

    Ok(Tensor { shape, data: flat_data })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tensor_valid() {
        let input = "[[1.0, 2.0], [3.0, 4.0]]";
        let tensor = parse_tensor(input).expect("parse failed");

        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_parse_tensor_ragged() {
        let input = "[[1.0], [2.0, 3.0]]";
        let err = parse_tensor(input).unwrap_err();
        assert_eq!(err, "ragged tensor");
    }

    #[test]
    fn test_parse_tensor_trailing_comma() {
        let input = "[1.0, 2.0,]";
        let err = parse_tensor(input).unwrap_err();
        assert_eq!(err, "expected number or array");
    }

    #[test]
    fn test_parse_tensor_missing_comma() {
        let input = "[1.0 2.0]";
        let err = parse_tensor(input).unwrap_err();
        assert_eq!(err, "missing comma");
    }

    #[test]
    fn test_parse_tensor_data_shape_mismatch() {
        let input = "[[1.0, 2.0], [3.0, 4.0, 5.0]]";
        let err = parse_tensor(input).unwrap_err();
        assert_eq!(err, "ragged tensor");
    }

    #[test]
    fn test_parse_tensor_single_value() {
        let input = "[[42.0]]";
        let tensor = parse_tensor(input).expect("parse failed");
        assert_eq!(tensor.shape, vec![1, 1]);
        assert_eq!(tensor.data, vec![42.0]);
    }

    #[test]
    fn test_parse_tensor_whitespace_okay() {
        let input = "  [ \n[1.0 , 2.0] ,\n[3.0,4.0] ] ";
        let tensor = parse_tensor(input).expect("parse failed");
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.data, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
