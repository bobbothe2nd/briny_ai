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
//! - Tensors are strongly typed: `Tensor<T>` for any element type (usually `f32` or `f64`)
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
//! use briny_ai::tensors::Tensor;
//! let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
//! assert_eq!(t.shape, vec![2, 3]);
//! ```

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

impl<T> Tensor<T> {
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

/// A container for tracking gradients of values (used in autograd).
///
/// Typically used as `WithGrad<Tensor<f32>>` or `WithGrad<f32>`.
#[derive(Debug, Clone)]
pub struct WithGrad<T> {
    pub value: T,
    pub grad: T,
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
    a: &'a WithGrad<Tensor<f32>>, 
    b: &'a WithGrad<Tensor<f32>>
) -> (Tensor<f32>, impl Fn(&Tensor<f32>) -> (Tensor<f32>, Tensor<f32>) + 'a) {
    assert_eq!(a.value.shape, b.value.shape);

    let out = Tensor::new(
        a.value.shape.clone(),
        a.value.data.iter().zip(&b.value.data).map(|(x, y)| x + y).collect()
    );

    let a_shape = a.value.shape.clone();
    let b_shape = b.value.shape.clone();

    let back = move |grad_output: &Tensor<f32>| {
        (
            Tensor::new(a_shape.clone(), grad_output.data.clone()),
            Tensor::new(b_shape.clone(), grad_output.data.clone()),
        )
    };

    (out, back)
}

/// Computes `a + b` for scalars, with autograd-compatible backward pass.
pub fn add(a: &WithGrad<f32>, b: &WithGrad<f32>) -> (f32, impl Fn(f32) -> (f32, f32)) {
    let y = a.value + b.value;
    let back = move |grad_output: f32| (grad_output, grad_output);
    (y, back)
}

/// Computes `a * b` for scalars, with autograd-compatible backward pass.
pub fn mul(a: &WithGrad<f32>, b: &WithGrad<f32>) -> (f32, impl Fn(f32) -> (f32, f32)) {
    let y = a.value * b.value;
    let back = move |grad_output: f32| (grad_output * b.value, grad_output * a.value);
    (y, back)
}

/// Performs SGD update in-place: `param -= lr * grad`, resets gradient to 0.0.
///
/// # Panics
/// Panics if shapes of `value` and `grad` mismatch.
pub fn sgd(w: &mut WithGrad<Tensor<f64>>, lr: f64) {
    for (w_i, g_i) in w.value.data.iter_mut().zip(&w.grad.data) {
        *w_i -= lr * *g_i;
    }
    for g_i in &mut w.grad.data {
        *g_i = 0.0;
    }
}

/// Defines a tensor from nested literal arrays.
///
/// Supports arbitrary dimensionality as long as sublists are uniform in shape.
///
/// # Example
/// ```
/// use briny_ai::tensor;
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

/// Parses a JSON string containing a flat or nested array into a `Tensor<f64>`.
///
/// # Format
/// Accepts standard JSON arrays (e.g. `[1, 2]` or `[[1.0, 2.0], [3.0, 4.0]]`).
///
/// # Returns
/// A `Tensor<f64>` if parsing succeeds, else an error string.
///
/// # Limitations
/// - Input must be syntactically valid JSON arrays.
/// - Ragged arrays (non-uniform shapes) are rejected.
/// - Only `f64` values are supported.
pub fn parse_tensor(json: &str) -> Result<Tensor<f64>, &'static str> {
    enum Tok { LBrack, RBrack, Comma, Num(f64) }

    fn next_token(i: &mut usize, s: &[u8]) -> Result<Tok, &'static str> {
        while *i < s.len() && s[*i].is_ascii_whitespace() { *i += 1; }
        if *i >= s.len() { return Err("unexpected EOF"); }
        let c = s[*i];
        *i += 1;
        Ok(match c {
            b'[' => Tok::LBrack,
            b']' => Tok::RBrack,
            b',' => Tok::Comma,
            b'-' | b'0'..=b'9' => {
                let start = *i - 1;
                while *i < s.len() && (s[*i].is_ascii_digit() || s[*i] == b'.' || s[*i] == b'e' || s[*i] == b'E' || s[*i] == b'+' || s[*i] == b'-') {
                    *i += 1;
                }
                let num = std::str::from_utf8(&s[start..*i]).unwrap().parse::<f64>()
                           .map_err(|_| "bad number")?;
                Tok::Num(num)
            }
            _ => return Err("invalid char"),
        })
    }

    let mut idx = 0;
    let bytes = json.as_bytes();
    let mut dims: Vec<usize> = Vec::new();
    let mut data: Vec<f64>  = Vec::new();
    let mut level = 0;
    let mut expect_val = true;

    loop {
        let t = next_token(&mut idx, bytes)?;
        match t {
            Tok::LBrack => {
                if dims.len() == level { dims.push(0); }
                level += 1;
                expect_val = true;
            }
            Tok::RBrack => {
                if expect_val && dims[level] != 0 { return Err("trailing comma"); }
                level -= 1;
                if level == 0 { break; }
                if dims[level] == 0 { dims[level] = dims[level+1]; }
                if dims[level] != dims[level+1] { return Err("ragged tensor"); }
                dims[level+1] = 0;
                expect_val = false;
            }
            Tok::Comma => {
                if expect_val { return Err("comma where value expected"); }
                expect_val = true;
            }
            Tok::Num(n) => {
                if !expect_val { return Err("two values without comma"); }
                data.push(n);
                dims[level] += 1;
                expect_val = false;
            }
        }
    }
    Ok(Tensor::new(dims[..level].to_vec(), data))
}
