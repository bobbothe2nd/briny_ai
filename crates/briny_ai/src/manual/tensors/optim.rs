//! Bindings to `tensor_optim` as the default tensor backend.
//!
//! Much of the logic in this module is just boilerplate code to get the
//! low-level `tensor_optim` crate to behave on more advanced systems.
//! Little here is really logic, just abstractions over what is provided
//! in `tensor_optim`.

use briny::SafeMemory;
#[cfg(not(feature = "dyntensor"))]
use briny::raw::Pod;
use tensor_optim::TensorOps;

/// A trait to flatten nested types.
pub trait Flatten<const N: usize> {
    /// The flattened type (from nested).
    type Flattened;

    /// Flattens `self` if nested, otherwise it's already flat.
    fn flatten(&self) -> Self::Flattened;
}

impl<T: Default + Copy, const X: usize, const Y: usize, const N: usize> Flatten<N> for [[T; X]; Y] {
    type Flattened = [T; N];

    fn flatten(&self) -> Self::Flattened {
        let mut arr = [T::default(); N];
        for (i, chunk) in self.iter().enumerate() {
            let start = i * X;
            let end = start + X;
            arr[start..end].copy_from_slice(chunk);
        }
        arr
    }
}

/// A trait for statically sized nested types.
pub trait StaticShape<const N: usize> {
    /// Returns the shape of `self` as a slice.
    fn sliced_shape(&self) -> &[usize; N];
}

impl<T: Default + Copy, const X: usize, const Y: usize> StaticShape<2> for [[T; X]; Y] {
    fn sliced_shape(&self) -> &[usize; 2] {
        &[X, Y]
    }
}

/// A minimal trait that represents a tensor-like structure supporting gradients.
pub trait TensorGrad: Clone {
    /// Returns the number of elements in the tensor.
    fn len(&self) -> usize;

    /// Returns whether the tensor is empty or not.
    fn is_empty(&self) -> bool;

    #[must_use]
    /// Creates a zero-filled tensor of the same shape.
    fn zeros_like(&self) -> Self;
}

#[cfg(feature = "dyntensor")]
impl<T: Clone + Default> TensorGrad for DynTensor<T> {
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize {
        self.data().len()
    }

    fn zeros_like(&self) -> Self {
        Self::new(&[self.len()])
    }
}

#[cfg(not(feature = "dyntensor"))]
impl<T: Copy + Default, const N: usize, const D: usize> TensorGrad for ArrTensor<T, N, D> {
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize {
        self.data().len()
    }

    fn zeros_like(&self) -> Self {
        Self::new([self.len(); D])
    }
}

#[cfg(feature = "dyntensor")]
use alloc::vec::Vec;

#[cfg(feature = "dyntensor")]
use tensor_optim::DynTensor;

#[cfg(not(feature = "dyntensor"))]
pub use tensor_optim::ArrTensor;

/// A simple tensor with decent flexibility.
///
/// It's entirely heap-allocated, so it won't typically
/// overflow the stack like the no-`alloc` tensor.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[cfg(feature = "dyntensor")]
#[repr(transparent)]
pub struct Tensor<T>(DynTensor<T>);

#[cfg(feature = "dyntensor")]
impl<T: SafeMemory> SafeMemory for Tensor<T> {}

#[cfg(feature = "dyntensor")]
impl<T: Clone> TensorOps<T> for Tensor<T> {
    fn data(&self) -> &[T] {
        self.0.data()
    }

    fn data_mut(&mut self) -> &mut [T] {
        self.0.data_mut()
    }

    fn shape(&self) -> &[usize] {
        self.0.shape()
    }
}

#[cfg(feature = "dyntensor")]
impl<T: Copy + Default> Tensor<T> {
    /// Parses a tensor from nested arrays.
    #[must_use]
    pub fn from_arr<
        const N: usize,
        const M: usize,
        F: Flatten<N, Flattened = [T; N]> + StaticShape<M>,
    >(
        arr: &F,
    ) -> Self {
        Self::new(arr.sliced_shape(), &arr.flatten())
    }
}

#[cfg(feature = "dyntensor")]
impl<T: Clone + Default> Tensor<T> {
    /// Constructs a new `Tensor<T>` with the given shape and data.
    ///
    /// # Panics
    ///
    /// The product of each entry in `shape` must be equal to the length of `data` or the constructor will panic at runtime.
    #[must_use]
    pub fn new(shape: &[usize], data: &[T]) -> Self {
        Self(DynTensor::with_data(shape, data))
    }

    /// Performs a simple transposiition through the center.
    ///
    /// Specifically, the method performs this:
    ///
    /// - for 2D tensors swaps axes 0 and 1
    /// - for other ranks reverses axes order
    #[must_use]
    pub fn transpose(&self) -> Self {
        Self(self.0.transpose())
    }

    /// Multiplies two arbitrary tensors, providing the dot product.
    ///
    /// # Panics
    ///
    /// If the shapes of the tensors is not what is expected by the function, this method panics hard.
    ///
    /// - both must be at least two dimensional
    /// - inner dimensions must match
    /// - batch dimensions must match
    ///
    /// These are checked *twice* in debug mode for sanity.
    /// Since that's unnecessary, the first checks are disabled in release mode, leaving the resposibility to an external crate, `tensor_optim`.
    #[must_use]
    pub fn matmul(&self, rhs: &Self) -> Self
    where
        T: Copy + core::ops::Add<Output = T> + core::ops::Mul<Output = T> + core::ops::AddAssign,
    {
        let self_rank = self.shape().len();
        let rhs_rank = rhs.shape().len();

        // require rank >= 2 for matrix multiply
        debug_assert!(
            !(self_rank < 2 || rhs_rank < 2),
            "matmul requires rank >= 2"
        );

        // validate last dims for multiplication compatibility
        let m = self.shape()[self_rank - 2];
        let k1 = self.shape()[self_rank - 1];
        let k2 = rhs.shape()[rhs_rank - 2];
        let n = rhs.shape()[rhs_rank - 1];

        debug_assert!(k1 == k2, "inner dimensions must match for matmul");

        // validate batch dims match
        let self_batch_dims = &self.shape()[..self_rank - 2];
        let rhs_batch_dims = &rhs.shape()[..rhs_rank - 2];

        debug_assert!(
            self_batch_dims == rhs_batch_dims,
            "batch dimensions must match for matmul"
        );

        // compute expected output shape
        let mut expected_out_shape = Vec::with_capacity(self_rank);
        expected_out_shape.extend_from_slice(self_batch_dims);
        expected_out_shape.push(m);
        expected_out_shape.push(n);

        let mut out = DynTensor::new(&expected_out_shape);
        self.0.matmul(&rhs.0, &mut out);
        Self(out)
    }

    /// Creates a zeroed tensor with the given shape.
    #[must_use]
    pub fn zeros(shape: &[usize]) -> Self {
        Self(DynTensor::new(shape))
    }
}

#[cfg(feature = "dyntensor")]
impl<T: Clone + Default> TensorGrad for Tensor<T> {
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize {
        self.data().len()
    }

    fn zeros_like(&self) -> Self {
        Self(DynTensor::new(self.shape()))
    }
}

/// A compile-time heavy tensor.
///
/// Although it's fast and memory efficient, it also
/// lives entirely on the stack.
#[cfg(not(feature = "dyntensor"))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Tensor<T, const N: usize, const D: usize>(ArrTensor<T, N, D>);

#[cfg(not(feature = "dyntensor"))]
impl<T: SafeMemory, const N: usize, const D: usize> SafeMemory for Tensor<T, N, D> {}
#[cfg(not(feature = "dyntensor"))]
#[allow(unsafe_code)]
unsafe impl<T: Pod, const N: usize, const D: usize> Pod for Tensor<T, N, D> {}

#[cfg(not(feature = "dyntensor"))]
impl<T: Clone, const N: usize, const D: usize> tensor_optim::TensorOps<T> for Tensor<T, N, D> {
    fn data(&self) -> &[T] {
        self.0.data()
    }

    fn data_mut(&mut self) -> &mut [T] {
        self.0.data_mut()
    }

    fn shape(&self) -> &[usize] {
        self.0.shape()
    }
}

#[cfg(not(feature = "dyntensor"))]
impl<T: Clone, const N: usize, const D: usize> tensor_optim::ConstTensorOps<T, N, D>
    for Tensor<T, N, D>
{
    fn data_array(&self) -> &[T; N] {
        self.0.data_array()
    }

    fn data_mut_array(&mut self) -> &mut [T; N] {
        self.0.data_mut_array()
    }

    fn shape_array(&self) -> &[usize; D] {
        self.0.shape_array()
    }
}

#[cfg(not(feature = "dyntensor"))]
impl<T: Copy + Default, const N: usize> Tensor<T, N, 2> {
    /// Parses a tensor from neted arrays.
    #[must_use]
    pub fn from_arr<F: Flatten<N, Flattened = [T; N]> + StaticShape<2>>(arr: &F) -> Self {
        Self::new(arr.sliced_shape(), &arr.flatten())
    }
}

#[cfg(not(feature = "dyntensor"))]
impl<T: Clone + Default, const N: usize, const D: usize> Tensor<T, N, D> {
    /// Constructs a new `Tensor<T>` with the given shape and data.
    ///
    /// # Panics
    ///
    /// The product of each entry in `shape` must be equal to the length of `data` or the constructor will panic at runtime.
    #[must_use]
    pub fn new(shape: &[usize; D], data: &[T; N]) -> Self {
        Self(ArrTensor::with_data(*shape, data.clone()))
    }

    /// Constructs a new `Tensor<T>` from the provided Tensor.
    #[must_use]
    pub const fn from_arrtensor(tensor: ArrTensor<T, N, D>) -> Self {
        Self(tensor)
    }

    /// Performs a simple transposiition through the center.
    ///
    /// Specifically, the method performs this:
    ///
    /// - for 2D tensors swaps axes 0 and 1
    /// - for other ranks reverses axes order
    #[must_use]
    pub fn transpose(&self) -> Self
    where
        T: Copy,
    {
        Self(self.0.transpose())
    }

    /// Multiplies two arbitrary tensors, providing the dot product.
    ///
    /// # Panics
    ///
    /// If the shapes of the tensors is not what is expected by the function, this method panics hard.
    ///
    /// - both must be at least two dimensional
    /// - inner dimensions must match
    /// - batch dimensions must match
    ///
    /// These are checked *twice* in debug mode for sanity.
    /// Since that's unnecessary, the first checks are disabled in release mode, leaving the resposibility to an external crate, `tensor_optim`.
    pub fn matmul<const M: usize, const K: usize>(&self, rhs: &Tensor<T, M, D>) -> Tensor<T, K, D>
    where
        T: Copy + core::ops::Add<Output = T> + core::ops::Mul<Output = T> + core::ops::AddAssign,
    {
        // require rank >= 2 for matrix multiply
        debug_assert!(D >= 2, "matmul requires rank >= 2");

        // validate last dims for multiplication compatibility
        let m = self.shape()[D - 2];
        let k1 = self.shape()[D - 1];
        let k2 = rhs.shape()[D - 2];
        let n = rhs.shape()[D - 1];

        debug_assert!(k1 == k2, "inner dimensions must match for matmul");

        // validate batch dims match
        let self_batch_dims = &self.shape()[..D - 2];
        let rhs_batch_dims = &rhs.shape()[..D - 2];

        debug_assert!(
            self_batch_dims == rhs_batch_dims,
            "batch dimensions must match for matmul"
        );

        // compute expected output shape
        let mut expected_out_shape = [0usize; D];
        expected_out_shape[..D - 2].copy_from_slice(self_batch_dims);
        expected_out_shape[D - 2] = m;
        expected_out_shape[D - 1] = n;

        let mut out = ArrTensor::<T, K, D>::new(expected_out_shape);
        self.0.matmul(&rhs.0, &mut out); // call the low-level kernel
        Tensor::<T, K, D>::from_arrtensor(out)
    }

    /// Creates a zeroed tensor with the given shape.
    #[must_use]
    pub fn zeros(shape: &[usize; D]) -> Self
    where
        T: Copy,
    {
        Self(ArrTensor::new(*shape))
    }
}

#[cfg(not(feature = "dyntensor"))]
impl<T: Copy + Default, const N: usize, const D: usize> TensorGrad for Tensor<T, N, D> {
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize {
        self.data().len()
    }

    fn zeros_like(&self) -> Self {
        use tensor_optim::ConstTensorOps;
        Self(ArrTensor::new(*self.shape_array()))
    }
}

#[cfg(feature = "dyntensor")]
#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::manual::tensors::IntoWithGrad;
    use alloc::vec;

    // Test basic WithGrad construction and zero initialization of gradients
    #[test]
    fn with_grad_basic_construction_and_mutation() {
        let data = vec![1.0f64, 2.0, 3.0, 4.0];
        let shape = &[2, 2];
        let tensor = Tensor::new(shape, &data);

        // Wrap tensor with gradients (should zero grad)
        let mut wg = tensor.clone().with_grad();
        assert_eq!(wg.get_value(), &tensor);
        assert_eq!(wg.get_grad().len(), tensor.len());

        // Gradient should be zero-initialized
        assert!(wg.get_grad().data().iter().all(|&v| v == 0.0));

        // Mutate gradient and value independently
        wg.get_grad_mut().data_mut()[0] = 42.0;
        wg.get_value_mut().data_mut()[1] = 99.0;

        // Check mutations persisted
        assert_eq!(wg.get_grad().data()[0], 42.0);
        assert_eq!(wg.get_value().data()[1], 99.0);

        // Cloning WithGrad clones both value and grad
        let wg_clone = wg.clone();
        assert_eq!(wg_clone.get_grad().data()[0], 42.0);
        assert_eq!(wg_clone.get_value().data()[1], 99.0);

        // Mutations on clone do not affect original (check clone-on-write semantics)
        let mut wg_clone_mut = wg_clone.clone();
        wg_clone_mut.get_value_mut().data_mut()[1] = 100.0;
        assert_ne!(wg_clone_mut.get_value().data()[1], wg.get_value().data()[1]);
    }

    #[test]
    fn tensorgrad_trait_behaviour() {
        let data = vec![5.0f64; 6];
        let tensor = Tensor::new(&[2, 3], &data);
        assert_eq!(tensor.len(), 6);
        assert!(!tensor.is_empty());

        let zeroed = tensor.zeros_like();
        assert_eq!(zeroed.len(), tensor.len());
        assert!(zeroed.data().iter().all(|&v| v == 0.0));

        // For an empty tensor
        let empty_tensor: Tensor<crate::manual::TensorFloat> = Tensor::new(&[0], &[]);
        assert!(empty_tensor.is_empty());
        assert_eq!(empty_tensor.len(), 0);
        let zeroed_empty = empty_tensor.zeros_like();
        assert!(zeroed_empty.is_empty());
    }

    #[test]
    fn withgrad_clone_and_mutation_isolation() {
        let tensor = Tensor::new(&[3], &[10.0, 20.0, 30.0]);
        let mut wg1 = tensor.clone().with_grad();
        let mut wg2 = wg1.clone();

        // Mutate wg1 grad and value
        wg1.get_grad_mut().data_mut()[0] = 1.0;
        wg1.get_value_mut().data_mut()[1] = 99.0;

        // wg2 remains unchanged (clone-on-write semantics)
        assert_eq!(wg2.get_grad().data()[0], 0.0);
        assert_eq!(wg2.get_value().data()[1], 20.0);

        // Mutate wg2 value
        wg2.get_value_mut().data_mut()[2] = 123.0;
        assert_eq!(wg2.get_value().data()[2], 123.0);

        // wg1 value unaffected
        assert_eq!(wg1.get_value().data()[2], 30.0);
    }

    #[test]
    fn tensor_with_grad_zeros_like_matches_shape() {
        let tensor = Tensor::new(&[4, 5], &[0.0; 20]);
        let zeroed = tensor.zeros_like();
        assert_eq!(zeroed.shape(), tensor.shape());
        assert_eq!(zeroed.data().len(), tensor.data().len());
        assert!(zeroed.data().iter().all(|&v| v == 0.0));
    }
}

#[cfg(not(feature = "dyntensor"))]
#[cfg(test)]
mod no_alloc_tests {
    use super::*;

    #[test]
    fn test_tensor_matmul_2x3_by_3x2() {
        // A: 2x3
        let a_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a_shape = [2, 3];
        let a = Tensor::<f32, 6, 2>::new(&a_shape, &a_data);

        // B: 3x2
        let b_data = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let b_shape = [3, 2];
        let b = Tensor::<f32, 6, 2>::new(&b_shape, &b_data);

        // matmul
        let out = a.matmul::<6, 4>(&b);

        // expected output 2x2
        let expected = [58.0, 64.0, 139.0, 154.0];
        assert_eq!(out.shape(), [2, 2]);
        assert_eq!(out.data(), expected);
    }
}
