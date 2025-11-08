use crate::nn::tensors::{Tensor, TensorGrad};
use crate::nn::FLOAT_LANES;
use alloc::vec::Vec;
use briny::traits::InteriorImmutable;
use lazy_simd::{scalar::Primitive, simd::SimdElement};
use tensor_optim::TensorOps;

#[cfg(feature = "dyntensor")]
use lazy_simd::simd::backend::NonAssociativeSimd;

#[cfg(not(feature = "dyntensor"))]
use lazy_simd::simd::backend::AlignedSimd;

/// The most flexible, fast, and secure Tensor available.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct VecTensor<T> {
    data: Vec<T>,
    shape: Vec<usize>,
}

#[cfg(feature = "dyntensor")]
impl<T> PartialEq<Tensor<T>> for VecTensor<T>
where
    T: SimdElement + Primitive + Default + PartialEq,
    [T; FLOAT_LANES]: NonAssociativeSimd<[T; FLOAT_LANES], T, FLOAT_LANES>,
{
    fn eq(&self, other: &Tensor<T>) -> bool {
        &self.into_tensor() == other
    }
}

#[cfg(not(feature = "dyntensor"))]
impl<T, const N: usize, const D: usize> PartialEq<Tensor<T, N, D>> for VecTensor<T>
where
    T: SimdElement + Primitive + Default + PartialEq,
    [T; FLOAT_LANES]: AlignedSimd<[T; FLOAT_LANES], T, FLOAT_LANES>,
{
    fn eq(&self, other: &Tensor<T, N, D>) -> bool {
        &self.into_tensor() == other
    }
}

#[cfg(feature = "dyntensor")]
impl<T> PartialEq<VecTensor<T>> for Tensor<T>
where
    T: SimdElement + Primitive + Default + PartialEq,
    [T; FLOAT_LANES]: NonAssociativeSimd<[T; FLOAT_LANES], T, FLOAT_LANES>,
{
    fn eq(&self, other: &VecTensor<T>) -> bool {
        self == &other.into_tensor()
    }
}

#[cfg(not(feature = "dyntensor"))]
impl<T, const N: usize, const D: usize> PartialEq<VecTensor<T>> for Tensor<T, N, D>
where
    T: SimdElement + Primitive + Default + PartialEq,
    [T; FLOAT_LANES]: AlignedSimd<[T; FLOAT_LANES], T, FLOAT_LANES>,
{
    fn eq(&self, other: &VecTensor<T>) -> bool {
        self == &other.into_tensor()
    }
}

unsafe impl<T> InteriorImmutable for VecTensor<T> where T: InteriorImmutable {}

impl<T> Default for VecTensor<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> TensorOps<T> for VecTensor<T> {
    fn data(&self) -> &[T] {
        &self.data
    }

    fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<T: Clone> TensorGrad<T> for VecTensor<T> {
    fn len(&self) -> usize {
        self.data().len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn new_with_data(data: &[T], shape: &[usize]) -> Self {
        Self::with_data(shape, data)
    }

    fn zeros_like(&self) -> Self {
        Self::new()
    }
}

impl<T> VecTensor<T> {
    /// Creates an empty tensor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            data: Vec::new(),
            shape: Vec::new(),
        }
    }

    /// Creates a tensor with data.
    ///
    /// This constructor should be used if a defined shape or actual
    /// data is required.
    pub fn with_data(shape: &[usize], data: &[T]) -> Self
    where
        T: Clone,
    {
        Self {
            data: Vec::from(data),
            shape: Vec::from(shape),
        }
    }

    /// Creates a `VecTensor` from a `Tensor`.
    ///
    /// The same data and shape is preserved.
    #[must_use]
    pub fn from_tensor<U>(tensor: &U) -> Self
    where
        U: TensorOps<T>,
        T: Clone,
    {
        Self::with_data(tensor.shape(), tensor.data())
    }

    /// Construct a `VecTensor<T>` from a `Tensor<T>`.
    #[must_use]
    #[cfg(feature = "dyntensor")]
    pub fn into_tensor(&self) -> Tensor<T>
    where
        T: SimdElement + Primitive + Default,
        [T; FLOAT_LANES]: NonAssociativeSimd<[T; FLOAT_LANES], T, FLOAT_LANES>,
    {
        Tensor::<T>::new(self.shape(), self.data())
    }

    /// Construct a `VecTensor<T>` from a `Tensor<T, N, D>`.
    ///
    /// # Panics
    ///
    /// As long as the compiler can assume the size of the output tensor, this
    /// function shouldn't panic. However, when specified with user-determined
    /// ranks and lengths, the function will panic.
    ///
    /// - size must be the same for output tensor and `self`
    /// - rank must be the same for output tensor and `self`
    #[must_use]
    #[cfg(not(feature = "dyntensor"))]
    pub fn into_tensor<const N: usize, const D: usize>(&self) -> Tensor<T, N, D>
    where
        T: SimdElement + Primitive + Default,
        [T; FLOAT_LANES]: AlignedSimd<[T; FLOAT_LANES], T, FLOAT_LANES>,
    {
        Tensor::<T, N, D>::new(
            self.shape()
                .try_into()
                .expect("shape mismatch on `VecTensor<T>` for given `Tensor<T, N, D>`"),
            self.data()
                .try_into()
                .expect("data length mismatch on `VecTensor<T>` for given `Tensor<T, N, D>`"),
        )
    }
}
