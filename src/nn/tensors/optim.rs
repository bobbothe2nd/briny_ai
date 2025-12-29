//! Bindings to `tensor_optim` as the default tensor backend.
//!
//! Much of the logic in this module is just boilerplate code to get the
//! low-level `tensor_optim` crate to behave on more advanced systems.
//! Little here is really logic, just abstractions over what is provided
//! in `tensor_optim`.

#[cfg(target_has_atomic = "16")]
use core::sync::atomic::AtomicI16;
#[cfg(target_has_atomic = "32")]
use core::sync::atomic::AtomicI32;
#[cfg(target_has_atomic = "64")]
use core::sync::atomic::AtomicI64;
#[cfg(target_has_atomic = "8")]
use core::sync::atomic::AtomicI8;
#[cfg(target_has_atomic = "ptr")]
use core::sync::atomic::AtomicIsize;

#[cfg(feature = "alloc")]
use crate::nn::tensors::VecTensor;

use core::{
    mem::{ManuallyDrop, MaybeUninit},
    num::Wrapping,
    ops::{Add, Div, Mul, Sub},
    sync::atomic::{AtomicBool, AtomicU16, AtomicU32, AtomicU64, AtomicU8, AtomicUsize},
};

use briny::traits::Writable;
use tensor_optim::TensorOps;

use crate::nn::FLOAT_LANES;

use lazy_simd::{scalar::Primitive, simd::SimdElement};

#[cfg(not(feature = "dyntensor"))]
use lazy_simd::simd::backend::AlignedSimd;
#[cfg(feature = "dyntensor")]
use lazy_simd::simd::backend::NonAssociativeSimd;

#[cfg(not(feature = "dyntensor"))]
use briny::traits::{RawConvert, StableLayout};

/// A trait to flatten nested types.
pub trait Flatten<const N: usize> {
    /// The flattened type (from nested).
    type Flattened;

    /// Flattens `self` if nested, otherwise it's already flat.
    fn flatten(&self) -> Self::Flattened;
}

/// A marker for types that are already flattened.
pub trait Unflatten {}

macro_rules! transparent_impl {
    ($tr:ident, $($ty:ident),*) => {
        $(
            impl<T: $tr> $tr for $ty<T> {}
        )*
    };
}

transparent_impl!(Unflatten, MaybeUninit, ManuallyDrop, Wrapping);

impl Unflatten for bool {}

impl Unflatten for u8 {}
impl Unflatten for u16 {}
impl Unflatten for u32 {}
impl Unflatten for u64 {}
impl Unflatten for u128 {}
impl Unflatten for usize {}

impl Unflatten for i8 {}
impl Unflatten for i16 {}
impl Unflatten for i32 {}
impl Unflatten for i64 {}
impl Unflatten for i128 {}
impl Unflatten for isize {}

impl Unflatten for f32 {}
impl Unflatten for f64 {}

#[cfg(target_has_atomic = "8")]
impl Unflatten for AtomicBool {}

#[cfg(target_has_atomic = "8")]
impl Unflatten for AtomicU8 {}
#[cfg(target_has_atomic = "16")]
impl Unflatten for AtomicU16 {}
#[cfg(target_has_atomic = "32")]
impl Unflatten for AtomicU32 {}
#[cfg(target_has_atomic = "64")]
impl Unflatten for AtomicU64 {}
#[cfg(target_has_atomic = "ptr")]
impl Unflatten for AtomicUsize {}

#[cfg(target_has_atomic = "8")]
impl Unflatten for AtomicI8 {}
#[cfg(target_has_atomic = "16")]
impl Unflatten for AtomicI16 {}
#[cfg(target_has_atomic = "32")]
impl Unflatten for AtomicI32 {}
#[cfg(target_has_atomic = "64")]
impl Unflatten for AtomicI64 {}
#[cfg(target_has_atomic = "ptr")]
impl Unflatten for AtomicIsize {}

impl<T: Default + Copy + Unflatten, const N: usize> Flatten<N> for [T; N] {
    type Flattened = [T; N];

    fn flatten(&self) -> Self::Flattened {
        *self
    }
}

impl<T: Default + Copy + Unflatten, const X: usize, const Y: usize, const N: usize> Flatten<N>
    for [[T; X]; Y]
{
    type Flattened = [T; N];

    fn flatten(&self) -> Self::Flattened {
        const {
            assert!(N == X * Y, "flattenning to an invalid length");
        }

        let mut arr = [T::default(); N];
        for (i, chunk) in self.iter().enumerate() {
            let start = i * X;
            let end = start + X;
            arr[start..end].copy_from_slice(chunk);
        }
        arr
    }
}

impl<
        T: Default + Copy + Unflatten,
        const X: usize,
        const Y: usize,
        const Z: usize,
        const N: usize,
    > Flatten<N> for [[[T; X]; Y]; Z]
{
    type Flattened = [T; N];

    fn flatten(&self) -> Self::Flattened {
        const {
            assert!(N == X * Y * Z, "flattenning to an invalid length");
        }

        let mut arr = [T::default(); N];
        for chunk_x in self {
            for (iy, chunk_y) in chunk_x.iter().enumerate() {
                let start = iy * X;
                let end = start + X;
                arr[start..end].copy_from_slice(chunk_y);
            }
        }
        arr
    }
}

/// A trait for statically sized nested types.
pub trait StaticShape<const N: usize> {
    /// Returns the shape of `self` as a slice.
    fn sliced_shape(&self) -> &[usize; N];

    /// Calculates the length based off the shape.
    fn len(&self) -> usize {
        self.sliced_shape().iter().product()
    }

    /// Checks if the length is 0 conditionally.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: Default + Copy + Unflatten, const N: usize> StaticShape<1> for [T; N] {
    fn sliced_shape(&self) -> &[usize; 1] {
        &[N]
    }
}

impl<T: Default + Copy + Unflatten, const X: usize, const Y: usize> StaticShape<2> for [[T; X]; Y] {
    fn sliced_shape(&self) -> &[usize; 2] {
        &[X, Y]
    }
}

impl<T: Default + Copy + Unflatten, const X: usize, const Y: usize, const Z: usize> StaticShape<3>
    for [[[T; X]; Y]; Z]
{
    fn sliced_shape(&self) -> &[usize; 3] {
        &[X, Y, Z]
    }
}

/// A minimal trait that represents a tensor-like structure supporting gradients.
pub trait TensorGrad<T>: Clone {
    /// Returns the number of elements in the tensor.
    fn len(&self) -> usize;

    /// Returns whether the tensor is empty or not.
    fn is_empty(&self) -> bool;

    /// Constructs a new tensor with the provided shape and data.
    #[must_use]
    fn new_with_data(data: &[T], shape: &[usize]) -> Self;

    /// Creates a zero-filled tensor of the same shape.
    #[must_use]
    fn zeros_like(&self) -> Self;

    /// Converts a regular tensor into a [`VecTensor`].
    #[cfg(feature = "alloc")]
    fn as_vectensor(&self) -> VecTensor<T>
    where
        T: Clone,
        Self: TensorOps<T>,
    {
        VecTensor::from_tensor(self)
    }
}

#[cfg(feature = "dyntensor")]
impl<T> TensorGrad<T> for DynTensor<T, { FLOAT_LANES }>
where
    T: SimdElement + Primitive + Default,
    [T; FLOAT_LANES]: NonAssociativeSimd<[T; FLOAT_LANES], T, FLOAT_LANES>,
{
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize {
        self.data().len()
    }

    fn new_with_data(data: &[T], shape: &[usize]) -> Self {
        Self::with_data(shape, data)
    }

    fn zeros_like(&self) -> Self {
        Self::new(&[self.len()])
    }
}

#[cfg(not(feature = "dyntensor"))]
impl<T, const N: usize, const D: usize> TensorGrad<T> for ArrTensor<T, N, D, { FLOAT_LANES }>
where
    T: SimdElement + Primitive + Default,
    [T; FLOAT_LANES]: AlignedSimd<[T; FLOAT_LANES], T, { FLOAT_LANES }>,
{
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize {
        self.data().len()
    }

    fn new_with_data(data: &[T], shape: &[usize]) -> Self {
        Self::with_data(shape.try_into().unwrap(), data.try_into().unwrap())
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

use crate::nn::TensorFloat;

#[cfg(feature = "dyntensor")]
impl<T> Add<Self> for Tensor<T>
where
    T: SimdElement + Primitive + Default,
    [T; FLOAT_LANES]: NonAssociativeSimd<[T; FLOAT_LANES], T, { FLOAT_LANES }>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

#[cfg(not(feature = "dyntensor"))]
impl<T, const N: usize, const D: usize> Add<Self> for Tensor<T, N, D>
where
    T: SimdElement + Primitive + Default,
    [T; FLOAT_LANES]: AlignedSimd<[T; FLOAT_LANES], T, { FLOAT_LANES }>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

#[cfg(feature = "dyntensor")]
impl<T> Add<T> for Tensor<T>
where
    T: SimdElement + Primitive + Default,
    [T; FLOAT_LANES]: NonAssociativeSimd<[T; FLOAT_LANES], T, { FLOAT_LANES }>,
{
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        Self(self.0 + rhs)
    }
}

#[cfg(not(feature = "dyntensor"))]
impl<T, const N: usize, const D: usize> Add<T> for Tensor<T, N, D>
where
    T: SimdElement + Primitive + Default,
    [T; FLOAT_LANES]: AlignedSimd<[T; FLOAT_LANES], T, { FLOAT_LANES }>,
{
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        Self(self.0 + rhs)
    }
}

#[cfg(feature = "dyntensor")]
impl<T> Sub<Self> for Tensor<T>
where
    T: SimdElement + Primitive + Default,
    [T; FLOAT_LANES]: NonAssociativeSimd<[T; FLOAT_LANES], T, { FLOAT_LANES }>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

#[cfg(not(feature = "dyntensor"))]
impl<T, const N: usize, const D: usize> Sub<Self> for Tensor<T, N, D>
where
    T: SimdElement + Primitive + Default,
    [T; FLOAT_LANES]: AlignedSimd<[T; FLOAT_LANES], T, { FLOAT_LANES }>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

#[cfg(feature = "dyntensor")]
impl<T> Sub<T> for Tensor<T>
where
    T: SimdElement + Primitive + Default,
    [T; FLOAT_LANES]: NonAssociativeSimd<[T; FLOAT_LANES], T, { FLOAT_LANES }>,
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        Self(self.0 - rhs)
    }
}

#[cfg(not(feature = "dyntensor"))]
impl<T, const N: usize, const D: usize> Sub<T> for Tensor<T, N, D>
where
    T: SimdElement + Primitive + Default,
    [T; FLOAT_LANES]: AlignedSimd<[T; FLOAT_LANES], T, { FLOAT_LANES }>,
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        Self(self.0 - rhs)
    }
}

//
#[cfg(feature = "dyntensor")]
impl<T> Mul<Self> for Tensor<T>
where
    T: SimdElement + Primitive + Default,
    [T; FLOAT_LANES]: NonAssociativeSimd<[T; FLOAT_LANES], T, { FLOAT_LANES }>,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

#[cfg(not(feature = "dyntensor"))]
impl<T, const N: usize, const D: usize> Mul<Self> for Tensor<T, N, D>
where
    T: SimdElement + Primitive + Default,
    [T; FLOAT_LANES]: AlignedSimd<[T; FLOAT_LANES], T, { FLOAT_LANES }>,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

#[cfg(feature = "dyntensor")]
impl<T> Mul<T> for Tensor<T>
where
    T: SimdElement + Primitive + Default,
    [T; FLOAT_LANES]: NonAssociativeSimd<[T; FLOAT_LANES], T, { FLOAT_LANES }>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self(self.0 * rhs)
    }
}

#[cfg(not(feature = "dyntensor"))]
impl<T, const N: usize, const D: usize> Mul<T> for Tensor<T, N, D>
where
    T: SimdElement + Primitive + Default,
    [T; FLOAT_LANES]: AlignedSimd<[T; FLOAT_LANES], T, { FLOAT_LANES }>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self(self.0 * rhs)
    }
}

#[cfg(feature = "dyntensor")]
impl<T> Div<Self> for Tensor<T>
where
    T: SimdElement + Primitive + Default,
    [T; FLOAT_LANES]: NonAssociativeSimd<[T; FLOAT_LANES], T, { FLOAT_LANES }>,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0 / rhs.0)
    }
}

#[cfg(not(feature = "dyntensor"))]
impl<T, const N: usize, const D: usize> Div<Self> for Tensor<T, N, D>
where
    T: SimdElement + Primitive + Default,
    [T; FLOAT_LANES]: AlignedSimd<[T; FLOAT_LANES], T, { FLOAT_LANES }>,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0 / rhs.0)
    }
}

#[cfg(feature = "dyntensor")]
impl<T> Div<T> for Tensor<T>
where
    T: SimdElement + Primitive + Default,
    [T; FLOAT_LANES]: NonAssociativeSimd<[T; FLOAT_LANES], T, { FLOAT_LANES }>,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        Self(self.0 / rhs)
    }
}

#[cfg(not(feature = "dyntensor"))]
impl<T, const N: usize, const D: usize> Div<T> for Tensor<T, N, D>
where
    T: SimdElement + Primitive + Default,
    [T; FLOAT_LANES]: AlignedSimd<[T; FLOAT_LANES], T, { FLOAT_LANES }>,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        Self(self.0 / rhs)
    }
}

/// A simple tensor with decent flexibility.
///
/// It's entirely heap-allocated, so it won't typically
/// overflow the stack like the no-`alloc` tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg(feature = "dyntensor")]
#[repr(transparent)]
pub struct Tensor<T>(DynTensor<T, { FLOAT_LANES }>)
where
    T: SimdElement + Primitive,
    [T; FLOAT_LANES]: NonAssociativeSimd<[T; FLOAT_LANES], T, FLOAT_LANES>;

#[cfg(feature = "dyntensor")]
unsafe impl<T> Writable for Tensor<T>
where
    T: SimdElement + Primitive + Writable,
    [T; FLOAT_LANES]: NonAssociativeSimd<[T; FLOAT_LANES], T, FLOAT_LANES>,
{
}

#[cfg(feature = "dyntensor")]
impl<T> TensorOps<T> for Tensor<T>
where
    T: SimdElement + Primitive,
    [T; FLOAT_LANES]: NonAssociativeSimd<[T; FLOAT_LANES], T, FLOAT_LANES>,
{
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
impl<T> Tensor<T>
where
    T: SimdElement + Primitive + Default,
    [T; FLOAT_LANES]: NonAssociativeSimd<[T; FLOAT_LANES], T, FLOAT_LANES>,
{
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
impl<T> Tensor<T>
where
    T: SimdElement + Primitive + Default,
    [T; FLOAT_LANES]: NonAssociativeSimd<[T; FLOAT_LANES], T, FLOAT_LANES>,
{
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

    /// Creates a zeroed tensor with the given shape.
    #[must_use]
    pub fn zeros(shape: &[usize]) -> Self {
        Self(DynTensor::new(shape))
    }

    /// Maps each element of the data.
    #[must_use]
    pub fn map<U, F>(&self, f: F) -> Tensor<U>
    where
        F: Fn(&T) -> U,
        U: SimdElement + Primitive + Default,
        [U; FLOAT_LANES]: NonAssociativeSimd<[U; FLOAT_LANES], U, { FLOAT_LANES }>,
    {
        Tensor(self.0.map(f))
    }

    /// Maps each element of the data of `self` and `other`.
    #[must_use]
    pub fn zip_map<U, V, F>(&self, other: &Tensor<U>, f: F) -> Tensor<V>
    where
        F: Fn(&T, &U) -> V,
        U: SimdElement + Primitive + Default,
        [U; FLOAT_LANES]: NonAssociativeSimd<[U; FLOAT_LANES], U, { FLOAT_LANES }>,
        V: SimdElement + Primitive + Default,
        [V; FLOAT_LANES]: NonAssociativeSimd<[V; FLOAT_LANES], V, { FLOAT_LANES }>,
    {
        Tensor(self.0.zip_map(&other.0, f))
    }
}

#[cfg(feature = "dyntensor")]
impl Tensor<TensorFloat>
where
    TensorFloat: SimdElement + Primitive,
    [TensorFloat; FLOAT_LANES]:
        NonAssociativeSimd<[TensorFloat; FLOAT_LANES], TensorFloat, FLOAT_LANES>,
{
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
    pub fn matmul(&self, rhs: &Self) -> Self {
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
        self.0.simd_matmul(&rhs.0, &mut out);
        Self(out)
    }
}

#[cfg(feature = "dyntensor")]
impl<T> TensorGrad<T> for Tensor<T>
where
    T: SimdElement + Primitive + Default,
    [T; FLOAT_LANES]: NonAssociativeSimd<[T; FLOAT_LANES], T, FLOAT_LANES>,
{
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize {
        self.data().len()
    }

    fn new_with_data(data: &[T], shape: &[usize]) -> Self {
        Self::new(shape, data)
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
#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct Tensor<T, const N: usize, const D: usize>(ArrTensor<T, N, D, { FLOAT_LANES }>)
where
    T: SimdElement + Primitive,
    [T; FLOAT_LANES]: AlignedSimd<[T; FLOAT_LANES], T, { FLOAT_LANES }>;

#[cfg(not(feature = "dyntensor"))]
unsafe impl<T, const N: usize, const D: usize> StableLayout for Tensor<T, N, D>
where
    T: SimdElement + Primitive + StableLayout,
    [T; FLOAT_LANES]: AlignedSimd<[T; FLOAT_LANES], T, { FLOAT_LANES }>,
{
}

#[cfg(not(feature = "dyntensor"))]
unsafe impl<T, const N: usize, const D: usize> RawConvert for Tensor<T, N, D>
where
    T: SimdElement + Primitive + RawConvert,
    [T; FLOAT_LANES]: AlignedSimd<[T; FLOAT_LANES], T, { FLOAT_LANES }>,
{
}

#[cfg(not(feature = "dyntensor"))]
unsafe impl<T, const N: usize, const D: usize> Writable for Tensor<T, N, D>
where
    T: SimdElement + Primitive + Writable,
    [T; FLOAT_LANES]: AlignedSimd<[T; FLOAT_LANES], T, { FLOAT_LANES }>,
{
}

#[cfg(not(feature = "dyntensor"))]
impl<T, const N: usize, const D: usize> tensor_optim::TensorOps<T> for Tensor<T, N, D>
where
    T: SimdElement + Primitive,
    [T; FLOAT_LANES]: AlignedSimd<[T; FLOAT_LANES], T, { FLOAT_LANES }>,
{
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
impl<T, const N: usize, const D: usize> tensor_optim::ConstTensorOps<T, N, D> for Tensor<T, N, D>
where
    T: SimdElement + Primitive + Default,
    [T; FLOAT_LANES]: AlignedSimd<[T; FLOAT_LANES], T, { FLOAT_LANES }>,
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
impl<T, const N: usize> Tensor<T, N, 2>
where
    T: SimdElement + Primitive + Default,
    [T; FLOAT_LANES]: AlignedSimd<[T; FLOAT_LANES], T, { FLOAT_LANES }>,
{
    /// Parses a tensor from neted arrays.
    #[must_use]
    pub fn from_arr<F: Flatten<N, Flattened = [T; N]> + StaticShape<2>>(arr: &F) -> Self {
        Self::new(arr.sliced_shape(), &arr.flatten())
    }
}

#[cfg(not(feature = "dyntensor"))]
impl<T, const N: usize, const D: usize> Tensor<T, N, D>
where
    T: SimdElement + Primitive + Default,
    [T; FLOAT_LANES]: AlignedSimd<[T; FLOAT_LANES], T, { FLOAT_LANES }>,
{
    /// Constructs a new `Tensor<T>` with the given shape and data.
    ///
    /// # Panics
    ///
    /// The product of each entry in `shape` must be equal to the length of `data` or the constructor will panic at runtime.
    #[must_use]
    pub fn new(shape: &[usize; D], data: &[T; N]) -> Self {
        Self(ArrTensor::with_data(*shape, *data))
    }

    /// Constructs a new `Tensor<T>` from the provided Tensor.
    #[must_use]
    pub const fn from_arrtensor(tensor: ArrTensor<T, N, D, { FLOAT_LANES }>) -> Self {
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

    /// Creates a zeroed tensor with the given shape.
    #[must_use]
    pub fn zeros(shape: &[usize; D]) -> Self
    where
        T: Copy,
    {
        Self(ArrTensor::new(*shape))
    }

    #[must_use]
    pub fn map<U, F>(&self, f: F) -> Tensor<U, N, D>
    where
        F: FnMut(T) -> U,
        U: SimdElement + Primitive + Default,
        [U; FLOAT_LANES]: AlignedSimd<[U; FLOAT_LANES], U, { FLOAT_LANES }>,
    {
        Tensor(self.0.map(f))
    }

    #[must_use]
    pub fn zip_map<U, V, F>(&self, other: &Tensor<U, N, D>, f: F) -> Tensor<V, N, D>
    where
        F: FnMut(T, U) -> V,
        U: SimdElement + Primitive + Default,
        [U; FLOAT_LANES]: AlignedSimd<[U; FLOAT_LANES], U, { FLOAT_LANES }>,
        V: SimdElement + Primitive + Default,
        [V; FLOAT_LANES]: AlignedSimd<[V; FLOAT_LANES], V, { FLOAT_LANES }>,
    {
        Tensor(self.0.zip_map(&other.0, f))
    }
}

#[cfg(not(feature = "dyntensor"))]
impl<const N: usize, const D: usize> Tensor<TensorFloat, N, D> {
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
    pub fn matmul<const M: usize, const K: usize>(
        &self,
        rhs: &Tensor<TensorFloat, M, D>,
    ) -> Tensor<TensorFloat, K, D> {
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

        debug_assert_eq!(
            self_batch_dims, rhs_batch_dims,
            "batch dimensions must match for matmul"
        );

        // compute expected output shape
        let mut expected_out_shape = [0usize; D];
        expected_out_shape[..D - 2].copy_from_slice(self_batch_dims);
        expected_out_shape[D - 2] = m;
        expected_out_shape[D - 1] = n;

        let mut out = ArrTensor::<TensorFloat, K, D>::new(expected_out_shape);
        self.0.simd_matmul(&rhs.0, &mut out); // call the low-level kernel
        Tensor::<TensorFloat, K, D>::from_arrtensor(out)
    }
}

#[cfg(not(feature = "dyntensor"))]
impl<T, const N: usize, const D: usize> TensorGrad<T> for Tensor<T, N, D>
where
    T: SimdElement + Primitive + Default,
    [T; FLOAT_LANES]: AlignedSimd<[T; FLOAT_LANES], T, { FLOAT_LANES }>,
{
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize {
        self.data().len()
    }

    fn new_with_data(data: &[T], shape: &[usize]) -> Self {
        Self::new(shape.try_into().unwrap(), data.try_into().unwrap())
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
    use crate::nn::tensors::IntoWithGrad;

    // Test basic WithGrad construction and zero initialization of gradients
    #[test]
    fn with_grad_basic_construction_and_mutation() {
        let data = [1.0f64, 2.0, 3.0, 4.0];
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
        let data = [5.0f64; 6];
        let tensor = Tensor::new(&[2, 3], &data);
        assert_eq!(tensor.len(), 6);
        assert!(!tensor.is_empty());

        let zeroed = tensor.zeros_like();
        assert_eq!(zeroed.len(), tensor.len());
        assert!(zeroed.data().iter().all(|&v| v == 0.0));

        // For an empty tensor
        let empty_tensor: Tensor<crate::nn::TensorFloat> = Tensor::new(&[0], &[]);
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
        let a_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a_shape = [2, 3];
        let a = Tensor::<TensorFloat, 6, 2>::new(&a_shape, &a_data);

        // B: 3x2
        let b_data = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let b_shape = [3, 2];
        let b = Tensor::<TensorFloat, 6, 2>::new(&b_shape, &b_data);

        // matmul
        let out = a.matmul::<6, 4>(&b);

        // expected output 2x2
        let expected = [58.0, 64.0, 139.0, 154.0];
        assert_eq!(out.shape(), [2, 2]);
        assert_eq!(out.data(), expected);
    }
}
