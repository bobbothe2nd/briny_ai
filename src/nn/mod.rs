//! Tedious manual tensor operations.
//!
//! The alternative is to use high level abstractions, which might not be as efficient.
//! Though, manually implementing everything isn't portable; some targets may have `Tensor<T, N, D>`,
//! others have `Tensor<T>`. Some targets might have `T=f64`m others might have `T=f32`.
//!
//! Changing the features will completely change the API and wil probably not compile.
//! However, the abstract API is always the same, so it is much more portable and will
//! always compile for correct uses - whether features are removed, added, etc., the
//! abstraction is abstracted the same way every time.

pub use lazy_simd::scalar::{ArrayOf, Flatten, Primitive, SubsetOf, SupersetOf};

pub mod io;
pub mod ops;
pub mod tensors;

/// Conversions between primitive floats;
pub trait IntermediateFp {
    /// Convert to `f32`.
    fn into_f32(self) -> f32;
    /// Convert from `f32`.
    fn from_f32(x: f32) -> Self;

    /// Convert to `f64`.
    fn into_f64(self) -> f64;
    /// Convert from `f64`.
    fn from_f64(x: f64) -> Self;
}

impl IntermediateFp for f32 {
    fn from_f32(x: Self) -> Self {
        x
    }

    #[allow(clippy::cast_possible_truncation)]
    fn from_f64(x: f64) -> Self {
        x as Self
    }

    fn into_f32(self) -> Self {
        self
    }

    fn into_f64(self) -> f64 {
        f64::from(self)
    }
}

impl IntermediateFp for f64 {
    fn from_f32(x: f32) -> Self {
        Self::from(x)
    }

    fn from_f64(x: Self) -> Self {
        x
    }

    #[allow(clippy::cast_possible_truncation)]
    fn into_f32(self) -> f32 {
        self as f32
    }

    fn into_f64(self) -> Self {
        self
    }
}

type TensorFloatInner = f32;

/// The float used in tensors (`f32`/`fp32`).
pub type TensorFloat = TensorFloatInner;

/// The amount of [`TensorFloat`]s in a SIMD register on the target.
///
/// Equal to [`lazy_simd::MAX_SIMD_SINGLE_PRECISION_LANES`]
pub const FLOAT_LANES: usize = lazy_simd::MAX_SIMD_SIZE / size_of::<TensorFloat>();
