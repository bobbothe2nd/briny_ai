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

pub mod ops;
pub mod tensors;
// pub mod io;

#[cfg(feature = "f64")]
type TensorFloatInner = f64;
#[cfg(not(feature = "f64"))]
type TensorFloatInner = f32;

/// The float used in tensors.
///
/// - `f64` on 64-bit machines
/// - `f32` on 32-bit machines
/// - `f32` on 16-bit machines
pub type TensorFloat = TensorFloatInner;

/// The amount of [`TensorFloat`]s in a SIMD register on the target.
pub const FLOAT_LANES: usize = if cfg!(target_pointer_width = "64") {
    lazy_simd::MAX_SIMD_DOUBLE_PRECISION_LANES * 2 // should be 1/2, but doesn't compile if not
} else {
    lazy_simd::MAX_SIMD_SINGLE_PRECISION_LANES // equal to the above
};

pub trait IntermediateFp {
    fn into_f64(self) -> f64;
    fn from_f64(val: f64) -> Self;
    fn into_f32(self) -> f32;
    fn from_f32(val: f32) -> Self;
}

impl IntermediateFp for f32 {
    fn from_f32(val: Self) -> Self {
        val
    }

    fn from_f64(val: f64) -> Self {
        val as Self
    }

    fn into_f32(self) -> Self {
        self
    }

    fn into_f64(self) -> f64 {
        f64::from(self)
    }
}

impl IntermediateFp for f64 {
    fn from_f32(val: f32) -> Self {
        Self::from(val)
    }

    fn from_f64(val: Self) -> Self {
        val
    }

    fn into_f32(self) -> f32 {
        self as f32
    }

    fn into_f64(self) -> Self {
        self
    }
}
