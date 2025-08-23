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

/// The float used in tensors.
///
/// - `f64` on 64-bit machines
/// - `f32` on 32-bit machines
/// - `f32` on 16-bit machines
#[cfg(target_pointer_width = "64")]
pub type TensorFloat = f64;
/// The float used in tensors.
///
/// - `f64` on 64-bit machines
/// - `f32` on 32-bit machines
/// - `f32` on 16-bit machines
#[cfg(not(target_pointer_width = "64"))]
pub type TensorFloat = f32;

pub mod backprop;
pub mod ops;
pub mod tensors;

#[cfg(feature = "std")]
pub mod diskio;
