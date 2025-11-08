//! Backend selection module.
//!
//! This module defines the available computation backends for the framework
//! and provides functions to set and get the current backend.
//!
//! # Supported Backends
//!
//! - `Cpu` — Pure Rust backend using standard CPU operations (default).
//! - `Wgpu` — GPU-accelerated backend using `wgpu` (if available).
//! - `Cuda` — Placeholder for future support (currently not functional).
//!
//! The backend is stored globally using an `AtomicU8`, enabling fast
//! switching between CPU and GPU modes at runtime.
//!
//! # Abstractions
//!
//! When using the model abstractions, and not the manual operations,
//! backends are automatically configured, but they can be overwritten
//! using this module.

use briny::traits::{InteriorImmutable, RawConvert, StableLayout, Unaligned};
use core::convert::TryFrom;
use core::sync::atomic::{AtomicU8, Ordering};

/// Enumeration of supported computation backends.
///
/// Currently only `Cpu` and `Wgpu` are implemented. `Cuda` is reserved
/// for future compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum Backend {
    /// Pure CPU-based backend (default).
    #[default]
    Cpu = 0,
    /// GPU-accelerated backend using `wgpu`.
    Wgpu,
    /// Placeholder for future CUDA support.
    Cuda,
}

unsafe impl StableLayout for Backend {}
unsafe impl RawConvert for Backend {}
unsafe impl Unaligned for Backend {}
unsafe impl InteriorImmutable for Backend {}

impl TryFrom<u8> for Backend {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Cpu),
            1 => Ok(Self::Wgpu),
            2 => Ok(Self::Cuda),
            _ => Err(()),
        }
    }
}

/// Internal global state for the active backend.
///
/// This uses relaxed memory ordering because the backend is only expected
/// to change rarely, and not in real-time concurrent compute contexts.
#[cfg(target_has_atomic = "8")]
static GLOBAL_DEFAULT_BACKEND: AtomicU8 = AtomicU8::new(Backend::Cpu as u8);

/// A mutable non-atomic unsynchronized backend state.
///
/// It is assumed that this will not be accessed concurrently. There really
/// should be a mutex or something around this.
#[cfg(not(target_has_atomic = "8"))]
static mut UNSAFE_GLOBAL_BACKEND: u8 = Backend::Cpu as u8;

/// Sets the active backend to use for tensor computation.
///
/// # Example
///
/// ```
/// use briny_ai::backend::{set_backend, Backend};
/// set_backend(Backend::Wgpu);
/// ```
pub fn set_backend(b: Backend) {
    #[cfg(not(target_has_atomic = "8"))]
    unsafe {
        UNSAFE_GLOBAL_BACKEND = b as u8;
    }
    #[cfg(target_has_atomic = "8")]
    GLOBAL_DEFAULT_BACKEND.store(b as u8, Ordering::Release);
}

/// Returns the currently active computation backend.
///
/// If the stored value is invalid, defaults to [`Backend::Cpu`].
///
/// # Example
///
/// ```
/// use briny_ai::backend::get_backend;
/// let backend = get_backend();
/// ```
pub fn get_backend() -> Backend {
    #[cfg(not(target_has_atomic = "8"))]
    {
        Backend::try_from(unsafe { UNSAFE_GLOBAL_BACKEND }).unwrap_or_default()
    }
    #[cfg(target_has_atomic = "8")]
    Backend::try_from(GLOBAL_DEFAULT_BACKEND.load(Ordering::Acquire)).unwrap_or_default()
}
