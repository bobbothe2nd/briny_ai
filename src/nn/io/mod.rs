//! [[UNSTABLE]] Robust saving/loading of model weights.
//!
//! # `Briny Portable Accelerated Tensor` Model Serialization Format
//!
//! This module provides minimal, dependency-free utilities for saving and loading tensors in a custom binary format.
//! It's intended for saving small models in a compact, fast, and portable format for `briny_ai` or similar systems.
//!
//! # Format Overview
//!
//! A `.bpat` file stores one or more tensors in the following layout:
//!
//! ```text
//! ┌──────────────┬─────────────────────────┬────────────────────┐
//! │ Header       │ Tensor N .. N+1 .. N+2 ..  │ Checksum           │
//! ├──────────────┼─────────────────────────┼────────────────────┤
//! │ `bpat`       │ u64: ndim               │ u32: file checksum │
//! │ u8: count    │ [u64; ndim] shape       │                    │
//! │              │ [f64; prod(shape)] data │                    │
//! │              │ u32: checksum           │                    │
//! └──────────────┴─────────────────────────┴────────────────────┘
//! ```
//!
//! ## Header
//!
//! - `bpat` magic (4 bytes): ensures file is recognized
//! - `u8` tensor count: number of tensors to read
//!
//! ## Tensor Encoding
//!
//! For each tensor:
//! - `ndim` (`u64`): number of dimensions
//! - `shape` (`u64 * ndim`): each dimension size
//! - `data` (`f64 * prod(shape)`): flattened, row-major tensor data
//!
//! # Design Principles
//!
//! - Fully self-contained
//! - No compression or encryption (for simplicity and speed)
//! - Suitable for deterministic, reproducible serialization
//! - Works on little-endian platforms
//!
//! # Limitations
//!
//! - Assumes `f64` element type
//! - Maximum 255 tensors per file (due to `u8` count limit)
//! - No per-tensor metadata (names, dtypes, etc.)

#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::nn::tensors::TensorGrad;
use crate::nn::IntermediateFp;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use briny::traits::Pod;
#[cfg(feature = "std")]
use std::fs::File;
use tensor_optim::TensorOps;

#[cfg(feature = "alloc")]
use crate::nn::tensors::VecTensor;
#[cfg(feature = "std")]
use std::io::{BufReader, Read};

#[cfg(feature = "std")]
mod versions;

/// The original BPAT header.
///
/// Used on `briny_ai` `v0.1.0`-`v0.2.2`.
///
/// # Format
///
/// This version looks like this:
///
/// ```text
/// ┌──────────────┬────────────────────────────┐
/// │ Header       │ Tensor N .. N+1 .. N+2 ..  │
/// ├──────────────┼────────────────────────────┤
/// │ `bpat`       │ u64: ndim                  │
/// │ u8: count    │ [u64; ndim] shape          │
/// │              │ [f64; prod(shape)] data    │
/// └──────────────┴────────────────────────────┘
/// ```
pub const BPAT_MAGIC_V0: [u8; 4] = *b"bpat";

/// The first BPAT header with checksums.
///
/// Created on `briny_ai` `v0.3.0`.
///
/// # Format
///
/// This version looks like this:
///
/// ```text
/// ┌──────────────┬────────────────────────────┬────────────────────┐
/// │ Header       │ Tensor N .. N+1 .. N+2 ..  │ Checksum           │
/// ├──────────────┼────────────────────────────┼────────────────────┤
/// │ `BPATv1\0\0` │ u64: ndim                  │ u32: file checksum │
/// │ u8: count    │ [u64; ndim] shape          │                    │
/// │              │ [f64; prod(shape)] data    │                    │
/// │              │ u32: checksum              │                    │
/// └──────────────┴────────────────────────────┴────────────────────┘
/// ```
pub const BPAT_MAGIC_V1: [u8; 8] = *b"BPATv1\0\0";

/// The most compact BPAT header.
///
/// Created on `briny_ai` `v0.6.0`.
///
/// # Format
///
/// This version looks like this:
///
/// ```text
/// ┌──────────────┬────────────────────────────┬────────────────────┐
/// │ Header       │ Tensor N .. N+1 .. N+2 ..  │ Checksum           │
/// ├──────────────┼────────────────────────────┼────────────────────┤
/// │ `BPATv1m\0`  │ u32: ndim                  │ u32: file checksum │
/// │ u8: count    │ [u32; ndim] shape          │                    │
/// │              │ [f32; prod(shape)] data    │                    │
/// │              │ u32: checksum              │                    │
/// └──────────────┴────────────────────────────┴────────────────────┘
/// ```
pub const BPAT_MAGIC_V1_MICRO: [u8; 8] = *b"BPATv1m\0";

/// An enumerated header dispatching BPAT formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BpatHeader {
    /// A fast and compact format.
    BpatV0,

    /// A format that guarantees data integrity.
    BpatV1,

    /// A short and efficient format.
    BpatV1M,
}

#[allow(clippy::derivable_impls)]
impl Default for BpatHeader {
    fn default() -> Self {
        #[cfg(feature = "std")]
        {
            Self::BpatV1
        }
        #[cfg(not(feature = "std"))]
        {
            Self::BpatV0
        }
    }
}

/// Analysis of an error during tensor serialization
#[derive(Debug)]
pub struct SerialTensorError {
    /// The type of error.
    pub kind: SerialTensorErrorKind,
    /// The attached message.
    pub msg: &'static str,
}

/// The type of error in tensor serialization.
#[derive(Debug)]
pub enum SerialTensorErrorKind {
    /// Bad integrity: e.g. mismatched checksums.
    IntegrityUnverified,

    /// Invalid data: e.g. shape product != data len
    InvalidData,

    /// Invalid header: e.g. not valid `bpat` signature
    InvalidHeader,

    /// Invalid path: e.g. no file exists
    InvalidPath,
}

/// Saves the given tensors to a file.
///
/// # Errors
///
/// Failure conditions are as follows:
///
/// - The `std` feature is disabled
/// - The tensors are invalid
/// - The file path is invalid
#[cfg_attr(not(feature = "std"), allow(clippy::missing_const_for_fn))]
pub fn save_tensors<T: TensorGrad<U> + TensorOps<U>, U: Pod + Copy + IntermediateFp>(
    path: &str,
    tensors: &[T],
    header: BpatHeader,
) -> Result<(), SerialTensorError> {
    #[cfg(feature = "std")]
    {
        match header {
            BpatHeader::BpatV0 => versions::v0::save_tensors(path, tensors),
            BpatHeader::BpatV1 => versions::v1::save_tensors(path, tensors),
            BpatHeader::BpatV1M => versions::v1m::save_tensors(path, tensors),
        }
    }
    #[cfg(not(feature = "std"))]
    {
        Err(SerialTensorError {
            kind: SerialTensorErrorKind::InvalidPath,
            msg: "I/O disabled",
        })
    }
}

/// Loads tensors from a file.
///
/// # Errors
///
/// Returns a [`SerialTensorError`] on failure to load tensors.
///
/// - The file exists
/// - The file is in `bpat` format
/// - The file is not corrupted
/// - The `std` feature is enabled
#[cfg(feature = "alloc")]
#[cfg_attr(not(feature = "std"), allow(clippy::missing_const_for_fn))]
pub fn load_tensors<T: Copy + Pod + IntermediateFp>(
    path: &str,
) -> Result<Vec<VecTensor<T>>, SerialTensorError> {
    #[cfg(feature = "std")]
    {
        let mut file = BufReader::new(File::open(path).map_err(|_| SerialTensorError {
            kind: SerialTensorErrorKind::InvalidPath,
            msg: "no such file exists",
        })?);
        let mut file_start = [0; 8];
        file.read_exact(&mut file_start)
            .map_err(|_| SerialTensorError {
                kind: SerialTensorErrorKind::InvalidHeader,
                msg: "header not found",
            })?;
        let header = if file_start == BPAT_MAGIC_V1 {
            BpatHeader::BpatV1
        } else if file_start.starts_with(&BPAT_MAGIC_V0) {
            BpatHeader::BpatV0
        } else if file_start.starts_with(&BPAT_MAGIC_V1_MICRO) {
            BpatHeader::BpatV1M
        } else {
            return Err(SerialTensorError {
                kind: SerialTensorErrorKind::InvalidHeader,
                msg: "invalid magic header",
            });
        };
        match header {
            BpatHeader::BpatV0 => versions::v0::load_tensors(path),
            BpatHeader::BpatV1 => versions::v1::load_tensors(path),
            BpatHeader::BpatV1M => versions::v1m::load_tensors(path),
        }
    }
    #[cfg(not(feature = "std"))]
    {
        Err(SerialTensorError {
            kind: SerialTensorErrorKind::InvalidPath,
            msg: "I/O disabled",
        })
    }
}
