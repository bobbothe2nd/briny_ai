//! Robust saving/loading of model weights.
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
//! │ Header       │ Tensor N … N+1 … N+2 …  │ Checksum           │
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
//!
//! # Example
//!
//! ```rust
//! use briny_ai::manual::tensors::Tensor;
//! use briny_ai::manual::diskio::{save_tensors, load_tensors};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let tensor = Tensor::new(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
//!
//!     // Save a list of tensors
//!     save_tensors("model.bpat", &[tensor.clone()])?;
//!
//!     // Load them back
//!     let tensors = load_tensors::<f64>("model.bpat")?;
//!     println!("Recovered: {:?}", tensors);
//!
//!     Ok(())
//! }
//! ```

use crate::manual::tensors::TensorGrad;
use alloc::vec::Vec;
use briny::raw::{
    Pod,
    casting::{from_bytes_unaligned, slice_to_bytes},
};
use crc32fast::Hasher as Crc32;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use tensor_optim::TensorOps;

use crate::manual::tensors::VecTensor;
use std::io::{BufReader, Read};

/// The original BPAT header.
///
/// Used on `briny_ai` `v0.1.0`-`v0.2.2`.
///
/// # Format
///
/// This version looks like this:
///
/// ```text
/// ┌──────────────┬─────────────────────────┐
/// │ Header       │ Tensor N … N+1 … N+2 …  │
/// ├──────────────┼─────────────────────────┤
/// │ `bpat`       │ u64: ndim               │
/// │ u8: count    │ [u64; ndim] shape       │
/// │              │ [f64; prod(shape)] data │
/// └──────────────┴─────────────────────────┘
/// ```
pub const BPAT_MAGIC_V0: &[u8; 4] = b"bpat";
/// The new BPAT header.
///
/// Created on `briny_ai` `v0.3.0`.
///
/// # Format
///
/// This version looks like this:
///
/// ```text
/// ┌──────────────┬─────────────────────────┬────────────────────┐
/// │ Header       │ Tensor N … N+1 … N+2 …  │ Checksum           │
/// ├──────────────┼─────────────────────────┼────────────────────┤
/// │ `bpat`       │ u64: ndim               │ u32: file checksum │
/// │ u8: count    │ [u64; ndim] shape       │                    │
/// │              │ [f64; prod(shape)] data │                    │
/// │              │ u32: checksum           │                    │
/// └──────────────┴─────────────────────────┴────────────────────┘
/// ```
pub const BPAT_MAGIC_V1: &[u8; 8] = b"BPATv1\0\0";

/// Saves the given tensors to a file.
///
/// # Errors
///
/// When tensors don't seem correct, `std::io::Error` is returned with type of `InvalidData`. If writing to the file failed, a similar error is returned.
pub fn save_tensors<T: TensorGrad + TensorOps<U>, U: Copy + Pod>(
    path: &str,
    tensors: &[T],
) -> Result<(), io::Error> {
    let mut file = BufWriter::new(File::create(path)?);
    let mut hasher = Crc32::new();

    file.write_all(BPAT_MAGIC_V1)?;
    file.write_all(&(tensors.len() as u64).to_le_bytes())?;
    hasher.update(BPAT_MAGIC_V1);
    hasher.update(&(tensors.len() as u64).to_le_bytes());

    for tensor in tensors {
        let expected_len: usize = tensor.shape().iter().product();
        if expected_len != tensor.data().len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "tensor shape/data mismatch",
            ));
        }

        let mut buf = Vec::new();
        buf.extend_from_slice(&(tensor.shape().len() as u64).to_le_bytes());
        for &dim in tensor.shape() {
            buf.extend_from_slice(&(dim as u64).to_le_bytes());
        }
        buf.extend_from_slice(slice_to_bytes(tensor.data()));

        let mut crc = Crc32::new();
        crc.update(&buf);
        let tensor_crc = crc.finalize();

        file.write_all(&buf)?;
        file.write_all(&tensor_crc.to_le_bytes())?;

        hasher.update(&buf);
        hasher.update(&tensor_crc.to_le_bytes());
    }

    let file_crc = hasher.finalize();
    file.write_all(&file_crc.to_le_bytes())?;
    Ok(())
}

/// Loads tensors from a file.
///
/// # Errors
///
/// With most errors, like an invalid header or corrupted checksum, `std::io::Error` is returned - usually `InvalidData`, sometimes `UnexpectedEof`.
///
/// # Panics
///
/// When types cannot be properly coerced, the function will forcefully panic unexpectedly.
pub fn load_tensors<T: Copy + Pod + Default>(path: &str) -> Result<Vec<VecTensor<T>>, io::Error> {
    let mut file = BufReader::new(File::open(path)?);
    let mut full_data = Vec::new();
    file.read_to_end(&mut full_data)?;

    if full_data.len() < BPAT_MAGIC_V1.len() + 8 + 4 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "file too small",
        ));
    }

    // Split the last 4 bytes as the file CRC
    let (data, crc_bytes) = full_data.split_at(full_data.len() - 4);
    let expected_crc = u32::from_le_bytes(crc_bytes.try_into().unwrap());

    let mut hasher = Crc32::new();
    hasher.update(data);
    if hasher.finalize() != expected_crc {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "file checksum mismatch",
        ));
    }

    let mut offset = 0;

    // Check file magic
    if &data[offset..offset + 8] != BPAT_MAGIC_V1 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad file magic"));
    }
    offset += 8;

    // Read tensor count (u64)
    if offset + 8 > data.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "missing tensor count",
        ));
    }
    let count = usize::try_from(u64::from_le_bytes(
        data[offset..offset + 8].try_into().unwrap(),
    ))
    .unwrap();
    offset += 8;

    let mut tensors = Vec::with_capacity(count);

    for _ in 0..count {
        let tensor_start = offset;

        // Read ndim
        if offset + 8 > data.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "missing ndim"));
        }
        let ndim = usize::try_from(u64::from_le_bytes(
            data[offset..offset + 8].try_into().unwrap(),
        ))
        .unwrap();
        offset += 8;

        // Read shape
        let shape_bytes = ndim
            .checked_mul(8)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "ndim overflow"))?;
        if offset + shape_bytes > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "missing shape data",
            ));
        }

        let shape: Vec<u64> = (0..ndim)
            .map(|i| {
                let start = offset + i * 8;
                u64::from_le_bytes(data[start..start + 8].try_into().unwrap())
            })
            .collect();
        offset += shape_bytes;

        // Compute data length
        let len_u64 = shape
            .iter()
            .try_fold(1u64, |acc, &x| acc.checked_mul(x))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "shape overflow"))?;
        let len = usize::try_from(len_u64)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "shape size too large"))?;

        let elem_size = core::mem::size_of::<T>();
        let data_bytes = len
            .checked_mul(elem_size)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "data size overflow"))?;

        if offset + data_bytes > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "missing tensor data",
            ));
        }

        let raw_data = &data[offset..offset + data_bytes];
        offset += data_bytes;

        if offset + 4 > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "missing tensor checksum",
            ));
        }
        let expected_tensor_crc = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        offset += 4;

        let tensor_bytes = &data[tensor_start..offset - 4];
        let mut crc = Crc32::new();
        crc.update(tensor_bytes);
        let actual_crc = crc.finalize();

        if actual_crc != expected_tensor_crc {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "tensor checksum mismatch",
            ));
        }

        let elem_size = core::mem::size_of::<T>();
        if raw_data.len() % elem_size != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "data length mismatch",
            ));
        }

        let elem_size = core::mem::size_of::<T>();
        let tensor_vec: Vec<T> = raw_data
            .chunks_exact(elem_size)
            .map(|chunk| {
                from_bytes_unaligned::<T>(chunk).map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "unaligned or invalid tensor data",
                    )
                })
            })
            .collect::<Result<_, _>>()?;

        let shape_usize: Vec<usize> = shape
            .into_iter()
            .map(|d| {
                usize::try_from(d).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "shape dimension too large")
                })
            })
            .collect::<Result<_, _>>()?;

        tensors.push(VecTensor::with_data(&shape_usize, &tensor_vec));
    }

    Ok(tensors)
}
