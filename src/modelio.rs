//! Robust saving/loading of model weights.
//! 
//! # `.bpat` Model Serialization Format
//!
//! This module provides minimal, dependency-free utilities for saving and loading tensors in a custom binary format.
//! It's intended for saving small models in a compact, fast, and portable format for `briny_ai` or similar systems.
//!
//! # Format Overview
//!
//! A `.bpat` file stores one or more tensors in the following layout:
//!
//! ```text
//! ┌────────────┬────────────┬─────────────────────┐
//! │ Header     │ Tensor N   │ Tensor N+1 …        │
//! ├────────────┼────────────┼─────────────────────┤
//! │ "bpat"[4]  │ u64: ndim  │ u64: ndim           │
//! │ u8: count  │ [u64; ndim] shape                │
//! │            │ [f64; prod(shape)] data          │
//! └────────────┴──────────────────────────────────┘
//! ```
//!
//! ## Header
//! - `bpat` magic (4 bytes): ensures file is recognized
//! - `u8` tensor count: number of tensors to read
//!
//! ## Tensor Encoding
//! For each tensor:
//! - `ndim` (`u64`): number of dimensions
//! - `shape` (`u64 * ndim`): each dimension size
//! - `data` (`f64 * prod(shape)`): flattened, row-major tensor data
//!
//! # Design Principles
//! - Fully self-contained
//! - No compression or encryption (for simplicity and speed)
//! - Suitable for deterministic, reproducible serialization
//! - Works on little-endian platforms
//!
//! # Limitations
//! - Assumes `f64` element type
//! - Maximum 255 tensors per file (due to `u8` count limit)
//! - No per-tensor metadata (names, dtypes, etc.)
//!
//! # Example
//!
//! ```no_run
//! # use briny_ai::tensors::Tensor;
//! # use briny_ai::modelio::{save_model, load_model};
//! let tensor = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
//!
//! // Save a list of tensors
//! save_model("model.bpat", &[tensor.clone()]).unwrap();
//!
//! // Load them back
//! let tensors = load_model("model.bpat").unwrap();
//! println!("Recovered: {:?}", tensors);
//! ```

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use crate::tensors::Tensor;
use std::error::Error;

/// Save a list of tensors to a `.bpat` file.
/// 
/// - Uses a simple binary format with a magic header and tensor count.
/// - Each tensor includes dimension count, shape, and flattened f64 data.
/// 
/// # Arguments
/// - `path`: Output file path.
/// - `tensors`: Slice of tensors to save.
///
/// # Errors
/// - Returns an error if file I/O or write fails.
/// 
pub fn save_model(path: &str, tensors: &[Tensor<f64>]) -> Result<(), Box<dyn Error>> {
    let mut file = BufWriter::new(File::create(path)?);

    // Write magic header and tensor count
    file.write_all(b"bpat")?;
    file.write_all(&[tensors.len() as u8])?;

    for tensor in tensors {
        let dims = tensor.shape.len() as u64;
        file.write_all(&dims.to_le_bytes())?;

        for &dim in &tensor.shape {
            file.write_all(&(dim as u64).to_le_bytes())?;
        }

        for &val in &tensor.data {
            file.write_all(&val.to_le_bytes())?;
        }
    }

    Ok(())
}

/// Load a `.bpat` file containing multiple tensors.
///
/// - Validates magic header and reads shape and data.
/// - Assumes all data is `f64`, little-endian encoded.
///
/// # Arguments
/// - `path`: File path to read.
///
/// # Returns
/// - A `Vec<Tensor<f64>>` loaded from the file.
///
/// # Errors
/// - Fails if the file does not start with `bpat` or is corrupted.
///
/// # Panics
/// - Panics if shape and data sizes mismatch (shouldn't occur with valid files).
///
pub fn load_model(path: &str) -> Result<Vec<Tensor<f64>>, Box<dyn Error>> {
    let mut file = BufReader::new(File::open(path)?);
    let mut buf8 = [0u8; 8];

    // Check magic header
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    if &magic != b"bpat" {
        return Err("invalid magic header".into());
    }

    // Read tensor count
    let mut count = [0u8; 1];
    file.read_exact(&mut count)?;
    let count = count[0] as usize;

    let mut tensors = Vec::with_capacity(count);

    for _ in 0..count {
        file.read_exact(&mut buf8)?;
        let dims = u64::from_le_bytes(buf8) as usize;

        let mut shape = vec![0usize; dims];
        for dim in &mut shape {
            file.read_exact(&mut buf8)?;
            *dim = u64::from_le_bytes(buf8) as usize;
        }

        let size = shape.iter().product();
        let mut data = vec![0.0f64; size];
        for val in &mut data {
            file.read_exact(&mut buf8)?;
            *val = f64::from_le_bytes(buf8);
        }

        tensors.push(Tensor::new(shape.clone(), data));
    }

    Ok(tensors)
}
