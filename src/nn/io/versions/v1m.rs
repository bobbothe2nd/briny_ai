#![allow(clippy::cast_possible_truncation)]

use crate::nn::io::{SerialTensorError, SerialTensorErrorKind, BPAT_MAGIC_V1_MICRO};
use crate::nn::tensors::TensorGrad;
use crate::nn::IntermediateFp;
use alloc::{vec, vec::Vec};
use briny::{raw::slice_to_bytes, traits::Pod};
use crc32fast::Hasher;
use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
};
use tensor_optim::TensorOps;

#[cfg(feature = "std")]
pub fn save_tensors<T: TensorGrad<U> + TensorOps<U>, U: Pod + Copy + IntermediateFp>(
    path: &str,
    tensors: &[T],
) -> Result<(), SerialTensorError> {
    let mut file = BufWriter::new(File::create(path).map_err(|_| SerialTensorError {
        kind: SerialTensorErrorKind::InvalidPath,
        msg: "no such file exists",
    })?);
    let len = 8
        + 4
        + tensors
            .iter()
            .map(|t| 4 + t.shape().len() * 4 + t.data().len() * 4 + 4)
            .sum::<usize>()
        + 4;
    let mut buf = vec![0; len];
    let len = serialize_tensors(tensors, &mut buf)?;
    file.write_all(&buf[..len]).map_err(|_| SerialTensorError {
        kind: SerialTensorErrorKind::InvalidPath,
        msg: "no such file exists",
    })?;
    Ok(())
}

#[cfg(feature = "std")]
pub fn load_tensors<T: TensorGrad<U> + TensorOps<U>, U: Pod + Copy + IntermediateFp>(
    path: &str,
) -> Result<Vec<T>, SerialTensorError> {
    let mut file = BufReader::new(File::open(path).map_err(|_| SerialTensorError {
        kind: SerialTensorErrorKind::InvalidPath,
        msg: "no such file exists",
    })?);
    let mut vec = Vec::with_capacity(5); // will be at least 5 bytes
    file.read_to_end(&mut vec).map_err(|_| SerialTensorError {
        kind: SerialTensorErrorKind::InvalidHeader,
        msg: "file too short",
    })?;

    // magic header
    let magic = &vec[0..8];
    if magic != BPAT_MAGIC_V1_MICRO {
        return Err(SerialTensorError {
            kind: SerialTensorErrorKind::InvalidHeader,
            msg: "invalid magic header",
        });
    }

    let count = u32::from_le_bytes(vec[8..12].try_into().map_err(|_| unreachable!())?);

    if count > u32::MAX / 3 {
        return Err(SerialTensorError {
            kind: SerialTensorErrorKind::InvalidData,
            msg: "",
        });
    }

    let mut tensors = vec![T::new_with_data(&[], &[]); count as usize];

    deserialize_tensors(&mut tensors, &vec)?;
    Ok(tensors)
}

#[cfg(feature = "alloc")]
pub fn serialize_tensors<T: TensorGrad<U> + TensorOps<U>, U: Pod + Copy + IntermediateFp>(
    tensors: &[T],
    buf: &mut [u8],
) -> Result<usize, SerialTensorError> {
    buf[0..8].copy_from_slice(&BPAT_MAGIC_V1_MICRO);
    buf[8..12].copy_from_slice(&(tensors.len() as u32).to_le_bytes());

    let mut hasher = Hasher::new();

    // index into the buffer
    let mut idx = 12;

    for t in tensors {
        debug_assert_eq!(
            t.data().len(),
            t.shape().iter().product(),
            "tensor shape/data mismatch"
        );

        idx += serialize_tensor(t, &mut buf[idx..])?;
    }

    hasher.update(&buf[..idx]);

    let file_crc = hasher.finalize();
    buf[idx..idx + 4].copy_from_slice(&file_crc.to_le_bytes());
    Ok(idx + 4)
}

#[cfg(feature = "alloc")]
pub fn deserialize_tensors<T: TensorGrad<U> + TensorOps<U>, U: Pod + Copy + IntermediateFp>(
    tensors: &mut [T],
    buf: &[u8],
) -> Result<(), SerialTensorError> {
    // tensor count
    let count = u32::from_le_bytes(buf[8..12].try_into().map_err(|_| unreachable!())?) as usize;

    debug_assert_eq!(count, tensors.len());

    // index into the buffer
    let mut idx = 12;

    for t in tensors.iter_mut().take(count) {
        idx += deserialize_tensor(t, &buf[idx..])?;
    }

    let stored_crc = u32::from_le_bytes(buf[idx..idx + 4].try_into().unwrap());

    let mut hasher = Hasher::new();
    hasher.update(&buf[..idx]);
    let computed_crc = hasher.finalize();

    if stored_crc != computed_crc {
        return Err(SerialTensorError {
            kind: SerialTensorErrorKind::IntegrityUnverified,
            msg: "file checksum mismatch",
        });
    }

    Ok(())
}

#[cfg(feature = "alloc")]
pub fn serialize_tensor<T: TensorGrad<U> + TensorOps<U>, U: Pod + Copy + IntermediateFp>(
    tensor: &T,
    buf: &mut [u8],
) -> Result<usize, SerialTensorError> {
    let mut idx = 0;

    let expected_len: usize = tensor.shape().iter().product();
    if expected_len != tensor.data().len() {
        return Err(SerialTensorError {
            kind: SerialTensorErrorKind::InvalidData,
            msg: "tensor shape/data length mismatch",
        });
    }

    buf[idx..idx + 4].copy_from_slice(&(tensor.shape().len() as u32).to_le_bytes());
    idx += 4;
    for &dim in tensor.shape() {
        buf[idx..idx + 4].copy_from_slice(&(dim as u32).to_le_bytes());
        idx += 4;
    }

    let t_len = tensor.len() * size_of::<f32>();
    buf[idx..idx + t_len].copy_from_slice(slice_to_bytes(tensor.data()));
    idx += t_len;

    let mut crc = Hasher::new();
    crc.update(&buf[..idx]);
    let tensor_crc = crc.finalize();

    buf[idx..idx + 4].copy_from_slice(&tensor_crc.to_le_bytes());

    Ok(idx + 4)
}

#[cfg(feature = "alloc")]
pub fn deserialize_tensor<T: TensorGrad<U> + TensorOps<U>, U: Pod + Copy + IntermediateFp>(
    tensor: &mut T,
    buf: &[u8],
) -> Result<usize, SerialTensorError> {
    let mut idx = 0;

    let buf4 = &buf[idx..idx + 4];
    idx += 4;

    let ndim = u32::from_le_bytes(buf4.try_into().map_err(|_| unreachable!())?) as usize;

    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        let buf4 = &buf[idx..idx + 4];
        idx += 4;

        shape.push(u32::from_le_bytes(
            buf4.try_into().map_err(|_| unreachable!())?,
        ));
    }

    let size: usize = shape.iter().product::<u32>() as usize;
    let mut data = Vec::with_capacity(size);
    for _ in 0..size {
        let buf4 = &buf[idx..idx + 4];
        idx += 4;

        data.push(U::from_f32(f32::from_le_bytes(
            buf4.try_into().map_err(|_| unreachable!())?,
        )));
    }

    let shape_usize: Vec<usize> = shape.iter().map(|&x| x as usize).collect();

    if shape_usize.iter().product::<usize>() != data.len() {
        return Err(SerialTensorError {
            kind: SerialTensorErrorKind::InvalidData,
            msg: "tensor shape incorrect for data",
        });
    }

    *tensor = T::new_with_data(&data, &shape_usize);

    let stored_crc = u32::from_le_bytes(buf[idx..idx + 4].try_into().unwrap());
    idx += 4;

    let mut hasher = Hasher::new();
    hasher.update(&buf[..idx - 4]);
    let computed_crc = hasher.finalize();

    if stored_crc != computed_crc {
        return Err(SerialTensorError {
            kind: SerialTensorErrorKind::IntegrityUnverified,
            msg: "tensor checksum mismatch",
        });
    }

    Ok(idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "std")]
    fn save_load_roundrip() {
        use crate::nn::tensors::{Tensor, VecTensor};

        let a = Tensor::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Tensor::new(&[1, 4], &[7.0, 4.0, 9.0, 10.0]);
        let original = [a.as_vectensor(), b.as_vectensor()];

        save_tensors("checkpoints/test/v1m.bpat", &original).unwrap();

        let loaded: Vec<VecTensor<f32>> = load_tensors("checkpoints/test/v1m.bpat").unwrap();

        assert_eq!(original, loaded.as_slice());
    }
}
