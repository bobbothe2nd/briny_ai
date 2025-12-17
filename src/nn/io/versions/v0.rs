use crate::nn::IntermediateFp;
use crate::nn::io::{SerialTensorError, SerialTensorErrorKind, BPAT_MAGIC_V0};
use crate::nn::tensors::TensorGrad;
#[cfg(feature = "alloc")]
use alloc::{vec, vec::Vec};
use briny::raw::{slice_to_bytes, to_bytes};
#[cfg(feature = "std")]
use briny::traits::Pod;
use core::mem::zeroed;
#[cfg(feature = "std")]
use std::fs::File;
#[cfg(feature = "std")]
use std::io::{BufReader, BufWriter, Read, Write};
use tensor_optim::TensorOps;

#[cfg(feature = "std")]
pub fn save_tensors<T: TensorGrad<U> + TensorOps<U>, U: Copy + Pod + IntermediateFp>(
    path: &str,
    tensors: &[T],
) -> Result<(), SerialTensorError> {
    let mut file = BufWriter::new(File::create(path).map_err(|_| SerialTensorError {
        kind: SerialTensorErrorKind::InvalidPath,
        msg: "no such file exists",
    })?);
    let len = 5 + {
        let mut size = 0;
        for t in tensors {
            size += 8;
            size += t.data().len() * size_of::<f64>();
            size += t.shape().len() * size_of::<u64>();
        }
        size
    };
    let mut buf = vec![0; len];
    serialize_tensors(tensors, &mut buf);
    file.write_all(&buf).map_err(|_| SerialTensorError {
        kind: SerialTensorErrorKind::InvalidPath,
        msg: "no such file exists",
    })?;
    Ok(())
}

#[cfg(feature = "std")]
pub fn load_tensors<T: TensorGrad<U> + TensorOps<U>, U: Copy + Pod + IntermediateFp>(
    path: &str,
) -> Result<Vec<T>, SerialTensorError> {
    let mut file = BufReader::new(File::open(path).map_err(|_| SerialTensorError {
        kind: SerialTensorErrorKind::InvalidPath,
        msg: "no such file exists",
    })?);
    let mut vec = Vec::with_capacity(5); // will be at least 5 bytes
    let len = file.read_to_end(&mut vec).map_err(|_| SerialTensorError {
        kind: SerialTensorErrorKind::InvalidHeader,
        msg: "file too short",
    })?;
    let buf = &vec[0..len];
    let mut tensors = vec![T::new_with_data(&[], &[]); buf[4] as usize];
    deserialize_tensors(&mut tensors, buf)?;
    Ok(tensors)
}

#[cfg(feature = "alloc")]
pub fn serialize_tensors<T: TensorGrad<U> + TensorOps<U>, U: Copy + Pod + IntermediateFp>(
    tensors: &[T],
    buf: &mut [u8],
) {
    buf[0..4].copy_from_slice(&BPAT_MAGIC_V0);
    buf[4] = tensors.len() as u8;

    // tensor count
    let count = buf[4] as usize;

    debug_assert_eq!(count, tensors.len());

    // index into the buffer
    let mut idx = 5;

    for t in tensors {
        debug_assert_eq!(
            t.data().len(),
            t.shape().iter().product(),
            "tensor shape/data mismatch"
        );

        idx += serialize_tensor(t, &mut buf[idx..]);
    }
}

#[cfg(feature = "alloc")]
pub fn deserialize_tensors<T: TensorGrad<U> + TensorOps<U>, U: Copy + Pod + IntermediateFp>(
    tensors: &mut [T],
    buf: &[u8],
) -> Result<(), SerialTensorError> {
    // magic header
    let magic = &buf[0..4];
    if magic != BPAT_MAGIC_V0 {
        return Err(SerialTensorError {
            kind: SerialTensorErrorKind::InvalidHeader,
            msg: "invalid magic header",
        });
    }

    // tensor count
    let count = buf[4] as usize;

    assert_eq!(count, tensors.len());

    // index into the buffer
    let mut idx = 5;

    for t in tensors.iter_mut().take(count) {
        idx += deserialize_tensor(t, &buf[idx..])?;
    }

    Ok(())
}

#[cfg(feature = "alloc")]
pub fn serialize_tensor<T: TensorGrad<U> + TensorOps<U>, U: Copy + Pod + IntermediateFp>(
    tensor: &T,
    buf: &mut [u8],
) -> usize {
    let mut idx = 0;

    let data = tensor
        .data()
        .iter()
        .map(|&x| x.into_f64())
        .collect::<Vec<f64>>();
    let data = data.as_slice();
    let shape = tensor
        .shape()
        .iter()
        .map(|&x| x as u64)
        .collect::<Vec<u64>>();
    let shape = shape.as_slice();

    let rank = shape.len() as u64;
    let ndim = to_bytes(&rank);
    buf[idx..idx + 8].copy_from_slice(ndim);
    idx += 8;

    let data_size = size_of_val(data);
    let shape_size = size_of_val(shape);
    let shape = slice_to_bytes(shape);
    let data = slice_to_bytes(data);
    buf[idx..idx + shape_size].copy_from_slice(shape);
    idx += shape_size;
    buf[idx..idx + data_size].copy_from_slice(data);
    idx += data_size;

    idx
}

#[cfg(feature = "alloc")]
pub fn deserialize_tensor<T: TensorGrad<U> + TensorOps<U>, U: Copy + Pod + IntermediateFp>(
    tensor: &mut T,
    buf: &[u8],
) -> Result<usize, SerialTensorError> {
    let mut idx = 0;

    let buf8 = &buf[idx..idx + 8];
    idx += 8;

    let ndim = u64::from_ne_bytes(buf8.try_into().map_err(|_| SerialTensorError {
        kind: SerialTensorErrorKind::InvalidData,
        msg: "tensor rank invalid",
    })?) as usize;

    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        let buf8 = &buf[idx..idx + 8];
        idx += 8;

        shape.push(u64::from_ne_bytes(buf8.try_into().map_err(|_| {
            SerialTensorError {
                kind: SerialTensorErrorKind::InvalidData,
                msg: "tensor rank incorrect for shape",
            }
        })?));
    }

    let size: usize = shape.iter().product::<u64>() as usize;
    let mut data = Vec::with_capacity(size);
    for _ in 0..size {
        let buf8 = &buf[idx..idx + 8];
        idx += 8;

        data.push(
            U::from_f64(f64::from_ne_bytes(buf8.try_into().map_err(|_| SerialTensorError {
                kind: SerialTensorErrorKind::InvalidData,
                msg: "tensor shape incorrect for data",
            })?)),
        );
    }

    let shape_usize: Vec<usize> = shape.iter().map(|&x| x as usize).collect();

    if shape_usize.iter().product::<usize>() != data.len() {
        return Err(SerialTensorError {
            kind: SerialTensorErrorKind::InvalidData,
            msg: "tensor shape incorrect for data",
        });
    }

    *tensor = T::new_with_data(&data, &shape_usize);

    Ok(idx)
}

pub fn save_tensor<
    T: TensorGrad<U> + TensorOps<U>, U: Copy + Pod + IntermediateFp,
    const N: usize,
    const D: usize,
>(
    tensor: &T,
    buf: &mut [u8],
) -> usize {
    let mut idx = 0;

    let mut data = unsafe { zeroed::<[f64; N]>() };
    #[allow(clippy::unnecessary_cast)]
    for (i, val) in tensor.data().iter().enumerate() {
        data[i] = (*val).into_f64();
    }
    let mut shape = unsafe { zeroed::<[u64; D]>() };
    #[allow(clippy::unnecessary_cast)]
    for (i, val) in tensor.shape().iter().enumerate() {
        shape[i] = *val as _;
    }

    let rank = shape.len() as u64;
    let ndim = to_bytes(&rank);
    buf[idx..idx + 8].copy_from_slice(ndim);
    idx += 8;

    let data_size = const { 8 * N };
    let shape_size = const { 8 * D };
    let shape = slice_to_bytes(&shape);
    let data = slice_to_bytes(&data);
    buf[idx..idx + shape_size].copy_from_slice(shape);
    idx += shape_size;
    buf[idx..idx + data_size].copy_from_slice(data);
    idx += data_size;

    idx
}

pub fn load_tensor<
    T: TensorGrad<U> + TensorOps<U>, U: Copy + Pod + IntermediateFp,
    const N: usize,
    const D: usize,
>(
    tensor: &mut T,
    buf: &[u8],
) -> Result<usize, SerialTensorError> {
    let mut idx = 0; // skip ndim

    {
        let buf8 = &buf[idx..idx + 8];
        let ndim = u64::from_ne_bytes(buf8.try_into().map_err(|_| unreachable!())?);
        if (ndim as usize) != D {
            return Err(SerialTensorError {
                kind: SerialTensorErrorKind::InvalidData,
                msg: "expected constants fail to match actual values",
            });
        }
    }

    let mut shape = unsafe { zeroed::<[usize; D]>() };
    for val in &mut shape {
        let buf8 = &buf[idx..idx + 8];
        idx += 8;

        *val = u64::from_ne_bytes(buf8.try_into().map_err(|_| unreachable!())?) as usize;
    }

    {
        let size = shape.iter().product::<usize>();
        if size != N {
            return Err(SerialTensorError {
                kind: SerialTensorErrorKind::InvalidData,
                msg: "expected constants fail to match actual values",
            });
        }
    }

    let mut data = unsafe { zeroed::<[U; N]>() };
    for val in &mut data {
        let buf8 = &buf[idx..idx + 8];
        idx += 8;

        *val = U::from_f64(f64::from_ne_bytes(buf8.try_into().map_err(|_| unreachable!())?));
    }

    *tensor = T::new_with_data(&data, &shape);

    Ok(idx)
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(feature = "std")]
    fn save_load_roundrip() {
        use super::*;

        use crate::nn::tensors::{Tensor, VecTensor};

        let a = Tensor::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Tensor::new(&[1, 4], &[7.0, 8.0, 9.0, 10.0]);
        let original = [a.as_vectensor(), b.as_vectensor()];

        save_tensors("checkpoints/test/v0.bpat", &original).unwrap();

        let loaded: Vec<VecTensor<f32>> = load_tensors("checkpoints/test/v0.bpat").unwrap();

        assert_eq!(original, loaded.as_slice());
    }
}
