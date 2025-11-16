use crate::nn::io::{SerialTensorError, SerialTensorErrorKind, BPAT_MAGIC_V0};
use crate::nn::tensors::TensorGrad;
use crate::nn::TensorFloat;
#[cfg(feature = "alloc")]
use alloc::{vec, vec::Vec};
use briny::raw::{slice_to_bytes, to_bytes};
#[cfg(feature = "std")]
use std::fs::File;
#[cfg(feature = "std")]
use std::io::{BufReader, BufWriter, Read, Write};
use tensor_optim::TensorOps;

#[cfg(feature = "std")]
pub fn save_tensors<T: TensorGrad<TensorFloat> + TensorOps<TensorFloat>>(
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
pub fn load_tensors<T: TensorGrad<TensorFloat> + TensorOps<TensorFloat>>(
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
pub fn serialize_tensors<T: TensorGrad<TensorFloat> + TensorOps<TensorFloat>>(
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
pub fn deserialize_tensors<T: TensorGrad<TensorFloat> + TensorOps<TensorFloat>>(
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
pub fn serialize_tensor<T: TensorGrad<TensorFloat> + TensorOps<TensorFloat>>(
    tensor: &T,
    buf: &mut [u8],
) -> usize {
    let mut idx = 0;

    let data = tensor
        .data()
        .iter()
        .map(|&x| x as f64)
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
pub fn deserialize_tensor<T: TensorGrad<TensorFloat> + TensorOps<TensorFloat>>(
    tensor: &mut T,
    buf: &[u8],
) -> Result<usize, SerialTensorError> {
    let mut idx = 0;

    let buf8 = &buf[idx..idx + 8];
    idx += 8;

    let ndim = u64::from_le_bytes(buf8.try_into().map_err(|_| SerialTensorError {
        kind: SerialTensorErrorKind::InvalidData,
        msg: "tensor rank invalid",
    })?) as usize;

    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        let buf8 = &buf[idx..idx + 8];
        idx += 8;

        shape.push(u64::from_le_bytes(buf8.try_into().map_err(|_| {
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
            f64::from_le_bytes(buf8.try_into().map_err(|_| SerialTensorError {
                kind: SerialTensorErrorKind::InvalidData,
                msg: "tensor shape incorrect for data",
            })?) as TensorFloat,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "std")]
    fn save_load_roundrip() {
        use crate::nn::tensors::{Tensor, VecTensor};

        let a = Tensor::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Tensor::new(&[1, 4], &[7.0, 8.0, 9.0, 10.0]);
        let original = [a.as_vectensor(), b.as_vectensor()];

        save_tensors("checkpoints/test/v0.bpat", &original).unwrap();

        let loaded: Vec<VecTensor<TensorFloat>> = load_tensors("checkpoints/test/v0.bpat").unwrap();

        assert_eq!(original, loaded.as_slice());
    }
}
