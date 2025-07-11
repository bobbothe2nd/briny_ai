use std::fs::{create_dir_all, File};
use std::io::Read;
use std::path::Path;

use briny_ai::backprop::{matmul, mse_loss, relu, sgd};
use briny_ai::modelio::{load_model, save_model};
use briny_ai::tensors::{Tensor, WithGrad};
use briny_ai::backend::{Backend, set_backend, get_backend};

use flate2::read::GzDecoder;
use reqwest::blocking::get;

const TRAIN_IMAGES_URL: &str = "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz";
const TRAIN_LABELS_URL: &str = "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz";

fn download_and_extract(url: &str, output_path: &str) {
    use std::io::{copy};

    let resp = get(url).expect("Failed to fetch URL");

    if !resp.status().is_success() {
        panic!("Failed to download {}: HTTP {}", url, resp.status());
    }

    let mut decoder = GzDecoder::new(resp);
    let mut out = File::create(output_path).expect("Failed to create file");
    
    copy(&mut decoder, &mut out).expect("Failed to decompress");
}

fn load_images(path: &str) -> Vec<Vec<f64>> {
    let mut f = File::open(path).unwrap();
    let mut buf = vec![];
    f.read_to_end(&mut buf).unwrap();
    assert_eq!(&buf[0..4], &[0, 0, 8, 3]);
    let count = u32::from_be_bytes(buf[4..8].try_into().unwrap()) as usize;
    let rows = u32::from_be_bytes(buf[8..12].try_into().unwrap()) as usize;
    let cols = u32::from_be_bytes(buf[12..16].try_into().unwrap()) as usize;
    let mut images = Vec::with_capacity(count);
    for i in 0..count {
        let start = 16 + i * rows * cols;
        let end = start + rows * cols;
        let img: Vec<f64> = buf[start..end].iter().map(|&b| b as f64 / 255.0).collect();
        images.push(img);
    }
    images
}

fn load_labels(path: &str) -> Vec<Vec<f64>> {
    let mut f = File::open(path).unwrap();
    let mut buf = vec![];
    f.read_to_end(&mut buf).unwrap();
    assert_eq!(&buf[0..4], &[0, 0, 8, 1]);
    let count = u32::from_be_bytes(buf[4..8].try_into().unwrap()) as usize;
    let mut labels = Vec::with_capacity(count);
    for i in 0..count {
        let mut onehot = vec![0.0; 10];
        onehot[buf[8 + i] as usize] = 1.0;
        labels.push(onehot);
    }
    labels
}

fn main() {
    create_dir_all("mnist_data").unwrap();
    if !Path::new("mnist_data/train-images-idx3-ubyte").exists() {
        println!("Downloading MNIST dataset...");
        download_and_extract(TRAIN_IMAGES_URL, "mnist_data/train-images-idx3-ubyte");
        download_and_extract(TRAIN_LABELS_URL, "mnist_data/train-labels-idx1-ubyte");
    }

    let images = load_images("mnist_data/train-images-idx3-ubyte");
    let labels = load_labels("mnist_data/train-labels-idx1-ubyte");

    let n_samples = 1000;
    let input_dim = 784;
    let hidden_dim = 128;
    let output_dim = 10;

    let x_data: Vec<f64> = images.iter().take(n_samples).flat_map(|x| x.clone()).collect();
    let y_data: Vec<f64> = labels.iter().take(n_samples).flat_map(|y| y.clone()).collect();

    set_backend(Backend::Wgpu);
    println!("Current backend: {:#?}", get_backend());

    let x = WithGrad {
        value: Tensor::new(vec![n_samples, input_dim], x_data),
        grad: Tensor::new(vec![n_samples, input_dim], vec![0.0; n_samples * input_dim]),
    };
    let y = Tensor::new(vec![n_samples, output_dim], y_data);

    let mut w1 = WithGrad {
        value: Tensor::new(vec![input_dim, hidden_dim], (0..input_dim * hidden_dim)
            .map(|_| rand::random::<f64>() * 2.0 - 1.0).collect()),
        grad: Tensor::new(vec![input_dim, hidden_dim], vec![0.0; input_dim * hidden_dim]),
    };
    let mut w2 = WithGrad {
        value: Tensor::new(vec![hidden_dim, output_dim], (0..hidden_dim * output_dim)
            .map(|_| rand::random::<f64>() * 2.0 - 1.0).collect()),
        grad: Tensor::new(vec![hidden_dim, output_dim], vec![0.0; hidden_dim * output_dim]),
    };

    for epoch in 0..200 {
        let (z1, back1) = matmul(&x, &w1);
        let z1_wrapped = WithGrad {
            value: z1.clone(),
            grad: Tensor::new(z1.shape.clone(), vec![0.0; z1.data.len()]),
        };
        let (a1, back_relu) = relu(&z1_wrapped);
        let a1_wrapped = WithGrad {
            value: a1.clone(),
            grad: Tensor::new(z1.shape.clone(), vec![0.0; a1.data.len()]),
        };
        let (z2, back2) = matmul(&a1_wrapped, &w2);
        let z2_wrapped = WithGrad {
            value: z2.clone(),
            grad: Tensor::new(z2.shape.clone(), vec![0.0; z2.data.len()]),
        };
        let (loss, back_loss) = mse_loss(&z2_wrapped, &y);

        if epoch % 25 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch, loss);
        }

        let dloss = back_loss(1.0);
        let (dz2_a1, dz2_w2) = back2(&dloss);
        let d_relu = back_relu(&dz2_a1);
        let (_, dz1_w1) = back1(&d_relu);

        drop(back1);
        drop(back2);

        for (g, val) in w1.grad.data.iter_mut().zip(dz1_w1.data) {
            *g += val;
        }
        for (g, val) in w2.grad.data.iter_mut().zip(dz2_w2.data) {
            *g += val;
        }

        sgd(&mut w1, 0.01);
        sgd(&mut w2, 0.01);
    }

    println!("Saving model...");
    save_model("mnist_model.bpat", &[w1.value.clone(), w2.value.clone()]).unwrap();

    let restored = load_model("mnist_model.bpat").unwrap();
    for tensor in restored {
        println!("Shape: {:?}, First vals: {:?}", tensor.shape, &tensor.data[..5]);
    }
}
