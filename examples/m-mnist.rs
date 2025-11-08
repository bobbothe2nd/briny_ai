#[cfg(all(feature = "std", feature = "dyntensor"))]
fn main() {
    use std::fs::File;
    use std::io::Read;

    use briny_ai::backend::{set_backend, Backend};
    use briny_ai::nn::ops::dispatch::{matmul, mse_loss, relu, sgd};
    // use briny_ai::nn::io::{load_tensors, save_tensors, BpatHeader};
    use briny_ai::nn::tensors::{IntoWithGrad, Tensor, TensorOps};

    use briny_ai::nn::TensorFloat;

    fn load_images(path: &str) -> Vec<Vec<TensorFloat>> {
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
            let img: Vec<TensorFloat> = buf[start..end]
                .iter()
                .map(|&b| b as TensorFloat / 255.0)
                .collect();
            images.push(img);
        }
        images
    }

    fn load_labels(path: &str) -> Vec<Vec<TensorFloat>> {
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

    let images = load_images("checkpoints/mnist/data/train-images-idx3-ubyte");
    let labels = load_labels("checkpoints/mnist/data/train-labels-idx1-ubyte");

    let n_samples = 1000;
    let input_dim = 784;
    let hidden_dim = 128;
    let output_dim = 10;

    let x_data = images
        .iter()
        .take(n_samples)
        .flat_map(|x| x.clone())
        .collect::<Vec<TensorFloat>>();
    let y_data = labels
        .iter()
        .take(n_samples)
        .flat_map(|y| y.clone())
        .collect::<Vec<TensorFloat>>();

    set_backend(Backend::Wgpu);

    let x = Tensor::new(&[n_samples, input_dim], &x_data).with_grad();

    let y = Tensor::new(&[n_samples, output_dim], &y_data);

    let mut w1 = Tensor::new(
        &[input_dim, hidden_dim],
        (0..input_dim * hidden_dim)
            .map(|_| rand::random::<TensorFloat>() * 2.0 - 1.0)
            .collect::<Vec<TensorFloat>>()
            .as_slice(),
    )
    .with_grad();

    let mut w2 = Tensor::new(
        &[hidden_dim, output_dim],
        (0..hidden_dim * output_dim)
            .map(|_| rand::random::<TensorFloat>() * 2.0 - 1.0)
            .collect::<Vec<TensorFloat>>()
            .as_slice(),
    )
    .with_grad();

    for epoch in 0..1000 {
        let (z1, back1) = matmul(&x, &w1);
        let z1_wrapped = z1.with_grad();

        let (a1, back_relu) = relu(&z1_wrapped);
        let a1_wrapped = a1.with_grad();

        let (z2, back2) = matmul(&a1_wrapped, &w2);
        let z2_wrapped = z2.with_grad();

        let (loss, back_loss) = mse_loss(&z2_wrapped, &y);

        if (epoch + 1) % 100 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch, loss);
        }

        // Backprop
        let dloss = back_loss(1.0);
        let (dz2_a1, dz2_w2) = back2(dloss);
        let d_relu = back_relu(dz2_a1);
        let (_, dz1_w1) = back1(d_relu);

        let grads_w1 = dz1_w1.data().to_vec();
        let grads_w2 = dz2_w2.data().to_vec();

        drop(back1);
        drop(back2);
        drop(back_relu);

        // Accumulate grads
        for (g, val) in w1.get_grad_mut().data_mut().iter_mut().zip(grads_w1) {
            *g += val;
        }
        for (g, val) in w2.get_grad_mut().data_mut().iter_mut().zip(grads_w2) {
            *g += val;
        }

        sgd(&mut w1, 0.01);
        sgd(&mut w2, 0.01);
    }

    // println!("Saving tensors...");
    // save_tensors(
    //     "checkpoints/m-mnist/model.bpat",
    //     &[w1.get_value().clone(), w2.get_value().clone()],
    //     BpatHeader::BpatV1
    // )
    // .unwrap();

    // let restored = load_tensors::<TensorFloat>("checkpoints/m-mnist/model.bpat").unwrap();
    // for tensor in restored {
    //     println!(
    //         "Shape: {:?}, First vals: {:?}",
    //         tensor.shape(),
    //         &tensor.data()[..5]
    //     );
    // }
}

#[cfg(not(all(feature = "std", feature = "dyntensor")))]
fn main() {
    println!("MNIST not supported without `std` and `dyntensor`");
}
