use briny_ai::manual::TensorFloat;
use briny_ai::manual::backprop::{matmul, mse_loss, relu, sgd};
use briny_ai::manual::tensors::IntoWithGrad;
#[cfg(all(feature = "dyntensor", feature = "std"))]
use briny_ai::manual::tensors::TensorGrad;
use briny_ai::manual::tensors::{Tensor, TensorOps};
use rand::seq::SliceRandom;
use rand_distr::{Distribution, Normal};

#[cfg(feature = "dyntensor")]
fn main() {
    // should noise be applied to the input?
    const USE_NOISE: bool = true;
    // how much noise should be applied to the input?
    const NOISE_STD: TensorFloat = 0.1;

    // XOR base set
    let base_inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let base_targets = [0.0, 1.0, 1.0, 0.0];

    // model: 2 → 4 → 1
    let mut w1 = Tensor::new(&[2, 4], &[0.5, -0.2, 0.3, -0.4, -0.3, 0.8, -0.1, 0.7]).with_grad();
    let mut w2 = Tensor::new(&[4, 1], &[0.1, -0.3, 0.6, -0.1]).with_grad();

    let lr = 0.001;
    let epochs = 1000;
    let mut rng = rand::rng();

    let noise_dist = Normal::new(0.0, NOISE_STD).unwrap();

    let add_noise =
        |x: [TensorFloat; 2], rng: &mut _, dist: &Normal<TensorFloat>| -> [TensorFloat; 2] {
            if USE_NOISE {
                [x[0] + dist.sample(rng), x[1] + dist.sample(rng)]
            } else {
                x
            }
        };

    for epoch in 0..epochs {
        let mut loss_accum = 0.0;
        let mut correct_train = 0;

        // shuffle training order
        let mut idx = [0, 1, 2, 3];
        idx.shuffle(&mut rng);

        for &i in &idx {
            let noise_x = add_noise(base_inputs[i], &mut rng, &noise_dist);
            let x0 = Tensor::new(&[1, 2], &noise_x);
            let y = Tensor::new(&[1, 1], &[base_targets[i]]);

            let x = x0.clone().with_grad();

            // forward pass
            let (z1, back1) = matmul(&x, &w1);
            let z1_wrapped = z1.clone().with_grad();
            let (a1, back_relu) = relu(&z1_wrapped);
            let a1_wrapped = a1.clone().with_grad();
            let (z2, back2) = matmul(&a1_wrapped, &w2);
            let prediction = z2.clone().with_grad();

            // Loss
            let (loss, back_loss) = mse_loss(&prediction, &y);
            loss_accum += loss;

            // accuracy
            let pred_val = prediction.get_value().data()[0];
            if (pred_val >= 0.5 && y.data()[0] == 1.0) || (pred_val < 0.5 && y.data()[0] == 0.0) {
                correct_train += 1;
            }

            // backprop
            let dloss = back_loss(1.0);
            let (dz2_a1, dz2_w2) = back2(dloss);
            let d_relu = back_relu(dz2_a1);
            let (_, dz1_w1) = back1(d_relu);

            let grads_w1 = dz1_w1.data().to_vec();
            let grads_w2 = dz2_w2.data().to_vec();

            drop(back1);
            drop(back2);

            // Accumulate grads
            for (g, val) in w1.get_grad_mut().data_mut().iter_mut().zip(grads_w1) {
                *g += val;
            }
            for (g, val) in w2.get_grad_mut().data_mut().iter_mut().zip(grads_w2) {
                *g += val;
            }
        }

        // SGD update
        sgd(&mut w1, lr);
        sgd(&mut w2, lr);

        // test loop
        let mut test_loss = 0.0;
        let mut correct_test = 0;
        for i in 0..4 {
            let noise_x = add_noise(base_inputs[i], &mut rng, &noise_dist);
            let target = (noise_x[0].round() as u8 ^ noise_x[1].round() as u8) as TensorFloat;

            let x = Tensor::new(&[1, 2], &noise_x);
            let xg = x.clone().with_grad();

            let (z1, _) = matmul(&xg, &w1);
            let (a1, _) = relu(&z1.clone().with_grad());
            let (z2, _) = matmul(&a1.clone().with_grad(), &w2);

            let pred_val = z2.data()[0];
            let loss = (pred_val - target).powi(2);
            test_loss += loss;

            if (pred_val >= 0.5 && target == 1.0) || (pred_val < 0.5 && target == 0.0) {
                correct_test += 1;
            }
        }

        if (epoch + 1) % 100 == 0 {
            println!(
                "Epoch {:4}: train_loss={:.6} train_acc={:.2}% test_loss={:.6} test_acc={:.2}%",
                epoch + 1,
                loss_accum / 4.0,
                100.0 * correct_train as TensorFloat / 4.0,
                test_loss / 4.0,
                100.0 * correct_test as TensorFloat / 4.0
            );
        }
    }

    #[cfg(feature = "std")]
    {
        briny_ai::manual::diskio::save_tensors(
            "xor_model.bpat",
            &[w1.get_value().clone(), w2.get_value().clone()],
        )
        .unwrap();
        println!("Model saved to xor_model.bpat");

        let loaded =
            briny_ai::manual::diskio::load_tensors::<TensorFloat>("xor_model.bpat").unwrap();
        println!("Loaded tensors:");
        for t in loaded {
            println!(
                "Shape: {:?}, First few vals: {:?}",
                t.shape(),
                &t.data()[..4.min(t.len())]
            );
        }
    }
}

#[cfg(not(feature = "dyntensor"))]
fn main() {
    // should noise be applied to the input?
    const USE_NOISE: bool = true;
    // how much noise should be applied to the input?
    const NOISE_STD: TensorFloat = 0.1;

    // XOR base set
    let base_inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let base_targets = [0.0, 1.0, 1.0, 0.0];

    // model: 2 → 4 → 1
    let mut w1 = Tensor::new(&[2, 4], &[0.5, -0.2, 0.3, -0.4, -0.3, 0.8, -0.1, 0.7]).with_grad();
    let mut w2 = Tensor::new(&[4, 2], &[0.1, -0.3, 0.6, -0.1, -0.3, -0.8, -0.2, -0.5]).with_grad();

    let lr = 0.001;
    let epochs = 1000;
    let mut rng = rand::rng();

    let noise_dist = Normal::new(0.0, NOISE_STD).unwrap();

    let add_noise =
        |x: [TensorFloat; 2], rng: &mut _, dist: &Normal<TensorFloat>| -> [TensorFloat; 2] {
            if USE_NOISE {
                [x[0] + dist.sample(rng), x[1] + dist.sample(rng)]
            } else {
                x
            }
        };

    for epoch in 0..epochs {
        let mut loss_accum = 0.0;
        let mut correct_train = 0;

        // shuffle training order
        let mut idx = [0, 1, 2, 3];
        idx.shuffle(&mut rng);

        for &i in &idx {
            let noise_x = add_noise(base_inputs[i], &mut rng, &noise_dist);
            let x0 = Tensor::new(&[1, 2], &noise_x);
            let y = Tensor::new(&[1, 2], &[base_targets[i], base_targets[i]]);

            let x = x0.clone().with_grad();

            // forward pass
            let (z1, back1) = matmul(&x, &w1);
            let z1_wrapped = z1.clone().with_grad();
            let (a1, back_relu) = relu(&z1_wrapped);
            let a1_wrapped = a1.clone().with_grad();
            let (z2, back2) = matmul::<4, 8, 2>(&a1_wrapped, &w2);
            let prediction = z2.clone().with_grad();

            // loss
            let (loss, back_loss) = mse_loss(&prediction, &y);
            loss_accum += loss;

            // accuracy
            let pred_val = prediction.get_value().data()[0];
            if (pred_val >= 0.5 && y.data()[0] == 1.0) || (pred_val < 0.5 && y.data()[0] == 0.0) {
                correct_train += 1;
            }

            // backprop
            #[cfg(feature = "alloc")]
            let dloss = back_loss(1.0);
            #[cfg(not(feature = "alloc"))]
            let dloss = back_loss.call(1.0);
            #[cfg(feature = "alloc")]
            let (dz2_a1, dz2_w2) = back2(dloss);
            #[cfg(not(feature = "alloc"))]
            let (dz2_a1, dz2_w2) = back2.call(dloss);
            #[cfg(feature = "alloc")]
            let d_relu = back_relu(dz2_a1);
            #[cfg(not(feature = "alloc"))]
            let d_relu = back_relu.call(dz2_a1);
            #[cfg(feature = "alloc")]
            let (_, dz1_w1) = back1(d_relu);
            #[cfg(not(feature = "alloc"))]
            let (_, dz1_w1) = back1.call(d_relu);

            let grads_w1 = dz1_w1.data().to_vec();
            let grads_w2 = dz2_w2.data().to_vec();

            drop(back1);
            drop(back2);

            // accumulate grads
            for (g, val) in w1.get_grad_mut().data_mut().iter_mut().zip(grads_w1) {
                *g += val;
            }
            for (g, val) in w2.get_grad_mut().data_mut().iter_mut().zip(grads_w2) {
                *g += val;
            }
        }

        // SGD update
        sgd(&mut w1, lr);
        sgd(&mut w2, lr);

        // test loop
        let mut test_loss = 0.0;
        let mut correct_test = 0;
        for i in 0..4 {
            let noise_x = add_noise(base_inputs[i], &mut rng, &noise_dist);
            let target = (noise_x[0].round() as u8 ^ noise_x[1].round() as u8) as TensorFloat;

            let x = Tensor::new(&[1, 2], &noise_x);
            let xg = x.clone().with_grad();

            let (z1, _) = matmul::<2, 8, 4>(&xg, &w1);
            let (a1, _) = relu(&z1.clone().with_grad());
            let (z2, _) = matmul::<4, 8, 2>(&a1.clone().with_grad(), &w2);

            let pred_val = z2.data()[0];
            let loss = (pred_val - target).powi(2);
            test_loss += loss;

            if (pred_val >= 0.5 && target == 1.0) || (pred_val < 0.5 && target == 0.0) {
                correct_test += 1;
            }
        }

        if (epoch + 1) % 100 == 0 {
            println!(
                "Epoch {:4}: train_loss={:.6} train_acc={:.2}% test_loss={:.6} test_acc={:.2}%",
                epoch + 1,
                loss_accum / 4.0,
                100.0 * correct_train as TensorFloat / 4.0,
                test_loss / 4.0,
                100.0 * correct_test as TensorFloat / 4.0
            );
        }
    }

    #[cfg(feature = "std")]
    {
        use briny_ai::manual::tensors::TensorGrad;
        briny_ai::manual::diskio::save_tensors(
            "xor_model.bpat",
            &[w1.get_value().clone(), w2.get_value().clone()],
        )
        .unwrap();
        println!("Model saved to xor_model.bpat");

        let loaded =
            briny_ai::manual::diskio::load_tensors::<TensorFloat>("xor_model.bpat").unwrap();
        println!("Loaded tensors:");
        for t in loaded {
            println!(
                "Shape: {:?}, First few vals: {:?}",
                t.shape(),
                &t.data()[..4.min(t.len())]
            );
        }
    }
}
