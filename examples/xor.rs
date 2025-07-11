use briny_ai::tensor;
use briny_ai::tensors::{Tensor, WithGrad};
use briny_ai::backprop::{matmul, relu, mse_loss, sgd};
use briny_ai::modelio::{save_model, load_model};

fn main() {
    briny_ai::backend::set_backend(briny_ai::backend::Backend::Wgpu);
    // XOR input and output
    let inputs = tensor!([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ]);
    let targets = tensor!([[0.0], [1.0], [1.0], [0.0]]);

    // model: 2-input → hidden-4 → output-1
    let mut w1 = WithGrad {
        value: Tensor::new(vec![2, 4], vec![
            0.5, -0.2, 0.3, -0.4,
            -0.3, 0.8, -0.1, 0.7
        ]),
        grad: Tensor::new(vec![2, 4], vec![0.0; 8]),
    };
    let mut w2 = WithGrad {
        value: Tensor::new(vec![4, 1], vec![0.1, -0.3, 0.6, -0.1]),
        grad: Tensor::new(vec![4, 1], vec![0.0; 4]),
    };

    let lr = 0.1;
    let epochs = 1000;

    for epoch in 0..epochs {
        let mut loss_accum = 0.0;

        for i in 0..4 {
            // prepare one training example
            let x = Tensor::new(vec![1, 2], inputs.data[i * 2..i * 2 + 2].to_vec());
            let y = Tensor::new(vec![1, 1], vec![targets.data[i]]);

            let x = WithGrad {
                value: x.clone(),
                grad: Tensor::new(x.shape.clone(), vec![0.0; 2]),
            };

            // forward pass
            let (z1, back1) = matmul(&x, &w1);
            let z1_wrapped = WithGrad {
                value: z1.clone(),
                grad: Tensor::new(z1.shape.clone(), vec![0.0; z1.data.len()]),
            };
            let (a1, back_relu) = relu(&z1_wrapped);
            let a1_wrapped = WithGrad {
                value: a1.clone(),
                grad: Tensor::new(a1.shape.clone(), vec![0.0; a1.data.len()]),
            };
            let (z2, back2) = matmul(&a1_wrapped, &w2);
            let prediction = WithGrad {
                value: z2.clone(),
                grad: Tensor::new(z2.shape.clone(), vec![0.0; z2.data.len()]),
            };

            // compute loss
            let (loss, back_loss) = mse_loss(&prediction, &y);
            loss_accum += loss;

            // backward pass
            let dloss = back_loss(1.0);
            let (dz2_a1, dz2_w2) = back2(&dloss);
            let d_relu = back_relu(&dz2_a1);
            let (_, dz1_w1) = back1(&d_relu);

            // extract gradient data early to end borrows
            let grads_w1 = dz1_w1.data;
            let grads_w2 = dz2_w2.data;

            drop(back1);
            drop(back2);

            // accumulate grads
            for (g, val) in w1.grad.data.iter_mut().zip(grads_w1) {
                *g += val;
            }
            for (g, val) in w2.grad.data.iter_mut().zip(grads_w2) {
                *g += val;
            }
        }

        // apply SGD update
        sgd(&mut w1, lr);
        sgd(&mut w2, lr);

        if epoch % 100 == 0 {
            println!("Epoch {epoch}: Loss = {:.6}", loss_accum / 4.0);
        }
    }

    // save model
    save_model("xor_model.bpat", &[w1.value.clone(), w2.value.clone()]).unwrap();
    println!("Model saved to xor_model.bpat");

    // load and inspect
    let loaded = load_model("xor_model.bpat").unwrap();
    println!("Loaded tensors:");
    for t in loaded {
        println!("Shape: {:?}, First few vals: {:?}", t.shape, &t.data[..4.min(t.data.len())]);
    }
}
