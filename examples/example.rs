use briny_ai::{
    tensors::{Tensor, WithGrad},
    backprop::{matmul, mse_loss, sgd},
    modelio::{save_model, load_model},
};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // prepare training data: y = 2x + 1
    let x_data = vec![1.0, 2.0, 3.0, 4.0];
    let y_data = vec![3.0, 5.0, 7.0, 9.0];

    let x = Tensor::new(vec![4, 1], x_data);
    let y = Tensor::new(vec![4, 1], y_data);

    // wrap inputs (no gradients needed for input)
    let x_wg = WithGrad {
        value: x,
        grad: Tensor::new(vec![4, 1], vec![0.0; 4]),
    };

    // initialize weights and bias with zero gradients
    let mut w = WithGrad {
        value: Tensor::new(vec![1, 1], vec![0.5]),
        grad: Tensor::new(vec![1, 1], vec![0.0]),
    };
    let mut b = WithGrad {
        value: Tensor::new(vec![1, 1], vec![0.0]),
        grad: Tensor::new(vec![1, 1], vec![0.0]),
    };

    let learning_rate = 0.1;
    let epochs = 1000;

    for epoch in 0..epochs {
        // forward: y_pred = x @ w + b
        let (xw, xw_back) = matmul(&x_wg, &w);

        let y_pred_data: Vec<f64> = xw
            .data
            .iter()
            .zip(&b.value.data)
            .map(|(&xwi, &bi)| xwi + bi)
            .collect();
        let y_pred = Tensor::new(vec![4, 1], y_pred_data);

        let y_pred_wg = WithGrad {
            value: y_pred,
            grad: Tensor::new(vec![4, 1], vec![0.0; 4]),
        };

        // compute loss and backprop closure
        let (loss, loss_back) = mse_loss(&y_pred_wg, &y);

        // backward pass
        let grad_pred = loss_back(1.0);

        // backprop through addition y_pred = xw + b
        let grad_b = grad_pred.data.iter().sum::<f64>();
        b.grad.data[0] = grad_b;

        // backprop through matmul x @ w
        let (_, grad_w) = xw_back(&grad_pred);
        // drop the closure explicitly to release borrows
        drop(xw_back);

        // accumulate gradient on w
        for (gw_i, w_grad_i) in grad_w.data.iter().zip(&mut w.grad.data) {
            *w_grad_i += *gw_i;
        }

        sgd(&mut w, learning_rate);
        sgd(&mut b, learning_rate);

        if epoch % 100 == 0 {
            println!("Epoch {:4} Loss {:.6}", epoch, loss);
        }
    }

    // save model
    save_model("model", &[w.value.clone(), b.value.clone()])?;

    // load model
    let loaded_tensors = load_model("model")?;
    println!("Loaded weights: {:?}", loaded_tensors);

    Ok(())
}
