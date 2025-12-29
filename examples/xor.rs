use briny_ai::prelude::*;

static_model!(
    @loss mse_loss
    @optim sgd
    @model XorModel
    {
        InputLayer([4, 2]),
        {
            conv0: Collapse([4, 2]) => CollapseLayer,
            act0: Activation([4, 1], ReLU) => ActivationLayer,
        },
        OutputLayer([4, 1]),
    }
);

// const PATH_TO_MODEL: &str = "checkpoints/xor/model.bpat";

fn main() {
    // GPU overhead is far too much for a small network; use CPU
    set_backend(Backend::Cpu); // default, but being explicit

    let base_inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let base_targets = [[0.0], [1.0], [1.0], [0.0]];

    let dataset = Dataset::new(&base_inputs, &base_targets);

    let mut model = XorModel::new(1.0).with_lr(0.01);

    #[cfg(feature = "std")]
    println!("Loading model...");

    // let _ = model.load(PATH_TO_MODEL);

    #[cfg(feature = "std")]
    println!("Beginning training...");

    let mut score = 0.0;
    let mut i = 0u128;

    while score != 100.0 {
        #[cfg_attr(not(feature = "std"), allow(unused_variables))]
        let loss = model.fit(&dataset, 10000000, decay_lr);

        #[cfg(feature = "std")]
        println!(
            "\n=== EPOCH {:?} ===\nTRAINING: loss={:?}, lr={:?}",
            i,
            loss,
            model.get_lr()
        );

        if i.is_multiple_of(10) {
            let eval = model.infer(&dataset);

            score = eval.score;

            #[cfg(feature = "std")]
            println!(
                "TESTING: loss={:?}, acc={:?}%, score={:?}%",
                eval.loss, eval.acc, score
            );

            // model.save(PATH_TO_MODEL).unwrap();
        }

        i += 1;
    }

    #[cfg(feature = "std")]
    println!("model reached 100% acc");
    // model.save(PATH_TO_MODEL).unwrap();
}
