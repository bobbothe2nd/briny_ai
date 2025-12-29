use briny_ai::prelude::*;

const SEQ: usize = 4;
const VOCAB: usize = 4;

static_model!(
    @loss cross_entropy_loss
    @optim sgd(0.01)
    @model CharRNN
    {
        InputLayer([SEQ, VOCAB]),
        {
            dense0: Collapse([VOCAB, VOCAB]) => CollapseLayer,
            act0: Activation([SEQ, VOCAB], Tanh) => ActivationLayer,
            dense1: Collapse([VOCAB, VOCAB]) => CollapseLayer,
            act1: Activation([SEQ, VOCAB], ReLU) => ActivationLayer,
            dense2: Collapse([VOCAB, VOCAB]) => CollapseLayer,
            soft: Softmax([SEQ, VOCAB](1.1, 1.0)) => SoftmaxLayer,
        },
        OutputLayer([SEQ, VOCAB]),
    }
);

// const PATH_TO_MODEL: &str = "checkpoints/char_rnn/model.bpat";

fn main() {
    let inputs = [
        [1.0, 0.0, 0.0, 0.0], // H
        [0.0, 1.0, 0.0, 0.0], // E
        [0.0, 0.0, 1.0, 0.0], // L
        [0.0, 0.0, 1.0, 0.0], // L
    ];

    let targets = [
        [0.0, 1.0, 0.0, 0.0], // E
        [0.0, 0.0, 1.0, 0.0], // L
        [0.0, 0.0, 1.0, 0.0], // L
        [0.0, 0.0, 0.0, 1.0], // O
    ];

    let dataset = Dataset::new(&inputs, &targets);

    let mut model = CharRNN::new(1.0);

    #[cfg(feature = "std")]
    println!("Loading model...");

    // let _ = model.load(PATH_TO_MODEL);

    #[cfg(feature = "std")]
    println!("Beginning training...");

    let mut i = 0u128;
    loop {
        #[cfg(feature = "std")]
        println!("\n=== EPOCH {:?} ===", i);

        #[cfg_attr(not(feature = "std"), allow(unused_variables))]
        let loss = model.fit(&dataset, 50000, adapt_lr);

        #[cfg(feature = "std")]
        println!("TRAINING: loss={:?}, lr={:?}", loss, model.get_lr());

        if i.is_multiple_of(10) {
            #[cfg_attr(not(feature = "std"), allow(unused_variables))]
            let eval = model.infer(&dataset);

            #[cfg(feature = "std")]
            println!(
                "TESTING: loss={:?}, acc={:?}%, score={:?}%",
                eval.loss, eval.acc, eval.score,
            );

            // model.save(PATH_TO_MODEL).unwrap();
        }

        i += 1;
    }
}
