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
            dense0: Dense([VOCAB, VOCAB]) => DenseLayer,
            act0: Activation([SEQ, VOCAB], ReLU) => ActivationLayer,
            dense1: Dense([VOCAB, VOCAB]) => DenseLayer,
            soft: Activation([SEQ, VOCAB], Softmax) => ActivationLayer,
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

    let mut model = CharRNN::new();

    println!("Loading model...");

    // let _ = model.load(PATH_TO_MODEL);

    println!("Beginning training...");

    let mut i = 0u128;
    loop {
        println!("\n=== BATCH {:?} ===", i);

        let loss = model.fit(&dataset, 500000);

        println!(
            "TRAINING: loss={:?}, lr={:?}",
            loss,
            model.get_lr()
        );

        if i.is_multiple_of(10) {
            let eval = model.infer(&dataset);

            println!(
                "TESTING: loss={:?}, accuracy={:?}%, score={:?}%",
                eval.loss, eval.accuracy, eval.score,
            );

            // model.save(PATH_TO_MODEL).unwrap();
        }

        i += 1;
    }
}
