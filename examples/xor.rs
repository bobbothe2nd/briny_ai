use briny_ai::prelude::*;

static_model!(
    @loss binary_cross_entropy_loss
    @optim Sgd
    @model XorModel(model)
    {
        InputLayer([4, 2]),
        {
            collapse0: Collapse([2, 4]) => CollapseLayer,
            act1: Activation([4, 4], Tanh) => ActivationLayer,
            collapse1: Collapse([4, 1]) => CollapseLayer,
            sigmoid: Sigmoid([4, 1]) => SigmoidLayer,
        },
        OutputLayer([4, 1]),
    }
);

const PATH_TO_MODEL: &str = "checkpoints/xor/model.bpat";

fn main() {
    let base_inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let base_targets = [[0.0], [1.0], [1.0], [0.0]];

    let dataset = Dataset::new(base_inputs, base_targets);

    let mut model = XorModel::new(0.4);

    println!("Loading model...");

    let _ = model.load(PATH_TO_MODEL);

    println!("Beginning training...");

    let mut score = 0.0;
    let mut i = 0u128;

    while score != 100.0 {
        #[cfg_attr(not(feature = "std"), allow(unused_variables))]
        let loss = model.fit(&dataset, 10000, adapt_lr);

        println!(
            "\n=== EPOCH {:?} ===\nTRAINING: loss={:?}, lr={:?}",
            i,
            loss,
            model.get_lr()
        );

        if i.is_multiple_of(100) {
            let eval = model.infer(&dataset);

            score = eval.score;

            println!(
                "TESTING: loss={:?}, acc={:?}%, score={:?}%",
                eval.loss, eval.acc, score
            );

            model.save(PATH_TO_MODEL, BpatHeader::BpatV1).unwrap();
        }

        i += 1;
    }

    println!("model reached 100% acc");
    model.save(PATH_TO_MODEL, BpatHeader::BpatV1).unwrap();
}
