use briny_ai::prelude::*;

const PATH_TO_MODEL: &str = "checkpoints/transformer/model.bpat";

const SEQ: usize = 8;
const VOCAB: usize = 9;
const D_MODEL: usize = 6;
const D_FF: usize = D_MODEL * 4;

const DEPTH: usize = 2;
const PARAMETERS: usize =
    (VOCAB * D_MODEL) + (DEPTH * ((4 * D_MODEL * D_MODEL) + (2 * D_MODEL * D_FF)));

const CONTEXT_LIMIT: usize = 10;

const TOKENS: [[f32; VOCAB]; VOCAB] =
    [THE, QUICK, BROWN, FOX, JUMPED, OVER, LAZY, DOG, STOP_SAMPLE];
const TOKENS_STR: [&str; VOCAB] = [
    "THE", "QUICK", "BROWN", "FOX", "JUMPED", "OVER", "LAZY", "DOG", "[eof]",
];

const TOKEN_POS: f32 = 0.9;
const TOKEN_NEG: f32 = (1.0 - TOKEN_POS) / (VOCAB - 1) as f32;

const PAD: [f32; VOCAB] = [0.0; VOCAB];
const THE: [f32; VOCAB] = [
    TOKEN_POS, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG,
    TOKEN_NEG,
];
const QUICK: [f32; VOCAB] = [
    TOKEN_NEG, TOKEN_POS, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG,
    TOKEN_NEG,
];
const BROWN: [f32; VOCAB] = [
    TOKEN_NEG, TOKEN_NEG, TOKEN_POS, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG,
    TOKEN_NEG,
];
const FOX: [f32; VOCAB] = [
    TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_POS, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG,
    TOKEN_NEG,
];
const JUMPED: [f32; VOCAB] = [
    TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_POS, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG,
    TOKEN_NEG,
];
const OVER: [f32; VOCAB] = [
    TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_POS, TOKEN_NEG, TOKEN_NEG,
    TOKEN_NEG,
];
const LAZY: [f32; VOCAB] = [
    TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_POS, TOKEN_NEG,
    TOKEN_NEG,
];
const DOG: [f32; VOCAB] = [
    TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_POS,
    TOKEN_NEG,
];
const STOP_SAMPLE: [f32; VOCAB] = [
    TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG,
    TOKEN_POS,
];

const DATASET_COUNT: usize = 4;
const DESIRED_LOSS: f32 = 0.3;

const INPUTS: [[[f32; VOCAB]; SEQ]; DATASET_COUNT] = [
    [THE, QUICK, BROWN, FOX, JUMPED, OVER, THE, LAZY],
    [QUICK, BROWN, FOX, JUMPED, OVER, THE, LAZY, DOG],
    [DOG, JUMPED, PAD, PAD, PAD, PAD, PAD, PAD],
    [JUMPED, OVER, PAD, PAD, PAD, PAD, PAD, PAD],
];

const TARGETS: [[[f32; VOCAB]; SEQ]; DATASET_COUNT] = [
    [QUICK, BROWN, FOX, JUMPED, OVER, THE, LAZY, DOG],
    [BROWN, FOX, JUMPED, OVER, THE, LAZY, DOG, STOP_SAMPLE],
    [DOG, JUMPED, OVER, PAD, PAD, PAD, PAD, PAD],
    [JUMPED, OVER, FOX, PAD, PAD, PAD, PAD, PAD],
];

fn match_distr_tok(distr: &[f32; VOCAB]) -> [f32; VOCAB] {
    fn closest_token(p: &[f32; VOCAB]) -> usize {
        p.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(core::cmp::Ordering::Less))
            .unwrap_or((0, &0.0))
            .0
    }

    let idx = closest_token(distr);
    TOKENS[idx]
}

fn tok_to_str(tok: &[f32; VOCAB]) -> &'static str {
    let idx = TOKENS.iter().position(|x| x == tok).unwrap();
    TOKENS_STR[idx]
}

fn tok_stream_to_str(toks: &[[f32; VOCAB]]) -> String {
    let mut string = String::new();
    for tok in toks {
        string.push_str(tok_to_str(tok));
        string.push(' ');
    }
    string.truncate(string.len() - 1);
    string
}

static_model!(
    @loss cross_entropy_loss
    @optim Adam(0.001)
    @model SeqTransformer(model)
    {
        InputLayer([SEQ, VOCAB]),
        {
            // project vocab to features
            embed: Collapse([VOCAB, D_MODEL]) => CollapseLayer,

            // transformer 0
            ln0: LayerNorm([1, D_MODEL]) => LayerNormLayer,
            attn0: Residual([D_MODEL, D_MODEL], <CausalSelfAttention>) => ResidualLayer(a[SEQ, SEQ], [SEQ, D_MODEL]),
            ln1: LayerNorm([1, D_MODEL]) => LayerNormLayer,
            ff0: Residual([D_MODEL, D_FF], <FeedForward>, GELU) => ResidualLayer(a[SEQ, D_FF], [SEQ, D_MODEL]),

            // transformer 1
            ln2: LayerNorm([1, D_MODEL]) => LayerNormLayer,
            attn1: Residual([D_MODEL, D_MODEL], <CausalSelfAttention>) => ResidualLayer(a[SEQ, SEQ], [SEQ, D_MODEL]),
            ln3: LayerNorm([1, D_MODEL]) => LayerNormLayer,
            ff1: Residual([D_MODEL, D_FF], <FeedForward>, Swish) => ResidualLayer(a[SEQ, D_FF], [SEQ, D_MODEL]),

            // final normalization
            ln4: LayerNorm([1, D_MODEL]) => LayerNormLayer,

            // project features back
            extract: Collapse([D_MODEL, VOCAB]) => CollapseLayer,
            soft: Softmax([SEQ, VOCAB](1.5, 0.2)) => SoftmaxLayer,
        },
        OutputLayer([SEQ, VOCAB]),
    }
);

fn main() {
    let mut model = SeqTransformer::new(0.4);

    println!("Loading Model\n {} Total Parameters", PARAMETERS);

    if model.load(PATH_TO_MODEL).is_ok() {
        println!(" Loaded Successfully");
    } else {
        println!(" Loading Failed");
    }

    // training on "THE QUICK BROWN FOX JUMPED OVER THE LAZY DOG"
    let datasets = Dataset::arr_of(INPUTS, TARGETS);

    println!("TRAINING:");

    let mut min_expected = 2.4;
    let mut b_i = 0;
    'training: while min_expected > DESIRED_LOSS {
        println!(" GROUP {:?}:", b_i);

        let mut i = 0;
        loop {
            if i % 1000 == 0 {
                println!("  EPOCH {:?}:", i);
                if i % 10000 == 0 {
                    test(&model);
                }
            }

            let mut train_loss_low = f32::INFINITY;
            let mut train_loss_high = 0.0;
            let mut eval_low = TestEval::INF_LOSS;
            let mut eval_high = TestEval::new();
            for dataset in &datasets {
                let loss = model.fit_epoch(dataset);

                if loss > train_loss_high {
                    train_loss_high = loss;
                }
                if loss < train_loss_low {
                    train_loss_low = loss;
                }

                let eval = model.infer(&dataset);

                if eval.is_better_than(&eval_low) {
                    eval_low = eval;
                }
                if eval_high.is_better_than(&eval) {
                    eval_high = eval;
                }
            }

            if train_loss_high <= f32::EPSILON {
                println!("  EPOCH {}:", i);
                println!(
                    "   training closed: model collapse (loss={:?})",
                    train_loss_high
                );
                break 'training;
            }

            let (low_acc, high_acc) = if eval_low.acc < eval_high.acc {
                (eval_low.acc, eval_high.acc)
            } else {
                (eval_high.acc, eval_low.acc)
            };
            let (low_score, high_score) = if eval_low.score < eval_high.score {
                (eval_low.score, eval_high.score)
            } else {
                (eval_high.score, eval_low.score)
            };

            if eval_high.loss < min_expected {
                println!("  EPOCH {}:", i);
                println!(
                    "   TRAINING: loss=({:?}..{:?}), lr={:?}",
                    train_loss_low,
                    train_loss_high,
                    model.get_lr()
                );
                model.save(PATH_TO_MODEL, BpatHeader::BpatV1).unwrap();
                println!(
                    "   TESTING: loss=({:?}..{:?}), acc=({:?}..{:?})%, score=({:?}..{:?})%",
                    eval_low.loss, eval_high.loss, low_acc, high_acc, low_score, high_score
                );
                println!("   batch closed: {:?} < {:?}", eval_high.loss, min_expected);
                break;
            } else if i % 1000 == 0 {
                println!(
                    "   TRAINING: loss=({:?}..{:?}), lr={:?}",
                    train_loss_low,
                    train_loss_high,
                    model.get_lr()
                );
                model.save(PATH_TO_MODEL, BpatHeader::BpatV1).unwrap();
                println!(
                    "   TESTING: loss=({:?}..{:?}), acc=({:?}..{:?})%, score=({:?}..{:?})%",
                    eval_low.loss, eval_high.loss, low_acc, high_acc, low_score, high_score
                );
            }

            i += 1;
        }
        min_expected *= 0.95;
        b_i += 1;
    }

    println!("FINAL EVALUATION:");
    let text = [THE, PAD, PAD, PAD, PAD, PAD, PAD, PAD];
    let text_flat: [f32; VOCAB * SEQ] = text.flatten();
    let mut text_data = Context::from_parts([VOCAB, SEQ], text_flat);
    let mut nested_context = Vec::with_capacity(SEQ);
    nested_context.push(THE);
    let mut idx = 0;
    loop {
        let sample = model.sample(&text_data);
        let next_tok = match_distr_tok(&sample[(sample.len() - VOCAB)..].try_into().unwrap());
        println!(
            "  {:?} -> {:?}",
            &sample[(sample.len() - VOCAB)..],
            tok_to_str(&next_tok)
        );
        if next_tok == STOP_SAMPLE {
            break;
        } else if idx > CONTEXT_LIMIT {
            println!("  FORCE ABORT (LENGTH EXCEEDING LIMIT)");
            break;
        }
        let text_mut = text_data.raw_mut().get_value_mut().data_mut();
        for i in 0..SEQ * VOCAB - 1 {
            text_mut[i] = text_mut[i + 1];
        }
        text_mut[SEQ * (VOCAB - 1)..].copy_from_slice(&next_tok);
        nested_context.push(next_tok);
        idx += 1;
    }
    println!(" {}", tok_stream_to_str(&nested_context));
}

fn test(model: &SeqTransformer) {
    print!("   SAMPLES:\n     ");
    let text = [THE, PAD, PAD, PAD, PAD, PAD, PAD, PAD];
    let text_flat: [f32; VOCAB * SEQ] = text.flatten();
    let mut text_data = Context::from_parts([SEQ, VOCAB], text_flat);
    let mut nested_context = Vec::with_capacity(SEQ);
    nested_context.push(THE);
    let mut idx = 0;
    let mut overflow = 0;
    loop {
        if overflow > CONTEXT_LIMIT {
            println!("\n     FORCE ABORT (LENGTH EXCEEDING LIMIT)");
            break;
        }

        let sample = model.sample(&text_data);
        let next_tok = match_distr_tok(&sample[(sample.len() - VOCAB)..].try_into().unwrap());

        print!("{} ", tok_to_str(&next_tok));

        if next_tok == STOP_SAMPLE {
            break;
        }
        let text_mut = text_data.raw_mut().get_value_mut().data_mut();

        idx += 1;
        if idx >= SEQ {
            idx -= 1;
            overflow += 1;
            for i in 0..SEQ * (VOCAB - 1) {
                text_mut[i] = text_mut[i + 1];
            }
        }

        text_mut[VOCAB * idx..VOCAB * (idx + 1)].copy_from_slice(&next_tok);
        nested_context.push(next_tok);
    }
    print!("\n");
}
