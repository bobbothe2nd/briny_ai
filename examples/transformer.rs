#![cfg_attr(feature = "f64", allow(dead_code, unused))]

use core::mem::MaybeUninit;
use briny_ai::prelude::{Context, Dataset, Flatten, TestEval, TensorOps, static_model};

const SEQ: usize = 9;
const D_MODEL: usize = 12;
const VOCAB: usize = 9;

const TOKENS: [[f32; VOCAB]; VOCAB] = [
    THE, QUICK, BROWN, FOX, JUMPED, OVER, LAZY, DOG, STOP_SAMPLE,
];
const TOKENS_STR: [&str; VOCAB] = [
    "THE", "QUICK", "BROWN", "FOX", "JUMPED", "OVER", "LAZY", "DOG", "[eof]",
];

const TOKEN_POS: f32 = 0.9;
const TOKEN_NEG: f32 = (1.0 - TOKEN_POS) / (VOCAB - 1) as f32;

const PAD: [f32; VOCAB] = [0.0; VOCAB];
const THE: [f32; VOCAB] = [TOKEN_POS, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG];
const QUICK: [f32; VOCAB] = [TOKEN_NEG, TOKEN_POS, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG];
const BROWN: [f32; VOCAB] = [TOKEN_NEG, TOKEN_NEG, TOKEN_POS, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG];
const FOX: [f32; VOCAB] = [TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_POS, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG];
const JUMPED: [f32; VOCAB] = [TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_POS, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG];
const OVER: [f32; VOCAB] = [TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_POS, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG];
const LAZY: [f32; VOCAB] = [TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_POS, TOKEN_NEG, TOKEN_NEG];
const DOG: [f32; VOCAB] = [TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_POS, TOKEN_NEG];
const STOP_SAMPLE: [f32; VOCAB] = [TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_NEG, TOKEN_POS];

const DATASET_COUNT: usize = 9;
const DESIRED_LOSS: f32 = 1.2;

const INPUTS: [[[f32; VOCAB]; SEQ]; DATASET_COUNT] = [
    [PAD, PAD, PAD, PAD, PAD, PAD, PAD, PAD, THE],
    [PAD, PAD, PAD, PAD, PAD, PAD, PAD, THE, QUICK],
    [PAD, PAD, PAD, PAD, PAD, PAD, THE, QUICK, BROWN],
    [PAD, PAD, PAD, PAD, PAD, THE, QUICK, BROWN, FOX],
    [PAD, PAD, PAD, PAD, THE, QUICK, BROWN, FOX, JUMPED],
    [PAD, PAD, PAD, THE, QUICK, BROWN, FOX, JUMPED, OVER],
    [PAD, PAD, THE, QUICK, BROWN, FOX, JUMPED, OVER, THE],
    [PAD, THE, QUICK, BROWN, FOX, JUMPED, OVER, THE, LAZY],
    [THE, QUICK, BROWN, FOX, JUMPED, OVER, THE, LAZY, DOG],
];

const TARGETS: [[[f32; VOCAB]; SEQ]; DATASET_COUNT] = [
    [PAD, PAD, PAD, PAD, PAD, PAD, PAD, THE, QUICK],
    [PAD, PAD, PAD, PAD, PAD, PAD, THE, QUICK, BROWN],
    [PAD, PAD, PAD, PAD, PAD, THE, QUICK, BROWN, FOX],
    [PAD, PAD, PAD, PAD, THE, QUICK, BROWN, FOX, JUMPED],
    [PAD, PAD, PAD, THE, QUICK, BROWN, FOX, JUMPED, OVER],
    [PAD, PAD, THE, QUICK, BROWN, FOX, JUMPED, OVER, THE],
    [PAD, THE, QUICK, BROWN, FOX, JUMPED, OVER, THE, LAZY],
    [THE, QUICK, BROWN, FOX, JUMPED, OVER, THE, LAZY, DOG],
    [QUICK, BROWN, FOX, JUMPED, OVER, THE, LAZY, DOG, STOP_SAMPLE],
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
        if tok.iter().all(|x| *x == 0.0) {
            continue;
        }

        string.push_str(tok_to_str(tok));
        string.push(' ');
    }
    string.truncate(string.len() - 1);
    string
}

static_model!(
    @loss cross_entropy_nonzero_loss
    @optim sgd(0.001)
    @model SeqTransformer
    {
        InputLayer([VOCAB, SEQ]),
        {
            // project vocab to features
            dense0: Dense([D_MODEL, VOCAB]) => DenseLayer,
            act0: Activation([D_MODEL, SEQ], Tanh) => ActivationLayer,

            // mix features
            dense1: Dense([D_MODEL, D_MODEL]) => DenseLayer,
            act1: Activation([D_MODEL, SEQ], ReLU) => ActivationLayer,

            // grammar / sequence update
            grammar: Collapse([SEQ, SEQ]) => CollapseLayer,
            act2: Activation([D_MODEL, SEQ], GELU) => ActivationLayer,

            // alter through time series
            attn0: Temporal([SEQ, SEQ]) => TemporalLayer([D_MODEL, SEQ]),
            collapse: Collapse([SEQ, SEQ]) => CollapseLayer([D_MODEL, SEQ]),
            attn1: Temporal([SEQ, SEQ]) => TemporalLayer([D_MODEL, SEQ]),
            act3: Activation([D_MODEL, SEQ], Tanh) => ActivationLayer,

            // project features back
            dense2: Dense([VOCAB, D_MODEL]) => DenseLayer,
            soft: Softmax([VOCAB, SEQ](2.0, 0.5)) => SoftmaxLayer,
        },
        OutputLayer([VOCAB, SEQ]),
    }
);

#[cfg(feature = "f64")]
fn main() {
    println!("`SeqTransformer` not supported using `f64");
}

#[cfg(not(feature = "f64"))]
fn main() {
    let mut model = SeqTransformer::new(0.75);

    // training on "THE QUICK BROWN FOX JUMPED OVER THE LAZY DOG"
    let mut datasets_maybe = MaybeUninit::<[Dataset<2, { VOCAB * SEQ }, 2, { VOCAB * SEQ }>; DATASET_COUNT]>::uninit();
    for (i, (input, target)) in INPUTS.iter().zip(&TARGETS).enumerate() {
        let dataset = Dataset::new(input, target);
        unsafe {
            datasets_maybe.as_mut_ptr().cast::<Dataset<2, { VOCAB * SEQ }, 2, { VOCAB * SEQ }>>().add(i).write(dataset);
        }
    }
    let datasets = unsafe { datasets_maybe.assume_init() };

    println!("TRAINING:");

    let mut min_expected = 2.4;
    let mut b_i = 0;
    'training: while min_expected > DESIRED_LOSS {
        println!(" BATCH {:?}:", b_i);

        let mut i = 0;
        loop {
            if i % 10000 == 0 {
                println!("  EPOCH {:?}:", i);
                if i % 100000 == 0 {
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
                break 'training;
            }

            if i % 10000 == 0 {
                println!("   TRAINING: loss=({:?}..{:?}), lr={:?}", train_loss_low, train_loss_high, model.get_lr());
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

            if i % 10000 == 0 {
                println!(
                    "   TESTING: loss=({:?}..{:?}), acc=({:?}..{:?})%, score=({:?}..{:?})%",
                    eval_low.loss, eval_high.loss, low_acc, high_acc, low_score, high_score
                );
            }

            if eval_high.loss <= min_expected {
                break;
            }

            i += 1;
        }
        min_expected *= 0.95;
        b_i += 1;
    }

    println!("\nTESTING:\n");

    let mut idx = 0;
    let text = [THE, PAD, PAD, PAD, PAD, PAD, PAD, PAD, PAD];
    let text_flat: [f32; VOCAB * SEQ] = text.flatten();
    let mut text_data = Context::from_parts([VOCAB, SEQ], text_flat);
    let mut nested_context = text;
    loop {
        idx += 1;
        let idx_scaled = idx * VOCAB;
        let sample = model.sample(&text_data);
        let next_tok = match_distr_tok(&sample[(sample.len() - VOCAB)..].try_into().unwrap());
        println!("{:?} -> {:?}", &sample[(sample.len() - VOCAB)..], tok_to_str(&next_tok));
        if next_tok == STOP_SAMPLE || idx >= SEQ {
            break;
        }
        text_data.raw_mut().get_value_mut().data_mut()[idx_scaled..idx_scaled+VOCAB].copy_from_slice(&next_tok);
        nested_context[idx] = next_tok;
    }
    println!("\n{}", tok_stream_to_str(&nested_context));
}

#[cfg(not(feature = "f64"))]
fn test(model: &SeqTransformer) {
    println!("   SAMPLES:");
    let mut idx = 0;
    let text = [THE, PAD, PAD, PAD, PAD, PAD, PAD, PAD, PAD];
    let text_flat: [f32; VOCAB * SEQ] = text.flatten();
    let mut text_data = Context::from_parts([VOCAB, SEQ], text_flat);
    let mut nested_context = text;
    loop {
        idx += 1;
        let idx_scaled = idx * VOCAB;
        let sample = model.sample(&text_data);
        let next_tok = match_distr_tok(&sample[(sample.len() - VOCAB)..].try_into().unwrap());
        println!("     {:?} -> {:?}", &sample[(sample.len() - VOCAB)..], tok_to_str(&next_tok));
        if next_tok == STOP_SAMPLE || idx >= SEQ {
            break;
        }
        text_data.raw_mut().get_value_mut().data_mut()[idx_scaled..idx_scaled+VOCAB].copy_from_slice(&next_tok);
        nested_context[idx] = next_tok;
    }
    println!("    {}", tok_stream_to_str(&nested_context));
}
