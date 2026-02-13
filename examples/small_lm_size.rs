use briny_ai::prelude::*;
use std::{hint::black_box, time::Instant};

const SEQ: usize = 512;
const VOCAB: usize = 40478;
const D_MODEL: usize = 768;
const D_FF: usize = 4 * D_MODEL;

const DEPTH: usize = 12;
const PARAMETERS: usize =
    (VOCAB * D_MODEL) + (DEPTH * ((4 * D_MODEL * D_MODEL) + (2 * D_MODEL * D_FF)));

const PAD: [f32; VOCAB] = [0.0; VOCAB];

static INPUTS: [[f32; VOCAB]; SEQ] = [PAD; SEQ];
static TARGETS: [[f32; VOCAB]; SEQ] = [PAD; SEQ];

static_model!(
    @loss cross_entropy_loss
    @optim Adam(0.001)
    @model SeqTransformer(model)
    {
        InputLayer([SEQ, VOCAB]),
        {
            // project vocab to features
            embed: Collapse([VOCAB, D_MODEL]) => CollapseLayer([SEQ, D_MODEL]),

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

            // transformer 2
            ln4: LayerNorm([1, D_MODEL]) => LayerNormLayer,
            attn2: Residual([D_MODEL, D_MODEL], <CausalSelfAttention>) => ResidualLayer(a[SEQ, SEQ], [SEQ, D_MODEL]),
            ln5: LayerNorm([1, D_MODEL]) => LayerNormLayer,
            ff2: Residual([D_MODEL, D_FF], <FeedForward>, Swish) => ResidualLayer(a[SEQ, D_FF], [SEQ, D_MODEL]),

            // transformer 3
            ln6: LayerNorm([1, D_MODEL]) => LayerNormLayer,
            attn3: Residual([D_MODEL, D_MODEL], <CausalSelfAttention>) => ResidualLayer(a[SEQ, SEQ], [SEQ, D_MODEL]),
            ln7: LayerNorm([1, D_MODEL]) => LayerNormLayer,
            ff3: Residual([D_MODEL, D_FF], <FeedForward>, GELU) => ResidualLayer(a[SEQ, D_FF], [SEQ, D_MODEL]),

            // transformer 4
            ln8: LayerNorm([1, D_MODEL]) => LayerNormLayer,
            attn4: Residual([D_MODEL, D_MODEL], <CausalSelfAttention>) => ResidualLayer(a[SEQ, SEQ], [SEQ, D_MODEL]),
            ln9: LayerNorm([1, D_MODEL]) => LayerNormLayer,
            ff4: Residual([D_MODEL, D_FF], <FeedForward>, Swish) => ResidualLayer(a[SEQ, D_FF], [SEQ, D_MODEL]),

            // transformer 5
            ln10: LayerNorm([1, D_MODEL]) => LayerNormLayer,
            attn5: Residual([D_MODEL, D_MODEL], <CausalSelfAttention>) => ResidualLayer(a[SEQ, SEQ], [SEQ, D_MODEL]),
            ln11: LayerNorm([1, D_MODEL]) => LayerNormLayer,
            ff5: Residual([D_MODEL, D_FF], <FeedForward>, Swish) => ResidualLayer(a[SEQ, D_FF], [SEQ, D_MODEL]),

            // transformer 6
            ln12: LayerNorm([1, D_MODEL]) => LayerNormLayer,
            attn6: Residual([D_MODEL, D_MODEL], <CausalSelfAttention>) => ResidualLayer(a[SEQ, SEQ], [SEQ, D_MODEL]),
            ln13: LayerNorm([1, D_MODEL]) => LayerNormLayer,
            ff6: Residual([D_MODEL, D_FF], <FeedForward>, GELU) => ResidualLayer(a[SEQ, D_FF], [SEQ, D_MODEL]),

            // transformer 7
            ln14: LayerNorm([1, D_MODEL]) => LayerNormLayer,
            attn7: Residual([D_MODEL, D_MODEL], <CausalSelfAttention>) => ResidualLayer(a[SEQ, SEQ], [SEQ, D_MODEL]),
            ln15: LayerNorm([1, D_MODEL]) => LayerNormLayer,
            ff7: Residual([D_MODEL, D_FF], <FeedForward>, Swish) => ResidualLayer(a[SEQ, D_FF], [SEQ, D_MODEL]),

            // transformer 8
            ln16: LayerNorm([1, D_MODEL]) => LayerNormLayer,
            attn8: Residual([D_MODEL, D_MODEL], <CausalSelfAttention>) => ResidualLayer(a[SEQ, SEQ], [SEQ, D_MODEL]),
            ln17: LayerNorm([1, D_MODEL]) => LayerNormLayer,
            ff8: Residual([D_MODEL, D_FF], <FeedForward>, Swish) => ResidualLayer(a[SEQ, D_FF], [SEQ, D_MODEL]),

            // transformer 9
            ln18: LayerNorm([1, D_MODEL]) => LayerNormLayer,
            attn9: Residual([D_MODEL, D_MODEL], <CausalSelfAttention>) => ResidualLayer(a[SEQ, SEQ], [SEQ, D_MODEL]),
            ln19: LayerNorm([1, D_MODEL]) => LayerNormLayer,
            ff9: Residual([D_MODEL, D_FF], <FeedForward>, GELU) => ResidualLayer(a[SEQ, D_FF], [SEQ, D_MODEL]),

            // transformer 10
            ln20: LayerNorm([1, D_MODEL]) => LayerNormLayer,
            attn10: Residual([D_MODEL, D_MODEL], <CausalSelfAttention>) => ResidualLayer(a[SEQ, SEQ], [SEQ, D_MODEL]),
            ln21: LayerNorm([1, D_MODEL]) => LayerNormLayer,
            ff10: Residual([D_MODEL, D_FF], <FeedForward>, Swish) => ResidualLayer(a[SEQ, D_FF], [SEQ, D_MODEL]),

            // transformer 11
            ln22: LayerNorm([1, D_MODEL]) => LayerNormLayer,
            attn11: Residual([D_MODEL, D_MODEL], <CausalSelfAttention>) => ResidualLayer(a[SEQ, SEQ], [SEQ, D_MODEL]),
            ln23: LayerNorm([1, D_MODEL]) => LayerNormLayer,
            ff11: Residual([D_MODEL, D_FF], <FeedForward>, Swish) => ResidualLayer(a[SEQ, D_FF], [SEQ, D_MODEL]),

            // final normalization
            ln24: LayerNorm([1, D_MODEL]) => LayerNormLayer,

            // project features back
            extract: Collapse([D_MODEL, VOCAB]) => CollapseLayer,
            soft: Softmax([SEQ, VOCAB](1.5, 0.2)) => SoftmaxLayer,
        },
        OutputLayer([SEQ, VOCAB]),
    }
);

fn main() {
    let path = format!("checkpoints/test/llm{}.bpat", PARAMETERS);

    let dataset = Dataset::new_copy(&INPUTS, &TARGETS);

    let mut model = SeqTransformer::new(0.4);

    let start_load = Instant::now();
    let _ = model.load(&path);
    let elapsed_load = start_load.elapsed();

    println!("LOAD={:?}", elapsed_load);

    let start_save = Instant::now();
    let _ = model.save(&path, BpatHeader::BpatV1M);
    let elapsed_save = start_save.elapsed();

    println!("SAVE={:?}", elapsed_save);

    set_backend(Backend::Cpu);

    let start_cpu = Instant::now();
    let _ = black_box(model.fit_epoch(&dataset));
    let elapsed_cpu = start_cpu.elapsed();

    println!("CPU={:?}", elapsed_cpu);

    #[cfg(feature = "wgpu")]
    {
        set_backend(Backend::Wgpu);

        let start_wgpu = Instant::now();
        let _ = black_box(model.fit_epoch(&dataset));
        let elapsed_wgpu = start_wgpu.elapsed();

        println!("WGPU={:?}", elapsed_wgpu);
    }

    #[cfg(feature = "cuda")]
    {
        set_backend(Backend::Cuda);

        let start_cuda = Instant::now();
        let _ = black_box(model.fit_epoch(&dataset));
        let elapsed_cuda = start_cuda.elapsed();

        println!("CUDA={:?}", elapsed_cuda);
    }

    let _ = model.save(&path, BpatHeader::BpatV1M);
}
