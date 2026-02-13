use briny_ai::prelude::*;
use std::{hint::black_box, time::Instant};

const SEQ: usize = 48;
const VOCAB: usize = 16;
const D_MODEL: usize = 16;
const D_FF: usize = 16;

const AVG: u32 = 64;

const PAD: [f32; VOCAB] = [0.0; VOCAB];

const INPUTS: [[f32; VOCAB]; SEQ] = [PAD; SEQ];
const TARGETS: [[f32; VOCAB]; SEQ] = [PAD; SEQ];

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

            // transformer 2
            ln4: LayerNorm([1, D_MODEL]) => LayerNormLayer,
            attn2: Residual([D_MODEL, D_MODEL], <CausalSelfAttention>) => ResidualLayer(a[SEQ, SEQ], [SEQ, D_MODEL]),
            ln5: LayerNorm([1, D_MODEL]) => LayerNormLayer,
            ff2: Residual([D_MODEL, D_FF], <FeedForward>, Swish) => ResidualLayer(a[SEQ, D_FF], [SEQ, D_MODEL]),

            // transformer 3
            ln6: LayerNorm([1, D_MODEL]) => LayerNormLayer,
            attn3: Residual([D_MODEL, D_MODEL], <CausalSelfAttention>) => ResidualLayer(a[SEQ, SEQ], [SEQ, D_MODEL]),
            ln7: LayerNorm([1, D_MODEL]) => LayerNormLayer,
            ff3: Residual([D_MODEL, D_FF], <FeedForward>, Swish) => ResidualLayer(a[SEQ, D_FF], [SEQ, D_MODEL]),

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
            ff6: Residual([D_MODEL, D_FF], <FeedForward>, Swish) => ResidualLayer(a[SEQ, D_FF], [SEQ, D_MODEL]),

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
            ff9: Residual([D_MODEL, D_FF], <FeedForward>, Swish) => ResidualLayer(a[SEQ, D_FF], [SEQ, D_MODEL]),

            // transformer 10
            ln20: LayerNorm([1, D_MODEL]) => LayerNormLayer,
            attn10: Residual([D_MODEL, D_MODEL], <CausalSelfAttention>) => ResidualLayer(a[SEQ, SEQ], [SEQ, D_MODEL]),
            ln21: LayerNorm([1, D_MODEL]) => LayerNormLayer,
            ff10: Residual([D_MODEL, D_FF], <FeedForward>, Swish) => ResidualLayer(a[SEQ, D_FF], [SEQ, D_MODEL]),

            // final normalization
            ln22: LayerNorm([1, D_MODEL]) => LayerNormLayer,

            // project features back
            extract: Collapse([D_MODEL, VOCAB]) => CollapseLayer,
            soft: Softmax([SEQ, VOCAB](1.5, 0.2)) => SoftmaxLayer,
        },
        OutputLayer([SEQ, VOCAB]),
    }
);

fn main() {
    let mut model = SeqTransformer::new(0.4);
    let dataset = Dataset::new(INPUTS, TARGETS);

    #[cfg(feature = "std")]
    {
        let start_save = Instant::now();
        let _ = model.save("checkpoints/test/time0.bpat", BpatHeader::BpatV0);
        let _ = model.save("checkpoints/test/time1.bpat", BpatHeader::BpatV1);
        let _ = model.save("checkpoints/test/time2.bpat", BpatHeader::BpatV1M);
        let elapsed_save = start_save.elapsed() / 3;

        println!("SAVE={:?}", elapsed_save);

        let start_load = Instant::now();
        let _ = model.load("checkpoints/test/time0.bpat");
        let _ = model.load("checkpoints/test/time1.bpat");
        let _ = model.load("checkpoints/test/time2.bpat");
        let elapsed_load = start_load.elapsed() / 3;

        println!("LOAD={:?}", elapsed_load);
    }

    set_backend(Backend::Cpu);

    let start_cpu = Instant::now();
    for _ in 0..AVG {
        let _ = black_box(model.fit_epoch(&dataset));
    }
    let elapsed_cpu = start_cpu.elapsed() / AVG;

    println!("CPU={:?}ms", elapsed_cpu);

    #[cfg(feature = "wgpu")]
    {
        set_backend(Backend::Wgpu);

        let start_wgpu = Instant::now();
        for _ in 0..AVG {
            let _ = black_box(model.fit_epoch(&dataset));
        }
        let elapsed_wgpu = start_wgpu.elapsed() / AVG;

        println!("WGPU={:?}ms", elapsed_wgpu);
    }

    #[cfg(feature = "cuda")]
    {
        set_backend(Backend::Cuda);

        let start_cuda = Instant::now();
        for _ in 0..AVG {
            let _ = black_box(model.fit_epoch(&dataset));
        }
        let elapsed_cuda = start_cuda.elapsed() / AVG;

        println!("CUDA={:?}ms", elapsed_cuda);
    }
}
