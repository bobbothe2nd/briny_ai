@group(0) @binding(0) var<storage, read_write> weights: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<uniform> lr: f32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let base = id.x * 4u;
    let len = arrayLength(&weights);

    for (var offset = 0u; offset < 4u; offset = offset + 1u) {
        let i = base + offset;
        if (i >= len) {
            break;
        }
        weights[i] = weights[i] - lr * grad[i];
    }
}
