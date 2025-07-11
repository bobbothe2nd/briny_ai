@group(0) @binding(0) var<storage, read> prediction: array<f32>;
@group(0) @binding(1) var<storage, read> expected: array<f32>;
@group(0) @binding(2) var<storage, read_write> loss: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let len = arrayLength(&prediction);
    if (i >= len) {
        return;
    }

    let diff = prediction[i] - expected[i];
    loss[i] = diff * diff;
}
