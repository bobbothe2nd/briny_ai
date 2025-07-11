@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_idx = global_id.x * 4u;
    let len = arrayLength(&input);

    // process 4 elements per thread
    for (var offset = 0u; offset < 4u; offset = offset + 1u) {
        let i = base_idx + offset;
        if (i >= len) {
            break;
        }
        output[i] = max(input[i], 0.0);
    }
}
