@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x * 4u;

    let v = vec4<f32>(
        input[i + 0u],
        input[i + 1u],
        input[i + 2u],
        input[i + 3u],
    );

    let result = max(v, vec4<f32>(0.0));

    output[i + 0u] = result.x;
    output[i + 1u] = result.y;
    output[i + 2u] = result.z;
    output[i + 3u] = result.w;
}
