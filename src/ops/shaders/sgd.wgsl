@group(0) @binding(0) var<storage, read_write> weights: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<uniform> lr: f32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x * 4u;

    let w = vec4<f32>(
        weights[i + 0u],
        weights[i + 1u],
        weights[i + 2u],
        weights[i + 3u],
    );

    let g = vec4<f32>(
        grad[i + 0u],
        grad[i + 1u],
        grad[i + 2u],
        grad[i + 3u],
    );

    let updated = w - g * vec4<f32>(lr);

    weights[i + 0u] = updated.x;
    weights[i + 1u] = updated.y;
    weights[i + 2u] = updated.z;
    weights[i + 3u] = updated.w;
}
