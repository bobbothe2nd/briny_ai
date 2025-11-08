@group(0) @binding(0) var<storage, read_write> weights: array<f32>;
@group(0) @binding(1) var<storage, read_write> grad: array<f32>;
@group(0) @binding(2) var<uniform> lr: f32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let base = id.x * 4u;

    // bounds guard: if weights length isn't exact multiple (defensive)
    if (base + 3u >= arrayLength(&weights)) { return; }

    let w = vec4<f32>(
        weights[base + 0u],
        weights[base + 1u],
        weights[base + 2u],
        weights[base + 3u],
    );

    let g = vec4<f32>(
        grad[base + 0u],
        grad[base + 1u],
        grad[base + 2u],
        grad[base + 3u],
    );

    let updated = w - g * vec4<f32>(lr);

    weights[base + 0u] = updated.x;
    weights[base + 1u] = updated.y;
    weights[base + 2u] = updated.z;
    weights[base + 3u] = updated.w;

    // Zero out the gradients in-place on the GPU
    grad[base + 0u] = 0.0;
    grad[base + 1u] = 0.0;
    grad[base + 2u] = 0.0;
    grad[base + 3u] = 0.0;
}
