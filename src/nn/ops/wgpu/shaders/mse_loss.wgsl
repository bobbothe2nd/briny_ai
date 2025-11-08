@group(0) @binding(0) var<storage, read> prediction: array<f32>;
@group(0) @binding(1) var<storage, read> expected: array<f32>;
@group(0) @binding(2) var<storage, read_write> loss: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x * 4u;
    let len = arrayLength(&prediction);
    if i + 3u >= len {
        // Scalar fallback for tail
        if i + 0u < len {
            let d = prediction[i + 0u] - expected[i + 0u];
            loss[i + 0u] = d * d;
        }
        if i + 1u < len {
            let d = prediction[i + 1u] - expected[i + 1u];
            loss[i + 1u] = d * d;
        }
        if i + 2u < len {
            let d = prediction[i + 2u] - expected[i + 2u];
            loss[i + 2u] = d * d;
        }
        return;
    }

    let p = vec4<f32>(
        prediction[i + 0u],
        prediction[i + 1u],
        prediction[i + 2u],
        prediction[i + 3u],
    );
    let e = vec4<f32>(
        expected[i + 0u],
        expected[i + 1u],
        expected[i + 2u],
        expected[i + 3u],
    );

    let d = p - e;
    let sq = d * d;

    loss[i + 0u] = sq.x;
    loss[i + 1u] = sq.y;
    loss[i + 2u] = sq.z;
    loss[i + 3u] = sq.w;
}
