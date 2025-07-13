struct MatDims {
    m: u32,
    k: u32,
    n: u32,
    flags: u32, // bit 0 = transpose A, bit 1 = transpose B
};

@group(0) @binding(0) var<uniform> dims: MatDims;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ta = (dims.flags & 1u) != 0u;
    let tb = (dims.flags & 2u) != 0u;

    // A access: A[ta ? i * m + row : row * k + i]
    // B access: B[tb ? col * k + i : i * n + col]

    let row = gid.y;
    let col = gid.x;

    let m = dims.m;
    let k = dims.k;
    let n = dims.n;

    if row >= m || col >= n {
        return;
    }

    var acc = 0.0;

    let k4 = k / 4u; // number of vec4s
    let baseA = row * k;
    let baseB = col;

    for (var i = 0u; i < k4; i = i + 1u) {
        let ai = baseA + i * 4u;
        let bi = i * 4u * n + baseB;

        let va = vec4<f32>(
            A[ai + 0u],
            A[ai + 1u],
            A[ai + 2u],
            A[ai + 3u],
        );

        let vb = vec4<f32>(
            B[bi + 0u * n],
            B[bi + 1u * n],
            B[bi + 2u * n],
            B[bi + 3u * n],
        );

        acc = acc + dot(va, vb);
    }

    // Tail handling if k not divisible by 4
    let rem = k % 4u;
    for (var i = k - rem; i < k; i = i + 1u) {
        acc = acc + A[row * k + i] * B[i * n + col];
    }

    C[row * n + col] = acc;
}
