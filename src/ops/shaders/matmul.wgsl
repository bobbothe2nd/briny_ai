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
    let row = gid.y;
    let col = gid.x;

    let m = dims.m;
    let k = dims.k;
    let n = dims.n;

    if row >= m || col >= n {
        return;
    }

    let ta = (dims.flags & 1u) != 0u;
    let tb = (dims.flags & 2u) != 0u;

    let acc_base = 0.0;
    var acc = acc_base;

    let k4 = k / 4u;
    let rem = k % 4u;

    for (var i = 0u; i < k4; i = i + 1u) {
        let ki = i * 4u;

        // Indexing for A: handle transpose with select
        let a0 = select(row * k + ki + 0u, (ki + 0u) * m + row, ta);
        let a1 = select(row * k + ki + 1u, (ki + 1u) * m + row, ta);
        let a2 = select(row * k + ki + 2u, (ki + 2u) * m + row, ta);
        let a3 = select(row * k + ki + 3u, (ki + 3u) * m + row, ta);

        let va = vec4<f32>(
            A[a0],
            A[a1],
            A[a2],
            A[a3]
        );

        // Indexing for B: handle transpose with select
        let b0 = select((ki + 0u) * n + col, col * k + ki + 0u, tb);
        let b1 = select((ki + 1u) * n + col, col * k + ki + 1u, tb);
        let b2 = select((ki + 2u) * n + col, col * k + ki + 2u, tb);
        let b3 = select((ki + 3u) * n + col, col * k + ki + 3u, tb);

        let vb = vec4<f32>(
            B[b0],
            B[b1],
            B[b2],
            B[b3]
        );

        acc = acc + dot(va, vb);
    }

    // Tail handling for remainder
    if rem != 0u {
        for (var i = k - rem; i < k; i = i + 1u) {
            let a_idx = select(row * k + i, i * m + row, ta);
            let b_idx = select(i * n + col, col * k + i, tb);
            acc = acc + A[a_idx] * B[b_idx];
        }
    }

    C[row * n + col] = acc;
}
