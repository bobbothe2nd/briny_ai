struct U32x4 {
    value: u32,
    _pad: vec3<u32>,
};

struct TensorInfo {
    rank: u32,
    _pad: vec3<u32>,
    shape: array<U32x4, 8>, // up to rank-8 tensors
    stride: array<U32x4, 8>,
};

struct ContractionInfo {
    num_axes: u32,
    _pad: vec3<u32>,
    a_axes: array<U32x4, 4>,
    b_axes: array<U32x4, 4>,
};

@group(0) @binding(0) var<uniform> A_info: TensorInfo;
@group(0) @binding(1) var<uniform> B_info: TensorInfo;
@group(0) @binding(2) var<uniform> C_info: TensorInfo;
@group(0) @binding(3) var<uniform> contract: ContractionInfo;
@group(0) @binding(4) var<storage, read> A: array<f32>;
@group(0) @binding(5) var<storage, read> B: array<f32>;
@group(0) @binding(6) var<storage, read_write> C: array<f32>;

fn flatten_index8(indices: ptr<function, array<u32, 8>>, strides: ptr<function, array<u32, 8>>, rank: u32) -> u32 {
    var offset = 0u;
    for (var i = 0u; i < rank; i = i + 1u) {
        offset = offset + (*indices)[i] * (*strides)[i];
    }
    return offset;
}

fn unflatten_index8(flat: u32, shape: ptr<function, array<u32, 8>>, rank: u32, out_indices: ptr<function, array<u32, 8>>) {
    var idx = flat;
    for (var i = rank; i > 0u; i = i - 1u) {
        let dim = (*shape)[i - 1u];
        (*out_indices)[i - 1u] = idx % dim;
        idx = idx / dim;
    }
}

fn unflatten_index4(flat: u32, shape: ptr<function, array<u32, 4>>, rank: u32, out_indices: ptr<function, array<u32, 4>>) {
    var idx = flat;
    for (var i = rank; i > 0u; i = i - 1u) {
        let dim = (*shape)[i - 1u];
        (*out_indices)[i - 1u] = idx % dim;
        idx = idx / dim;
    }
}

fn num_elements8(shape: ptr<function, array<u32, 8>>, rank: u32) -> u32 {
    var total = 1u;
    for (var i = 0u; i < rank; i = i + 1u) {
        total = total * (*shape)[i];
    }
    return total;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat_index = gid.x;

    // extract shapes & strides from padded structs
    var a_shape = array<u32, 8>();
    var a_stride = array<u32, 8>();
    var b_shape = array<u32, 8>();
    var b_stride = array<u32, 8>();
    var c_shape = array<u32, 8>();
    var c_stride = array<u32, 8>();

    for (var i = 0u; i < A_info.rank; i = i + 1u) {
        a_shape[i] = A_info.shape[i].value;
        a_stride[i] = A_info.stride[i].value;
    }
    for (var i = 0u; i < B_info.rank; i = i + 1u) {
        b_shape[i] = B_info.shape[i].value;
        b_stride[i] = B_info.stride[i].value;
    }
    for (var i = 0u; i < C_info.rank; i = i + 1u) {
        c_shape[i] = C_info.shape[i].value;
        c_stride[i] = C_info.stride[i].value;
    }

    let total_c = num_elements8(&c_shape, C_info.rank);
    if flat_index >= total_c {
        return;
    }

    // output element coordinates
    var c_idx = array<u32, 8>();
    unflatten_index8(flat_index, &c_shape, C_info.rank, &c_idx);

    var acc = 0.0;

    // build contraction shape (axes in A)
    var contract_shape = array<u32, 4>();
    var total_contract_elems = 1u;
    for (var i = 0u; i < contract.num_axes; i = i + 1u) {
        contract_shape[i] = A_info.shape[contract.a_axes[i].value].value;
        total_contract_elems = total_contract_elems * contract_shape[i];
    }

    // iterate over all combinations of contracted indices
    for (var t = 0u; t < total_contract_elems; t = t + 1u) {
        var contract_idx = array<u32, 4>();
        unflatten_index4(t, &contract_shape, contract.num_axes, &contract_idx);

        var a_idx = array<u32, 8>();
        var b_idx = array<u32, 8>();

        // fill with output indices (default)
        for (var i = 0u; i < A_info.rank; i = i + 1u) {
            if (i < C_info.rank) {
                a_idx[i] = c_idx[i];
            } else {
                a_idx[i] = 0u;
            }
        }
        for (var i = 0u; i < B_info.rank; i = i + 1u) {
            if (i < C_info.rank) {
                b_idx[i] = c_idx[i];
            } else {
                b_idx[i] = 0u;
            }
        }

        // overwrite contracted axes
        for (var i = 0u; i < contract.num_axes; i = i + 1u) {
            let ax = contract.a_axes[i].value;
            let bx = contract.b_axes[i].value;
            a_idx[ax] = contract_idx[i];
            b_idx[bx] = contract_idx[i];
        }

        let a_flat = flatten_index8(&a_idx, &a_stride, A_info.rank);
        let b_flat = flatten_index8(&b_idx, &b_stride, B_info.rank);

        acc = acc + A[a_flat] * B[b_flat];
    }

    C[flat_index] = acc;
}
