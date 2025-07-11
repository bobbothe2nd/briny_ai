# Changelog

## [v0.2.0] – 2025-07-10

### Added

- **SIMD-accelerated CPU operations** using AVX2 (`--features simd`):
  - `matmul` (4-wide vectorized multiply-accumulate)
  - `relu` (branchless max)
- **Parallelization with Rayon** for forward/backward passes:
  - `matmul`
  - `mse_loss`
  - `relu`
- **Autograd support** via `WithGrad<T>`:
  - `.new()` constructor and `.with_grad()` on `Tensor<T>`
  - Backward closures for `matmul`, `mse_loss`, and `relu`
- **Initial GPU backend (`wgpu`)**:
  - Dynamically selected at runtime through `ops::dispatch`
  - Centralized `GpuContext` and device-queue abstraction
- **Modular operation backends**:
  - `ops::cpu`, `ops::wgpu`, and placeholder `ops::cuda`
  - Feature-gated and cleanly separated for future expansion
- **SGD optimizer**:
  - In-place updates on `WithGrad<Tensor<f64>>`
  - Fully compatible with autograd tracing1

---

### Changed

- Refactored ops into `ops::cpu` and `ops::wgpu` modules
- Unified fallback logic via `#[cfg(...)]` blocks
- Greatly improved performance of CPU ops using SIMD + Rayon
- Simplified backward propagation structure
- Added bounds and shape checks to CPU ops (`matmul`, `mse_loss`)

---

### Documentation

- Added full module-level docs (`cpu`, `dispatch`, `ops`)
- Documented all public types and methods:
  - `Tensor<T>`, `WithGrad<T>`, `GpuContext`
- Backpropagation logic explained inline with code examples

---

### Internal

- Reorganized code structure for extensibility and testing
- Added `cuda` module scaffold (not yet implemented)
- Improved test coverage and validation checks

---

### Coming Soon (`v0.2.1` and beyond)

- GPU kernel support for:
  - `matmul` (tiled and shared-memory optimized)
  - `relu`, `mse_loss`, `sgd`
- CUDA backend via [`cust`](https://crates.io/crates/cust) or [`accel`](https://crates.io/crates/accel)
- Batched matrix inference (`Tensor4D`, etc.)
- High-level neural layer API (e.g. `MLP::forward`)
