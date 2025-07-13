# Changelog

`briny_ai v0.2.1`: now with blazing-fast GPU backprop and the beginnings of a secure-by-design AI stack.

## [v0.2.1] – 2025-07-13

### Added

- **GPU-accelerated backward pass for matrix multiplication**
  - The backward pass (∂L/∂A, ∂L/∂B) now runs on the GPU when `wgpu` is enabled.
  - This delivers significant performance improvements for large-scale training workloads.
  - No API changes!
- **Initial integration of `briny` crate** for secure low-level byte handling.
  - Models Zero Trust Architecture (ZTA) at the byte level.
  - Provides zero-cost, compile-time guarantees that prevent entire classes of memory corruption and security vulnerabilities in AI workloads.
  - Enforces strong safety invariants without runtime overhead.
- `GpuFailure` type introduced for unified, structured GPU error handling.
  - Implements `From<&str>` and `From<String>` for ergonomic and consistent error handling.

### Changed

- All GPU shader functions now return `Result<_, GpuFailure>` instead of panicking.
  - Improves safety and robustness across GPU backends.
  - Examples include `run_sgd_shader` and `run_matmul_shader`.
- Improved tensor parsing logic; `parse_tensor` now handles edge cases properly.

### Internal

- Removed `bytemuck` dependency to reduce binary size and keep GPU layer minimal.
  - This eliminates a source of binary bloat while maintaining performance.
  - Marginally improves compile times.
- Improved buffer staging and mapping logic using `?`-based error flow.
