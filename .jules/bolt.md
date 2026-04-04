## 2024-03-24 - Numba Optimization Cleanup
**Learning:** Numba's JIT compiler is powerful, but leaving dead Python code inside a `@njit` block can cause maintainability issues and clutters the repository, even if LLVM strips it out at compilation.
**Action:** Always clean up dead code and scratchpad comments from `@njit` functions after testing optimizations.
