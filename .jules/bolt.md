## 2024-10-04 - [Optimize any() generator on fixed arrays]
**Learning:** Using `any()` with a generator expression on small fixed-size arrays (like a length-9 list) introduces significant function call and iteration overhead in Python hot paths.
**Action:** Unroll logical OR evaluations explicitly for small, fixed-length arrays to achieve large performance gains.
