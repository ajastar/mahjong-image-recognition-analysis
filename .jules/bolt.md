## 2024-05-18 - Unroll any() generator over tiny fixed array
**Learning:** In CPython hot paths, using `any()` with generator comprehensions over tiny fixed-size arrays (e.g. length 9) incurs significant function call and iteration overhead compared to explicitly unrolling the logical ORs.
**Action:** Unroll logical operations over small, fixed-length arrays (<= 9) directly instead of using generator comprehensions to eliminate overhead.
