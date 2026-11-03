
## 2024-11-03 - [Optimize get_suji hotpath]
**Learning:** Functions returning small dynamic lists from small fixed input domains (like 34 mahjong tiles) called in hot paths suffer from list instantiation overhead. Using `@functools.lru_cache` breaks Numba compilation.
**Action:** Extract list generations into O(1) module-level immutable tuple lookups (e.g., `SUJI_MAP`).

## 2024-11-03 - [Optimize any() generator hotpath]
**Learning:** In Python hot paths, evaluating generator comprehensions inside built-in functions like `any()` on small fixed-size arrays (like length-9 `counts_by_mod`) introduces massive function call and iteration overhead.
**Action:** Unroll these logical OR evaluations explicitly (e.g., `arr[0] >= 3 or arr[1] >= 3 ...`) for significant (~10x) speedups.
