## 2024-05-14 - Optimize small array sum in hot path
**Learning:** In CPython hot-path functions, evaluating generator comprehensions inside `sum()` (e.g., `sum(1 for x in obj)`) introduces significant function call overhead.
**Action:** Optimize by replacing these with unrolled additions (e.g., `(a > 0) + (b > 0)`) for small fixed structures (<= 4 elements), yielding ~4-7x speedup.
