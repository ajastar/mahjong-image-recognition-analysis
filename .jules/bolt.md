## 2024-05-15 - Unrolling `sum()` Generators in Hot Paths
**Learning:** In CPython, evaluating generator comprehensions inside `sum()` (e.g., `sum(1 for x in obj)`) introduces significant function call and instantiation overhead in hot paths like `mahjong_logic.py` evaluation loops.
**Action:** Optimize by replacing these with unrolled boolean additions (e.g., `(a > 0) + (b > 0)`) for small fixed structures, or early-return/manual `for` loops, yielding measurable speedup (2x-4x in benchmarks).
