## 2024-06-04 - Generator Comprehensions inside sum()
**Learning:** Using generator comprehensions inside `sum()` (e.g., `sum(1 for x in obj if cond)`) introduces significant function call overhead in CPython, especially in hot paths like hand evaluation in Mahjong.
**Action:** Replace `sum(1 for ...)` with explicit loops and accumulators for lists or, for small fixed-size sequences, use boolean addition (e.g., `(a > 0) + (b > 0)`). For flat sums over specific indices, direct unrolled addition is much faster than slices or generator comprehensions.
