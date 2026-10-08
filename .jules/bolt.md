## 2024-10-08 - Hot Path Generator Unrolling
**Learning:** In CPython hot-path functions (e.g., `mahjong_logic.py`), evaluating generator comprehensions inside `sum()` (e.g., `sum(1 for x in suit_counts if x > 0)`) introduces significant function call and iteration overhead.
**Action:** Optimize by replacing these with unrolled boolean additions (e.g., `(suit_counts[0] > 0) + (suit_counts[1] > 0) + (suit_counts[2] > 0)`) for small fixed structures like the length-3 `suit_counts` array, yielding ~5x speedups.
