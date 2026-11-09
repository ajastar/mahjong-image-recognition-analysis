
## 2024-11-09 - Unrolling Tiny Fixed-Size Array Operations
**Learning:** In CPython hot paths, evaluating generator comprehensions inside functions like `sum()` (e.g., `sum(1 for x in suit_counts if x > 0)`) over tiny fixed-size arrays introduces significant function call overhead.
**Action:** Replace generator comprehensions on very small fixed-length arrays (e.g., `suit_counts` of length 3) with unrolled boolean additions like `(suit_counts[0] > 0) + (suit_counts[1] > 0) + (suit_counts[2] > 0)` to eliminate overhead and speed up execution.
