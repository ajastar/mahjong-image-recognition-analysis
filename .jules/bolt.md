## 2024-10-26 - Unrolling Short Array Generators in Hot Paths
**Learning:** Generator comprehensions with `sum()` (e.g., `sum(1 for x in suit_counts if x > 0)`) introduce significant function call and iteration overhead in Python hot paths.
**Action:** For tiny, fixed-size arrays like `suit_counts` (length 3), unroll logical checks into boolean additions (e.g., `(suit_counts[0] > 0) + (suit_counts[1] > 0) + ...`) to achieve massive speedups (~6x improvement).
