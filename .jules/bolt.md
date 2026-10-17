## 2024-11-20 - Unroll small generator comprehensions for performance
**Learning:** In CPython hot paths, evaluating generator comprehensions inside sum() (e.g., sum(1 for x in obj)) introduces significant function call overhead. For very small fixed-size arrays (like length-3 suit_counts), unrolling operations directly (e.g., (suit_counts[0] > 0) + (suit_counts[1] > 0) + (suit_counts[2] > 0)) is much faster.
**Action:** Replace small generator comprehensions with unrolled boolean additions in hot paths where array sizes are small and fixed.
