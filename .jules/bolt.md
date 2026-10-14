## 2025-01-08 - Generator Comprehension Overhead in CPython
**Learning:** Evaluating generator comprehensions inside `sum()` (e.g., `sum(1 for x in suit_counts if x > 0)`) in pure CPython hot paths introduces significant function call and iteration overhead compared to manual unrolling or simple additions.
**Action:** For small fixed-size data structures like a 3-element list `suit_counts`, replace `sum()` generator expressions with unrolled boolean additions like `(suit_counts[0] > 0) + (suit_counts[1] > 0) + (suit_counts[2] > 0)` to drastically reduce overhead while preserving logic.
