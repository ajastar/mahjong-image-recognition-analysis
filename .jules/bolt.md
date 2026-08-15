## 2024-08-15 - Optimize Generator Comprehensions inside sum()
**Learning:** Generator comprehensions inside `sum()` (e.g., `sum(1 for x in obj)`) in hot paths (like Mahjong hand evaluation) introduce significant function call and iteration overhead in Python. Unrolling simple summations or using `len()` with list comprehensions offers measurable speedups.
**Action:** Replace `sum()` with generator expressions with unrolled math for fixed-size lists (like `suit_counts`), or `len([x for x in obj if cond])` for dynamic filtering in performance-critical code.
