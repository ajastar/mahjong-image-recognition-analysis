## 2026-08-13 - [Optimize generator expressions inside sum]
**Learning:** In CPython hot paths, evaluating generator comprehensions inside `sum()` (e.g., `sum(1 for x in obj)`) introduces significant function call overhead compared to evaluating a list comprehension with `len()` or manually unrolling a boolean addition across a small, static array (e.g., `suit_counts`).
**Action:** Replace generator sums with `len([x for x in obj])` or unrolled boolean checks for static small collections.
