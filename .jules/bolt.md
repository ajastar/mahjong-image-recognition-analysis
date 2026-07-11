
## 2024-05-18 - [Eliminated Python `sum()` with Generator Comprehensions in Hot Paths]
**Learning:** In CPython, evaluating generator comprehensions inside `sum()` (e.g., `sum(1 for c in hand_34 if c >= 2)`) introduces significant function call overhead inside hot paths.
**Action:** Replace `sum()` and generator expressions with manual loops and unrolled additions for improved performance in hot loops, at a small expense to readability but providing significant speedup.
