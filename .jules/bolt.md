## 2024-05-24 - [Replace Generator Comprehensions in sum() for Hot Paths]
**Learning:** Evaluating generator comprehensions inside `sum()` (e.g., `sum(1 for x in obj)`) introduces significant function call overhead in CPython hot-paths, degrading performance by >40% compared to alternatives.
**Action:** Replace generator comprehensions in `sum()` with list comprehensions evaluated by `len()` (e.g., `len([x for x in obj if condition])`) or manual index unrolling (e.g., `(a > 0) + (b > 0)`) in pure Python hot-paths for significant speedups.
