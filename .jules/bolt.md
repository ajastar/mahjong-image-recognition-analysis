## 2024-08-12 - Replacing CPython generators in sum()
**Learning:** Evaluating generator comprehensions inside `sum()` (e.g., `sum(1 for x in obj)`) introduces significant function call iteration overhead in CPython hot-paths compared to list comprehensions evaluated by `len()` or manual index/boolean unrolling.
**Action:** Optimize hot-path summation routines in small fixed-size domains by manually unrolling addition or replacing with `len([comprehensions])`.
