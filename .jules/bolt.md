## 2024-07-20 - Remove sum() generator overhead in hot paths
**Learning:** In CPython, evaluating generator comprehensions inside `sum()` (e.g., `sum(1 for x in obj)`) introduces significant function call overhead, especially in hot paths like hand evaluation logic.
**Action:** Replace `sum()` generator expressions with list comprehensions combined with `len()` (e.g., `len([x for x in obj])`), use unrolled boolean addition for fixed structures (e.g. `(a>0) + (b>0)`), or use explicit `for` loops with early exit for checking conditions.
