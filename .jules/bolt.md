## 2025-05-18 - [Eliminate sum() generator overhead]
**Learning:** In CPython hot-path functions, evaluating generator comprehensions inside `sum()` (e.g., `sum(1 for x in obj)`) introduces significant function call overhead.
**Action:** Optimize by replacing these with list comprehensions evaluated by `len()` (e.g., `len([x for x in obj if condition])`) or boolean addition (e.g., `(a > 0) + (b > 0)`) for small fixed structures.
