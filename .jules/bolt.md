## 2024-07-04 - Optimize `sum(1 for ...)`
**Learning:** In CPython hot-path functions, evaluating generator comprehensions inside `sum()` (e.g., `sum(1 for x in obj)`) introduces significant function call overhead. Optimize by replacing these with list comprehensions evaluated by `len()` or other fixed structures, or unroll additions when possible.
**Action:** Replace `sum(1 for ...)` generator comprehensions with faster equivalents like `len([x for x in obj if condition])`.
