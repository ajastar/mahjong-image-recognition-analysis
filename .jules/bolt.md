
## 2024-07-14 - Optimize generator expressions in sum()
**Learning:** In CPython hot-path functions, evaluating generator comprehensions inside `sum()` (e.g., `sum(1 for x in obj)`) introduces significant function call overhead.
**Action:** Replace these with list comprehensions evaluated by `len()` (e.g., `len([x for x in obj if condition])`) when early exit is not possible.
