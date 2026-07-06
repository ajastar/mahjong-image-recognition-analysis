## 2025-02-12 - List comprehension length vs Generator inside sum()
**Learning:** In CPython, evaluating generator comprehensions inside `sum()` (e.g., `sum(1 for x in obj)`) introduces significant function call overhead.
**Action:** Optimize by replacing these with list comprehensions evaluated by `len()` (e.g., `len([x for x in obj if condition])`) or boolean addition (e.g., `(a > 0) + (b > 0)`) for small fixed structures, which avoids the overhead and is significantly faster in pure Python hot paths.
