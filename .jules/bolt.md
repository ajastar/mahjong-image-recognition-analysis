## 2024-05-24 - Optimize hot path sum() with unrolling and list comprehensions
**Learning:** In CPython hot-path functions, evaluating generator comprehensions inside `sum()` introduces significant overhead.
**Action:** Optimize by replacing these with list comprehensions evaluated by `len()`, boolean additions, or manually unrolled additions.
