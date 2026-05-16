## 2024-05-16 - [Optimize Generators in Hot Paths]
**Learning:** Evaluating generator comprehensions inside `sum()` (e.g., `sum(1 for x in obj)`) introduces function call overhead for frame allocation and `next()` calls in hot evaluation paths.
**Action:** The optimal strategy in CPython is either unrolling loops (for fixed size small arrays), using `len()` with list comprehensions (e.g. `len([x for x in obj])`), or simple `for` loops counting manually.
