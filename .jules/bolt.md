## 2024-04-08 - [Python Generator Overhead in Hot Paths]
**Learning:** [In Python, generator comprehensions like `sum(1 for x in iter if cond)` have surprisingly high overhead due to generator object creation on every call. In tight, frequently executed loops like `is_agari` or `calculate_shanten` iterating over small arrays (like 34 tiles or 4 melds), standard `for` loops or unrolled addition are measurably faster (about 30-50% speedup).]
**Action:** [Avoid `sum()` with generators for trivial counts inside computational hot paths in Python; prefer explicit `for` loops or direct addition.]
