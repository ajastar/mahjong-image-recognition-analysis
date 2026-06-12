## 2024-06-12 - Dynamic List Allocation and Generator Overhead in Hot Paths
**Learning:** In CPython, evaluating generator comprehensions inside `sum()` and dynamically allocating static lists (e.g. `[0,8,9,17,18,26] + list(range(27,34))`) introduces significant overhead in hot paths (like structure evaluation or threat scoring).
**Action:** Replace dynamically constructed collections used for membership testing with module-level `frozenset` objects. Replace `sum()` generator comprehensions for small fixed structures with unrolled explicit additions.
