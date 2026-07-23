## 2024-07-23 - Python Generator Overhead in Hot Paths
**Learning:** In CPython, evaluating generator comprehensions inside functions like `sum()` or `any()` (e.g., `sum(1 for x in obj)`) introduces significant function call overhead, particularly for small domains.
**Action:** Optimize by replacing these with list comprehensions evaluated by `len()`, explicit `for` loops with early exits, or unrolled mathematical operations (e.g., `arr[0] + arr[1]`).
