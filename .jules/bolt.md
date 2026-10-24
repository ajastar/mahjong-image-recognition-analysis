## 2024-10-24 - Avoid generator comprehensions inside sum() for fixed-size arrays
**Learning:** In CPython hot-path functions, evaluating generator comprehensions inside \`sum()\` (e.g., \`sum(1 for x in obj)\`) introduces significant function call overhead.
**Action:** Optimize by replacing these with boolean additions (e.g., \`(a > 0) + (b > 0)\`) for small fixed structures (<= 4 elements), yielding substantial speedup.
