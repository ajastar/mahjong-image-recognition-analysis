## 2024-08-28 - Optimize sum calls in hand evaluation
**Learning:** In CPython hot-path functions, evaluating generator comprehensions inside `sum()` (e.g., `sum(1 for x in suit_counts if x > 0)`) introduces significant function call overhead.
**Action:** Replace generator expressions with boolean addition (e.g., `(a > 0) + (b > 0)`) for small fixed structures, or unroll additions (`arr[0] + arr[1]`) for specific indices instead of using dynamically generated lists/generators.
