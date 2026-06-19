## 2024-05-31 - [Optimize generator comprehensions in sum()]
**Learning:** In CPython hot-path functions (e.g., `mahjong_logic.py`), evaluating generator comprehensions inside `sum()` (e.g., `sum(1 for x in obj)`) introduces significant function call overhead.
**Action:** Optimize by replacing these with list comprehensions evaluated by `len()` (e.g., `len([x for x in obj if condition])`), use boolean addition (e.g., `(a > 0) + (b > 0)`) for small fixed structures, or unroll additions (`arr[0] + arr[1]`) for specific indices instead of using dynamically generated lists/generators.
