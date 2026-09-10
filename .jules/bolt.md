
## 2024-05-24 - [Avoid sum() and generators in hot paths]
**Learning:** In CPython hot-path functions (e.g., `mahjong_logic.py`), evaluating generator comprehensions inside `sum()` (e.g., `sum(1 for x in obj if x)`) introduces significant function call overhead.
**Action:** Replace these with early-return `for` loops or unrolled additions (e.g., `arr[0] + arr[1] + arr[2]`) for small fixed structures, yielding substantial speedups (up to 4-7x) without sacrificing readability.
