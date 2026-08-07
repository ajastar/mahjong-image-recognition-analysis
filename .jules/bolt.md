## 2024-05-18 - [CPython Generator Overhead in Hot Paths]
**Learning:** Using generator comprehensions inside `sum()` (e.g., `sum(1 for x in iter)`) introduces significant function call overhead in CPython compared to alternative approaches.
**Action:** In hot path functions, prefer list comprehensions wrapped in `len()` (e.g., `len([x for x in iter])`), boolean addition for small fixed sets (e.g., `(a>0)+(b>0)`), or unrolling `sum()` operations for small arrays entirely to avoid generation overhead.
