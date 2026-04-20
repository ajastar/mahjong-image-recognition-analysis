## 2025-04-20 - [Optimize sum(1 for ...) generators]
**Learning:** Python generator comprehensions inside `sum()` (e.g., `sum(1 for c in hand_34 if c >= 2)`) incur significant overhead in hot-path functions. They are measurably slower than list comprehensions inside `len()` (e.g., `len([c for c in hand_34 if c >= 2])`).
**Action:** Use list comprehensions inside `len()` when calculating counts over short lists or arrays in Python to improve execution speed.
