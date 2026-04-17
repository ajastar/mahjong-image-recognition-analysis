## 2024-04-17 - Performance Anti-Pattern: Generator Comprehensions in sum()
**Learning:** Python generator comprehensions inside `sum()` (e.g., `sum(1 for c in hand_34 if c >= 2)`) incur significant overhead from generator setup and teardown compared to C-optimized list comprehensions.
**Action:** Use `len([c for c in hand_34 if c >= 2])` instead of `sum(1 for ...)` for small lists to improve performance without sacrificing readability.
