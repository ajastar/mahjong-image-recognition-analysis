## 2024-05-18 - Avoid Generator Comprehensions in sum() for Performance
**Learning:** Python generator comprehensions inside `sum()` (e.g., `sum(1 for c in hand_34 if c >= 2)`) incur significant overhead in hot paths compared to C-optimized list comprehensions.
**Action:** Use list comprehensions combined with length checks (e.g., `len([c for c in hand_34 if c >= 2])`) or `sum([...])` for a measurable speedup without sacrificing code readability.
