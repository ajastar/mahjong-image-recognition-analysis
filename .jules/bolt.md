## 2024-05-24 - [List Comp over Gen Comp in Hot Paths]
**Learning:** Python generator comprehensions inside `sum()` (like `sum(1 for c in hand_34 if c >= 2)`) incur significant overhead compared to C-optimized list comprehensions combined with length checks (like `len([c for c in hand_34 if c >= 2])`).
**Action:** Use `len([...])` instead of `sum(1 for ...)` in hot paths like hand evaluation functions.
