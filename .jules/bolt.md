## 2024-04-16 - [Python Generator Overhead in sum()]
**Learning:** In hot-path functions, Python generator comprehensions inside `sum()` (like `sum(1 for c in hand_34 if c >= 2)`) incur significant overhead.
**Action:** To optimize this without sacrificing code readability, prefer C-optimized list comprehensions combined with length checks, such as `len([c for c in hand_34 if c >= 2])`.
