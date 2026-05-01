## 2024-10-24 - Replace Generator Comprehensions with List Comprehensions/len() in Hot Paths
**Learning:** Python generator comprehensions inside `sum()` (e.g. `sum(1 for c in hand_34 if c >= 2)`) incur significant overhead in hot paths.
**Action:** Prefer C-optimized list comprehensions combined with length checks `len([...])` or summing list comprehensions `sum([...])`.
