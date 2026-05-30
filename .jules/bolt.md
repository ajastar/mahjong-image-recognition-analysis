## 2024-05-30 - Replace `sum()` generator comprehensions with `len()` + list comprehension
**Learning:** CPython's list comprehensions wrapped in `len()` (or direct unrolled addition) are significantly faster than generator comprehensions inside `sum()` due to the overhead of frame allocation and `next()` function calls, particularly in hot-path methods like `is_agari` and `get_hand_han` in mahjong evaluations.
**Action:** Always prefer unrolling addition for constant lookups (e.g. terminals array lookup) or using `len([...])` instead of `sum(...)` for counting filtered attributes or sizes.
