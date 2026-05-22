
## 2024-05-24 - [Optimize dynamic list generation and sum generator expressions in hot paths]
**Learning:** Found that generating dynamic lists via concatenation `[0,8,9,17,18,26] + list(range(27,34))` inside loops incurs measurable overhead. Additionally, using generator expressions inside `sum()` for small fixed sets of indices is significantly slower than using simple unrolled addition directly.
**Action:** Lift invariant static collections to module-level constants (e.g. `frozenset` for membership testing, `tuple` for iteration) to prevent reallocation. Replace generator loops inside `sum()` with unrolled addition when aggregating flat counts over fixed indices.
