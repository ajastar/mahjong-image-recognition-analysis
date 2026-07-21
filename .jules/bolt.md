## 2024-05-19 - [Python Generator Overhead in sum()]
**Learning:** In hot paths (like Mahjong simulation loops evaluating tens of thousands of hands), using Python generator expressions inside `sum()` (e.g., `sum(1 for x in obj if cond)`) creates massive iteration and function call overhead compared to explicit loops, list comprehensions evaluated with `len()`, or boolean additions.
**Action:** Replace `sum(generator)` with `len([list comprehension])` or manual unrolled calculations in frequently called methods.
