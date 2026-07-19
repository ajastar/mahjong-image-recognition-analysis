## 2024-05-18 - [Optimized Python Generators]
**Learning:** In CPython hot-path functions, evaluating generator comprehensions inside sum() introduces significant function call overhead compared to explicit loops or unrolled booleans.
**Action:** Replace sum(generator) with unrolled booleans (a > 0) + (b > 0) or len([1 for x in obj]) to eliminate CPython iteration overhead in hot loops.
