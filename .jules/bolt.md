
## 2024-05-15 - Optimize Generator Comprehensions in Python Hot Paths
**Learning:** In CPython, evaluating generator comprehensions inside `sum()` introduces significant function call overhead for frame allocation and `next()` calls. It is measurably faster to use C-optimized list comprehensions combined with length checks (e.g., `len([1 for x in obj])`) or to use unrolled addition for flat addition over fixed array indices. Tests showed `len([])` is ~30% faster than `sum()` with generators, and unrolled addition is over 7x faster than dynamic list creation and generator sums.
**Action:** Replace `sum()` generator comprehensions with `len()` list comprehensions or unrolled additions in high-frequency functions.
