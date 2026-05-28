
## 2024-05-28 - [sum() Generator Comprehension Overhead]
**Learning:** [In CPython hot paths, evaluating generator comprehensions inside `sum()` (e.g., `sum(1 for x in obj)`) introduces function call overhead for frame allocation and `next()` calls. It is significantly slower than unrolling flat additions or even using `len()` with C-optimized list comprehensions.]
**Action:** [To optimize small array summations in hot paths, prefer unrolled additions like `arr[0] + arr[1] + ...`. For counting conditions, prefer `len([x for x in obj if cond])` instead of `sum(1 for x in obj if cond)`.]
