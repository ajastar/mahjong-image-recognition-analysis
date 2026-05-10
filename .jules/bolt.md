
## 2024-05-10 - Performance Anti-Pattern: Generator Comprehensions in sum() for Hot Paths
**Learning:** Evaluating generator comprehensions inside `sum()` (e.g., `sum(1 for x in obj)`) introduces function call overhead for frame allocation and `next()` calls, significantly slowing down CPython hot paths like those in `mahjong_logic.py`. This was proven by benchmarking, where generator comprehensions inside `sum` were ~1.5x to 7x slower than alternatives depending on the task.
**Action:** Use list comprehensions with `len()` (e.g., `len([x for x in obj])`) or manual loops for counting. For flat addition over fixed arrays (like summing terminals in a 34-length Mahjong tile array), directly unroll the addition (e.g., `arr[0] + arr[8] + ...`).
