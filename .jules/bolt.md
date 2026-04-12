## 2024-05-24 - Python Generator Comprehension Overhead in sum()
**Learning:** In hot-path functions executed frequently (like mahjong hand evaluation), using `sum(1 for c in iterable if condition)` incurs significant generator setup and execution overhead.
**Action:** Replace generator comprehensions inside `sum()` with explicit `for` loops and accumulator variables when optimizing Python bottlenecks, reducing iteration overhead by ~40%.
