## 2023-10-27 - [Replacing sum with generator expression for performance]
**Learning:** Generator expressions inside `sum()` are slow. Replacing them with unrolled iteration or list comprehension with `len()` improves performance in hot-paths.
**Action:** Always replace `sum(1 for x in ...)` with `len([x for x in ...])` when possible in performance-critical code.
