## 2024-05-19 - Replace sum with generator comprehension with faster equivalents
**Learning:** In CPython, `sum(generator_comprehension)` incurs significant generator setup and iteration overhead. Replacing it with unrolled addition, boolean addition, or `len([list_comprehension])` provides roughly a 2x-8x speedup.
**Action:** When counting conditions over small static lists or elements in hot loops in python logic, avoid `sum()` over generators in favor of faster equivalent operations.
