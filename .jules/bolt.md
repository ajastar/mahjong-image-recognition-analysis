## 2024-05-24 - Unrolling Generator Expressions in CPython
**Learning:** In CPython hot paths, using generator comprehensions inside built-in functions like `sum()` (e.g., `sum(1 for x in arr)`) on tiny fixed-size arrays (like the 3-element `suit_counts`) introduces massive function call and iteration overhead relative to the calculation itself.
**Action:** Always unroll mathematical or boolean operations explicitly for fixed arrays of length <= 4 (e.g., `arr[0] + arr[1] + arr[2]`) in hot paths to bypass CPython iterator overhead.
