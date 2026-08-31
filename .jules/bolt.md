## 2024-05-30 - [Optimize fixed-size generator sums]
**Learning:** In CPython hot-paths, evaluating generator comprehensions inside `sum()` for small, fixed structures (like `sum(len(d) for d in discards_hist)` where length is always 4, or `sum(1 for x in suit_counts if x > 0)` where length is always 3) introduces significant function call overhead.
**Action:** Replace these standard `sum()` calls with manual unrolled additions (e.g. `len(discards_hist[0]) + ... + len(discards_hist[3])`) or boolean additions for tiny fixed arrays to eliminate generator overhead.
