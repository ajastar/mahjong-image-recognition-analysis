## 2024-10-20 - [Unrolling Generator Comprehension Sums on Fixed Array Indices]
**Learning:** In CPython hot-paths like Mahjong hand evaluation, evaluating generator comprehensions inside `sum()` (e.g., `sum(hand_34[t] for t in [...])`) introduces significant function call and iteration overhead.
**Action:** For small, fixed arrays, manually unroll the additions explicitly (e.g., `arr[0] + arr[1] + ...`). This yields a ~4x speedup over generator-based `sum()`.
