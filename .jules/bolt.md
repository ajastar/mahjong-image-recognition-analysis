## 2024-05-24 - [Avoid `any()` with Generator Comprehensions on Tiny Arrays]
**Learning:** Using Python's built-in `any()` with a generator comprehension (e.g., `any(c >= 3 for c in counts_by_mod)`) on tiny, fixed-size arrays (like length-9 `counts_by_mod`) introduces significant function call and iteration overhead in hot paths.
**Action:** Manually unroll the logical OR evaluations (e.g., `arr[0] >= 3 or arr[1] >= 3 ...`) for these specific cases to avoid this overhead and achieve substantial speedups (~3x).
