
## 2023-10-23 - Unrolling any() for Fixed Arrays
**Learning:** In CPython, evaluating `any()` with a generator expression on small fixed-size arrays introduces significant function call and generator setup overhead.
**Action:** Replace `any(c >= 3 for c in arr)` with an unrolled condition `arr[0] >= 3 or arr[1] >= 3 or ...` for fixed-size small arrays (like `counts_by_mod` with exactly 9 elements) in performance-critical hot paths.
