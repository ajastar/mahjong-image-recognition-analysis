## 2026-06-10 - Optimization: Static Constants for Terminals
**Learning:** Dynamic list allocations like `[0,8,9,17,18,26] + list(range(27,34))` in hot paths cause unnecessary overhead. Using `frozenset` or `tuple` as constants avoids this.
**Action:** Replaced dynamic list constructions with global constants `TERMINALS_SET` and `YAO_INDICES_TUP`.
