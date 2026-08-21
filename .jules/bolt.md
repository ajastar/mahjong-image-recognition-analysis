## 2024-05-23 - Memoizing get_suji with lru_cache
**Learning:** Functions with extremely small input domains (like Mahjong tiles 0-33) called frequently in hot paths should use `functools.lru_cache` and return immutable types (`tuple`) to prevent caching side effects and improve performance.
**Action:** Apply `@functools.lru_cache(maxsize=34)` to such utility functions and change the return type to `tuple`.
