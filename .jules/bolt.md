## 2024-07-26 - Optimize get_suji with lru_cache
**Learning:** The `get_suji` function is frequently called across multiple files and can be optimized using `@functools.lru_cache(maxsize=34)` returning a tuple instead of dynamically creating a list on each call. This results in ~1.5x speedup.
**Action:** Always verify immutable return types for `lru_cache` to prevent cache pollution and ensure all duplicated function declarations are updated with the required import.
