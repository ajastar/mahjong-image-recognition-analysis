## 2024-08-14 - Optimize get_suji with lru_cache
**Learning:** Optimizing the `get_suji` function (which is duplicated across several files) by applying `@functools.lru_cache(maxsize=34)` and returning an immutable `tuple` instead of a list prevents cache pollution and side effects, yielding significant speedups. This function has a very small domain (0-33) and is called frequently in hot paths.
**Action:** Always check if small-domain utility functions used in hot paths can be memoized using `lru_cache` and returned as immutable types to prevent unintended side-effects.
