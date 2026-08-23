## 2024-08-23 - Optimize get_suji caching
**Learning:** Functions with small input domains (like 34 Mahjong tiles) called frequently in hot paths should return immutable tuples and use `functools.lru_cache` to avoid dynamic list allocation overhead.
**Action:** Always prefer tuples and `lru_cache` for pure lookup functions with small integer domains in simulation loops.
