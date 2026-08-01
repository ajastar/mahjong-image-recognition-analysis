## 2024-05-24 - [Optimize get_suji with lru_cache and immutable return]
**Learning:** Functions with small input domains (e.g., 0-33 for Mahjong tiles) called frequently in hot paths should use `functools.lru_cache` and return immutable types (e.g., `tuple` instead of `list`) to avoid cache pollution and achieve speedups of ~35-40%.
**Action:** Apply `@functools.lru_cache` and return `tuple` instead of `list` for pure utility functions duplicated across the codebase.
