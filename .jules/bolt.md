## 2024-05-18 - [Optimize `get_suji` caching]
**Learning:** Functions with small input domains (e.g., 0-33 for Mahjong tiles) called frequently in hot paths should use `functools.lru_cache` and return immutable types (e.g., `tuple` instead of `list`) to achieve speedups while ensuring cache safety.
**Action:** Always identify frequently called functions that rely on small input parameters and consider using LRU cache alongside immutable types to improve performance.
