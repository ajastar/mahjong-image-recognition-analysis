## 2024-08-09 - Caching Small-Domain Utility Functions
**Learning:** Functions with small input domains (like `get_suji` mapping 0-33 for Mahjong tiles) called frequently in hot paths can achieve significant speedups (~35-40%) by using `@functools.lru_cache` and returning immutable types (e.g., `tuple` instead of `list`).
**Action:** Always consider `lru_cache` for pure utility functions with small, bounded input domains in hot paths, and ensure they return immutable types to prevent cache pollution and downstream mutations.
