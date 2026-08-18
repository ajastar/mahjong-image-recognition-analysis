## 2024-05-24 - [Optimize get_suji with LRU cache]
**Learning:** Functions like `get_suji` in `mahjong_logic.py` and other files operate on a very small domain (34 tiles) but are called frequently. Using `functools.lru_cache(maxsize=34)` provides significant speedups.
**Action:** Always consider `lru_cache` for pure functions with a small input domain that are called frequently in hot paths. Remember to change mutable returns (like `list`) to immutable ones (like `tuple`) when using `lru_cache`.
