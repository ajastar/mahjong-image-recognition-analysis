## 2024-06-22 - [Optimize get_suji with lru_cache]
**Learning:** Functions with a small finite input domain (like 0-33 for Mahjong tiles) called frequently inside hot paths (e.g. during encode_state_god and evaluate loops) should be memoized. Returning mutable objects (lists) creates a risk for cache pollution if callers modify them.
**Action:** Use `@functools.lru_cache(maxsize=34)` for tile property functions and ensure they return immutable types (like `tuple` instead of `list`) to guarantee cache safety.
