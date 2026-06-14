
## 2024-06-14 - Optimize hot-path function returning static collections
**Learning:** Returning mutable lists from heavily used, small domain functions (like `get_suji` iterating 0-33) introduces allocation overhead and prevents caching. While the codebase memory mentions using `lru_cache` and `tuple` for `get_suji`, it was not universally applied.
**Action:** When a function maps a small, fixed set of inputs (e.g. Mahjong tile IDs) to small, static outputs, utilize `@functools.lru_cache` and return an immutable type like `tuple` instead of instantiating and appending to a `list` dynamically. Ensure the pattern is replicated across all duplicate files containing the logic.
