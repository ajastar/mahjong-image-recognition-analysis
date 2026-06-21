
## 2024-06-21 - [Optimize small-domain static returns with lru_cache]
**Learning:** Functions with a small input domain (e.g., 0-33 for Mahjong tiles) that map to static values and are called frequently in hot paths are prime targets for memoization. Additionally, returning immutable types like `tuple` instead of `list` prevents cache pollution side-effects and reduces memory allocation overhead.
**Action:** When implementing helper functions like `get_suji` that have limited integer inputs, apply `@functools.lru_cache` and ensure the return value is immutable (like `tuple`). Avoid returning `list` or dynamically appending to them in performance-critical paths.
