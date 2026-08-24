
## 2024-05-19 - [Optimize hot-path domain-specific calculations with LRU cache]
**Learning:** Functions with small, discrete input domains (e.g., Mahjong tiles 0-33) that are called heavily in hot paths benefit massively from `@functools.lru_cache`. Additionally, returning immutable structures like `tuple` avoids both list allocation overhead and prevents downstream mutation bugs.
**Action:** Always check if a frequently invoked static utility function has a small finite input range. If it does, apply `lru_cache`, ensure the output is immutable, and verify there are no downstream list mutability assumptions.

## 2024-05-19 - [Numba Compatibility & Tuple Optimization]
**Learning:** Returning immutable structs like `tuple` instead of instantiating `list` inside small utility functions yields measurable performance improvements without the overhead of `lru_cache`, while keeping it safe and fully compatible with Numba JIT compilation (`@njit`), which rejects Python decorators like `@functools.lru_cache`.
**Action:** When optimizing utility functions in environments executing Numba JIT compilations, strictly avoid standard Python decorators and prefer returning immutable structs to gain execution speed without risking `TypingError` regressions.
