## 2024-05-14 - Optimize small-domain utility functions
**Learning:** Returning hardcoded lists for simple functions like `get_suji` is faster than dynamically calling `.append()` and avoids Numba JIT compatibility issues compared to Python wrappers like `lru_cache`.
**Action:** For frequently called utility functions with tiny, bounded domains, avoid dynamically `.append()`-ing to lists. Return hardcoded list allocations (`[val1, val2]`) for fastest execution while strictly maintaining Numba compatibility. Do not use Python decorators on Numba target functions.
