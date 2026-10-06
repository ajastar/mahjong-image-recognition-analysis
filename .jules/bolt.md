## 2024-05-24 - Precomputing small-domain arrays and returning tuples
**Learning:** Returning a global mutable list from a precomputed lookup cache can lead to side-effects if callers mutate it.
**Action:** When precomputing lists for O(1) lookup in Python, always return immutable objects like `tuple`s, and ensure the initialization code remains readable.
