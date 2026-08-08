## 2024-08-08 - [Optimize get_suji with lru_cache and immutable return]
**Learning:** Functions with small input domains (like 34 mahjong tiles) that are called repeatedly in hot paths benefit greatly from caching, but returning mutable objects like lists from a cache can lead to cache pollution.
**Action:** Always return immutable structures (e.g. tuples) when caching domain-specific static lookups to ensure both speed and safety across simulation instances.
