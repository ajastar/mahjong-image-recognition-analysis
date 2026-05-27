## 2024-05-27 - Loop Invariant Code Motion in Danger Scoring
**Learning:** In CPython, evaluating tile properties (like `tile_idx % 9`, `tile_idx // 9`, and `is_dora_neighbor` iteration) inside an opponent loop for a frequently called function (`get_danger_score`) adds measurable overhead.
**Action:** Always extract invariant calculations outside of loops that iterate over static bounds (like 4 opponents), especially in hot-path heuristic functions, to eliminate redundant operations and improve benchmark speed by ~10%.
