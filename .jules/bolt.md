
## 2024-06-08 - Hoisting Loop-Invariant Computations
**Learning:** In highly called inner loops like opponent iteration inside `get_danger_score`, variables that depend entirely on the parent loop parameters (like `tile_idx` vs `r_player`) create redundant executions. Inlining modulo operations, truthiness checks, and boolean combinations for every iteration significantly degrades performance over thousands of tile evaluations.
**Action:** Always extract static variable conversions (`tile_idx % 9`, `tile_idx // 9`, `tile_idx in list`) *before* looping through opponents, and use explicit pre-computed booleans inside inner evaluation branches.
