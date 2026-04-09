
## 2024-04-09 - Avoid sum(generator) in Mahjong Logic Hot Paths
**Learning:** Python generator comprehensions inside `sum()` (e.g., `sum(1 for c in hand_34 if c >= 2)`) incur significant overhead in this project's hot-path game evaluation loops (like shanten and agari calculations). They generate local functions and yield overhead.
**Action:** Replace `sum(generator_comprehension)` with explicit `for` loops and accumulator variables for measurable performance gains (~1.5x to 3x speedups in micro-benchmarks), without overly sacrificing code readability.
