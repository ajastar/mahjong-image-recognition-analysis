## 2024-05-24 - Avoid sum() with Generator Comprehensions for Early-Exit Logic
**Learning:** Using `sum()` with a generator expression (e.g., `sum(hand[t] for t in [list])`) for boolean existence checks over tiny fixed domains (like checking if any terminal tile exists in Mahjong) introduces severe iteration and function call overhead.
**Action:** Replace `sum()` and generator comprehensions with a manual `for` loop over a static tuple that leverages an early `break` when the condition is met, yielding a ~5x speedup in hot paths.
