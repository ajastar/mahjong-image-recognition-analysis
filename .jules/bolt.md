## 2024-05-24 - [Optimize suit_counts iteration]
**Learning:** In `mahjong_logic.py`, the `suit_counts` list tracks exactly 3 elements (representing the three main suits). Unrolling operations over it, such as boolean additions `(suit_counts[0] > 0) + (suit_counts[1] > 0) + (suit_counts[2] > 0)`, is mathematically safe and effectively avoids CPython generator and loop overhead in hot paths.
**Action:** When working with fixed-size small arrays (like length 3 `suit_counts`), manually unroll boolean additions or sums instead of using generators to significantly boost performance.
