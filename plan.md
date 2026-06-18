1. **Define `TERMINALS_SET` at the module level in `mahjong_logic.py`**
   - Create a module-level constant: `TERMINALS_SET = frozenset([0,8,9,17,18,26,27,28,29,30,31,32,33])` near the top of the file.
   - This avoids recreating the list `[0,8,9,17,18,26] + list(range(27,34))` multiple times dynamically during hot paths.

2. **Replace list generation in `get_danger_score`**
   - In `get_danger_score`, replace the generator sum:
     `terminals = sum(hand_34[t] for t in [0,8,9,17,18,26] + list(range(27,34)))`
     with an unrolled sum using explicitly defined indices for maximum performance:
     `terminals = hand_34[0] + hand_34[8] + hand_34[9] + hand_34[17] + hand_34[18] + hand_34[26] + hand_34[27] + hand_34[28] + hand_34[29] + hand_34[30] + hand_34[31] + hand_34[32] + hand_34[33]`

3. **Replace list generation in other functions in `mahjong_logic.py`**
   - Replace local `terminals = [0,8,9,17,18,26] + list(range(27,34))` definitions with `terminals = TERMINALS_SET` inside `StructureUtils.evaluate_structure`, `AgariUtils.get_hand_han`, and `Strategy.estimate_hand_value`.

4. **Add journal entry**
   - Document the performance learning of flattening dynamically generated static lists and replacing generator comprehensions inside `sum()` with unrolled index addition or module-level frozensets for membership testing in `mahjong_logic.py`.

5. **Complete pre-commit steps**
   - Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.

6. **Submit PR**
   - Submit the PR with the required performance improvement format (`⚡ Bolt: [performance improvement]`).
