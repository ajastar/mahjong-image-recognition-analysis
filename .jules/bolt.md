## 2024-07-02 - Optimize terminal tile checks with static sets and early exits
**Learning:** Checking for elements in dynamically generated lists (`[0,8,9,17,18,26] + list(range(27,34))`) and using generator expressions inside `sum()` inside Python hot paths causes significant overhead.
**Action:** Always use explicit loops with early exits for boolean tests and module-level `frozenset` objects for O(1) membership testing in frequently executed calculation logic (like Yaku and heuristic scores).
