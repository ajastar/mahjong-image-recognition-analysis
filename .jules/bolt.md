## 2024-05-15 - [Pre-calculating invariant tile properties]
**Learning:** [Pre-calculating tile properties (like suit, number, dora neighbor status) outside of opponent loops in danger scoring functions (like get_danger_score) yields measurable performance improvements by eliminating redundant computation.]
**Action:** [Always look for invariant properties that are tied to the action or tile being evaluated rather than the opponent, and hoist these calculations out of player iteration loops.]
