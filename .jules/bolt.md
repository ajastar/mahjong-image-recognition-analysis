
## 2024-05-23 - [Mahjong Agari Head Constraint]
**Learning:** Checking every possible 2+ count tile as a potential head is a massive waste of cycles for an algorithm called inside PIMC simulation loops. Because a hand consists of 14 tiles grouped as 4 sets of 3 plus 1 set of 2, the exact suit containing the head must mathematically hold a total tile count of `2 (mod 3)`. All other suits MUST be `0 (mod 3)`. Any `1 (mod 3)` renders the hand invalid instantly.
**Action:** Always consider the overarching mathematical constraints of a domain specific problem to prune search trees *before* attempting recursive combinations.
