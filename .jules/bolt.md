## 2024-05-24 - [Math-based Pruning in Hot Paths]
**Learning:** Checking mod-3 properties of intra-suit sums effectively rejects 90% of invalid hands in ~0.5 microseconds without evaluating computationally expensive state decomposition.
**Action:** Exploit rigid mathematical constraints specific to the simulation domain to aggressively fast-fail edge states prior to complex logical checks.
