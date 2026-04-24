## 2024-06-25 - Python Generator vs List Comprehension Overhead in Hot Paths
**Learning:** In highly called internal loops like `mahjong_logic.py`'s scoring and evaluation functions, generator expressions inside `sum(...)` (e.g., `sum(1 for c in hand_34 if c >= 2)`) incur significant overhead.
**Action:** Replace `sum(...)` over generator comprehensions with `len([...])` combined with C-optimized list comprehensions (e.g., `len([c for c in hand_34 if c >= 2])`) or `sum([...])` over lists in performance critical sections to minimize evaluation overhead.
