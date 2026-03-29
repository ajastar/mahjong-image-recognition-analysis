## 2024-05-14 - Optimize NakiUtils.check_calls
**Learning:** Checking for prioritization when the elements are already added in priority order is redundant logic that introduces O(N) penalties.
**Action:** Remove redundant `any()` membership checks and loops completely and simply return `actions[0]` after checking for `RON`.
