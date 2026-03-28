## 2024-05-20 - [Redundant Resize Calls in TemplateMatcher]
**Learning:** During high-frequency template matching loops (like scanning HUDs), calling `cv2.resize` on the template inside the loop on every match attempt creates unnecessary overhead, as the target image size generally remains constant for a given execution context.
**Action:** Implement a caching mechanism for resized templates using `(label, w, h)` as the key to store and reuse resized templates, significantly reducing computation time.
