## 2024-05-20 - Cache resized templates in OpenCV matching
**Learning:** Repetitive `cv2.resize` operations within a tight matching loop on the same templates and target dimensions introduce unnecessary computational overhead. While `cv2.resize` is fast for a single invocation, repeatedly doing it across many templates per frame creates a cumulative performance bottleneck.
**Action:** When performing `cv2.matchTemplate` against a suite of templates across frames where the target sizes repeat, always cache the resized templates keyed by `(label, width, height)` to bypass redundant resizing.
