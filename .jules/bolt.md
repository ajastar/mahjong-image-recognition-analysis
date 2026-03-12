## 2024-05-24 - OpenCV cvtColor in inner loops
**Learning:** Calling `cv2.cvtColor` (e.g., BGR to HSV) inside inner loops processing many small image crops creates massive overhead. This is a common performance bottleneck in Python/OpenCV computer vision pipelines.
**Action:** Perform a single full-image color space conversion once per frame before the loop, and pass the already-converted cropped regions into the processing functions.
