## 2024-05-24 - Optimize BGR to HSV conversion in tile detection
**Learning:** Frequent calls to `cv2.cvtColor` (e.g., BGR to HSV) inside inner loops processing small image crops create massive overhead in Python/OpenCV.
**Action:** Perform a single full-image color space conversion once per frame before the loop, and pass the cropped regions from the pre-converted full image.
