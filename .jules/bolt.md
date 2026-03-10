## 2024-03-10 - [Optimize cvtColor for inner loop crops]
**Learning:** Frequent calls to `cv2.cvtColor` (e.g., BGR to HSV) inside inner loops processing small image crops create massive overhead in this specific CV logic pattern.
**Action:** Perform a single full-image color space conversion once per frame before the loop, and pass the cropped regions from the pre-converted full image. This drastically reduces the overhead of invoking `cvtColor` repeatedly.
