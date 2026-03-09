## 2024-05-18 - [Optimize cv2.cvtColor in is_back_side_tile]
**Learning:** Frequent small calls to `cv2.cvtColor` (like BGR to HSV) inside inner loops processing small image crops create massive overhead.
**Action:** Always perform single full-image color space conversions once per frame before the loop, and pass the cropped regions from the pre-converted full image into the inner loop logic.
