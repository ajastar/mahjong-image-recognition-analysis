## 2024-05-24 - HSV conversion overhead in loops
**Learning:** Calling `cv2.cvtColor` (BGR to HSV) repeatedly inside inner loops on small image crops creates massive overhead.
**Action:** Always perform a single full-image color space conversion once per frame before the loop, and pass the cropped regions from the pre-converted full image.
