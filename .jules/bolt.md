## 2024-05-19 - Removed cvtColor Bottleneck
**Learning:** Frequent calls to `cv2.cvtColor` (BGR to HSV) inside inner loops processing small image crops create massive overhead.
**Action:** Perform a single full-image color space conversion once per frame before the loop, and pass the cropped regions from the pre-converted full image. Applied to `is_back_side_tile` across `single_bot.py`, `Jantama_Attach_Mode.py`, and `Background_Multi_Mahjong.py`.
