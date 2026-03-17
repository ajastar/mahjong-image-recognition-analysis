## 2024-05-19 - cv2.cvtColor in small crops
**Learning:** Calling `cv2.cvtColor` (e.g., BGR to HSV) inside inner loops processing small image crops creates massive overhead.
**Action:** The optimization strategy is to perform a single full-image color space conversion once per frame before the loop, and pass the cropped regions from the pre-converted full image. This has been done in `server.py` but not in `single_bot.py`, `Jantama_Attach_Mode.py` and `Background_Multi_Mahjong.py`.
