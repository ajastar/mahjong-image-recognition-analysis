
## 2024-05-20 - Optimize redundant cv2.cvtColor in tight computer vision loops
**Learning:** Performing `cv2.cvtColor` on multiple small cropped images individually inside an inner loop causes significant overhead compared to converting the entire image once before the loop and passing the pre-converted image crops.
**Action:** When extracting and processing multiple regions of interest from a single image that require a different color space (e.g., BGR to HSV), perform `cvtColor` on the full frame once, then slice the pre-converted array for the individual regions.
