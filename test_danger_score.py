import sys
from mahjong_logic import SafetyUtils

# Create some dummy data
river_hist = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
riichi_status = [False, False, False, False]
visible_tiles_34 = [0] * 34
melds_arr = [[], [], [], []]
doras = [1]
riichi_indices = [-1, -1, -1, -1]

# Test with a regular tile
score1 = SafetyUtils.get_danger_score(4, river_hist, riichi_status, visible_tiles_34, turn=10, melds_arr=melds_arr, doras=doras, riichi_indices=riichi_indices)
print(f"Score 1 (tile 4): {score1}")

# Test with an honor tile
score2 = SafetyUtils.get_danger_score(30, river_hist, riichi_status, visible_tiles_34, turn=10, melds_arr=melds_arr, doras=doras, riichi_indices=riichi_indices)
print(f"Score 2 (tile 30): {score2}")

# Test with a terminal
score3 = SafetyUtils.get_danger_score(8, river_hist, riichi_status, visible_tiles_34, turn=10, melds_arr=melds_arr, doras=doras, riichi_indices=riichi_indices)
print(f"Score 3 (tile 8): {score3}")

print("Test run successfully")
