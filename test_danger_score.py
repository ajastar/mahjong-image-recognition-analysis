from mahjong_logic import SafetyUtils

# Mock data
river_hist = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
riichi_status = [False, True, False, False]
visible_tiles_34 = [0] * 34
melds_arr = [[], [], [], []]
doras = [1, 27]
riichi_indices = [-1, 2, -1, -1]

# Test different tile types
test_tiles = [0, 8, 15, 27, 30]

for tile_idx in test_tiles:
    score = SafetyUtils.get_danger_score(
        tile_idx=tile_idx,
        river_hist=river_hist,
        riichi_status=riichi_status,
        visible_tiles_34=visible_tiles_34,
        turn=10,
        melds_arr=melds_arr,
        doras=doras,
        riichi_indices=riichi_indices
    )
    print(f"Danger score for tile {tile_idx}: {score}")
