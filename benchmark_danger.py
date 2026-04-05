import time
import copy
from mahjong_logic import SafetyUtils

def create_dummy_data():
    river_hist = [
        [0, 5, 8, 27, 30, 15, 12, 1, 9, 18],
        [1, 2, 3, 28, 31, 16, 10, 4, 11, 19],
        [4, 6, 7, 29, 32, 17, 13, 2, 8, 20],
        [8, 5, 2, 30, 33, 14, 11, 0, 7, 21]
    ]
    riichi_status = [False, True, False, False]
    visible_tiles_34 = [0] * 34
    for r in river_hist:
        for t in r:
            visible_tiles_34[t] += 1

    melds_arr = [
        [[1, 2, 3]],
        [[27, 27, 27], [31, 31, 31]],
        [],
        []
    ]
    doras = [5, 15, 27]
    riichi_indices = [-1, 7, -1, -1]

    return river_hist, riichi_status, visible_tiles_34, melds_arr, doras, riichi_indices

def run_benchmark():
    river_hist, riichi_status, visible_tiles_34, melds_arr, doras, riichi_indices = create_dummy_data()

    # We will test all 34 tiles to simulate real usage
    NUM_ITER = 10000

    start_time = time.time()
    total_danger = 0

    for _ in range(NUM_ITER):
        for tile_idx in range(34):
            # simulate calculating danger score for a tile
            danger = SafetyUtils.get_danger_score(
                tile_idx=tile_idx,
                river_hist=river_hist,
                riichi_status=riichi_status,
                visible_tiles_34=visible_tiles_34,
                turn=12,
                melds_arr=melds_arr,
                doras=doras,
                riichi_indices=riichi_indices
            )
            total_danger += danger

    end_time = time.time()

    print(f"Total Danger (Checksum): {total_danger:.2f}")
    print(f"Execution Time: {end_time - start_time:.4f} seconds for {NUM_ITER} iterations")

if __name__ == '__main__':
    run_benchmark()