import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mahjong_logic import PIMCEngine, TILE_MAP, USE_NUMBA

def debug_pimc():
    print(f"Debug Configuration: USE_NUMBA={USE_NUMBA}")
    
    # Hand: Tenpai (Wait 6s/9s) to ensure fast win
    # 345m 345p 345s 11z 66s + 6s (Win)
    
    # Let's use 1-shanten
    # Hand: 14 tiles (Tenpai-to-be)
    # 345m 345p 345s 11z 68s + 2z
    hand_str_14 = ["3m","4m","5m", "3p","4p","5p", "3s","4s","5s", "1z","1z", "6s","8s", "2z"]
    
    hands_arr = [[0]*34 for _ in range(4)]
    my_hand = [0]*34
    for s in hand_str_14:
        my_hand[TILE_MAP[s]] += 1
    hands_arr[3] = my_hand 
    
    # Manual Agari Check
    print("--- Manual Agari Check ---")
    from mahjong_logic import AgariUtils
    test_hand = my_hand[:]
    test_hand[TILE_MAP["2z"]] -= 1 # Discard 2z
    test_hand[TILE_MAP["7s"]] += 1 # Draw 7s (Win)
    is_win = AgariUtils.is_agari(test_hand)
    print(f"Discard 2z, Draw 7s -> Agari? {is_win}")
    print("--------------------------")
    
    visible_counts = [0]*34
    
    candidates = [{"idx": TILE_MAP["2z"], "label": "2z", "base_conf": 0.8}]
    doras_ids = [TILE_MAP["1m"]]
    
    print("Running 10 simulations...")
    results = PIMCEngine.run(hands_arr, visible_counts, candidates, doras_ids, num_worlds=10)
    
    with open("debug_log.txt", "w") as f:
        f.write(str(results))
    print("Results:", results)

if __name__ == "__main__":
    debug_pimc()
