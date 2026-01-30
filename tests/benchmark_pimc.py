import sys
import os
import time
import sys

# Add parent directory to path to import mahjong_logic
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mahjong_logic import PIMCEngine, TILE_MAP, USE_NUMBA

def benchmark_pimc():
    print(f"Benchmark Configuration:")
    print(f"  USE_NUMBA: {USE_NUMBA}")
    
    # Setup Hand (1-shanten standard)
    # 123m 456p 78s 11z 222z + 5s (wait for 6s/9s etc)
    # Let's make a clear 1-shanten hand
    # Hand: 123m 456p 78s 55z 666z (Tenpai for 6s/9s if we discard something? No)
    # Let's use the one from code:
    # 345m 345p 345s 11z 23z -> 1-shanten
    
    # 13 tiles
    # Hand: 14 tiles (Tenpai-to-be)
    # 345m 345p 345s 11z 68s + 2z
    # Discard 2z -> 345m 345p 345s 11z 68s (Tenpai, wait 7s)
    # This has high win probability.
    hand_str_14 = ["3m","4m","5m", "3p","4p","5p", "3s","4s","5s", "1z","1z", "6s","8s", "2z"]
    hands_arr = [[0]*34 for _ in range(4)]
    
    my_hand = [0]*34
    for s in hand_str_14:
        my_hand[TILE_MAP[s]] += 1
    hands_arr[3] = my_hand # 14 tiles
    
    # Visible counts (Empty)
    visible_counts = [0]*34
    
    candidates = [
        {"idx": TILE_MAP["2z"], "label": "2z", "base_conf": 0.8}, # Good
        # {"idx": TILE_MAP["6s"], "label": "6s", "base_conf": 0.1}, # Bad
        # {"idx": TILE_MAP["1z"], "label": "1z", "base_conf": 0.1}  # Bad
    ]
    
    doras_ids = [TILE_MAP["1m"]]
    
    # Manual Agari Check
    print("--- Manual Agari Check ---")
    from mahjong_logic import AgariUtils
    test_hand = my_hand[:]
    test_hand[TILE_MAP["2z"]] -= 1 # Discard 2z
    test_hand[TILE_MAP["7s"]] += 1 # Draw 7s (Win)
    is_win = AgariUtils.is_agari(test_hand)
    print(f"Discard 2z, Draw 7s -> Agari? {is_win}")
    print("--------------------------")
    
    print(f"Candidates count: {len(candidates)}")
    for c in candidates:
        print(f"Cand: {c['label']}")

    print("\n[Cold Run] Starting PIMC Benchmark (1500 simulations)...")
    s_time = time.time()
    results = PIMCEngine.run(hands_arr, visible_counts, candidates, doras_ids, num_worlds=1500)
    e_time = time.time()
    cold_elapsed = e_time - s_time
    print(f"Cold Execution Time (inc. warmup): {cold_elapsed:.4f} seconds")

    print("\n[Warm Run] Starting PIMC Benchmark (1500 simulations)...")
    s_time = time.time()
    results = PIMCEngine.run(hands_arr, visible_counts, candidates, doras_ids, num_worlds=1500)
    e_time = time.time()
    warm_elapsed = e_time - s_time
    print(f"Warm Execution Time: {warm_elapsed:.4f} seconds")
    
    with open("benchmark_final.txt", "w") as f:
        f.write(f"Cold Execution Time (inc. warmup): {cold_elapsed:.4f} seconds\n")
        f.write(f"Warm Execution Time: {warm_elapsed:.4f} seconds\n")
        f.write("Results:\n")
        for c in results:
            f.write(f"  {c['label']}: Score={c.get('pimc_score', 0):.2f}\n")
            
        if warm_elapsed < 1.5:
            f.write("\n[PASS] Performance Excellent (< 1.5s)\n")
        elif warm_elapsed < 3.0:
            f.write("\n[PASS] Performance Good (< 3.0s)\n")
        else:
            f.write("\n[FAIL] Performance Slow (> 3.0s)\n")
            
    # Also print to stdout
    print(open("benchmark_final.txt").read())

if __name__ == "__main__":
    benchmark_pimc()
