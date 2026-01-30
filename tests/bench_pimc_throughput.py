
import time
import numpy as np
import sys
import os

sys.path.insert(0, r"E:\AI_Project_Hub\Mahjong_Maker")

try:
    from mahjong_logic import PIMCEngine, TILE_MAP
    # Ensure Numba is warm
    PIMCEngine._warmup_numba()
except ImportError:
    print("Import Error")
    sys.exit(1)

def benchmark_pimc():
    print("Benchmarking PIMC Throughput...")
    
    # 1. Setup Scenario
    # Hand: 1,2,3,4,5,6m 4,5p 6,7s 1,1z (East) + 1 free
    # Candidates: Discard 1m, 4m, 4p, 6s, 1z
    
    hand_34 = [0]*34
    # 1-6m
    for i in range(6): hand_34[i] = 1
    # 4-5p
    hand_34[12] = 1; hand_34[13] = 1
    # 6-7s
    hand_34[23] = 1; hand_34[24] = 1
    # 1z (East) pair
    hand_34[27] = 2
    # +1 random tsumo (say 8s)
    hand_34[25] = 1
    
    # Total 14 tiles
    
    visible_counts = [0]*34
    # Some discards
    visible_counts[0] = 1
    visible_counts[9] = 2
    
    # Candidates
    candidates = [
        {"idx": 0, "label": "1m", "base_conf": 0.5},
        {"idx": 12, "label": "4p", "base_conf": 0.3},
        {"idx": 23, "label": "6s", "base_conf": 0.1},
        {"idx": 27, "label": "1z", "base_conf": 0.1},
    ]
    
    hands_arr = [[0]*34 for _ in range(4)]
    hands_arr[3] = hand_34
    
    doras_ids = [4] # 5m dora
    
    NUM_WORLDS = 1500
    
    print(f"Running {NUM_WORLDS} simulations on full PIMC engine...")
    s = time.time()
    
    results = PIMCEngine.run(hands_arr, visible_counts, candidates, doras_ids, num_worlds=NUM_WORLDS)
    
    e = time.time()
    print(f"PIMC Total Time: {e-s:.4f}s")
    print("Top candidate:", results[0]["label"] if results else "None")

if __name__ == "__main__":
    benchmark_pimc()
