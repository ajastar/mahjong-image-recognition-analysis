
import time
import numpy as np
import sys
import os

sys.path.insert(0, r"E:\AI_Project_Hub\Mahjong_Maker")

try:
    from mahjong_logic import PIMCEngine
    # Ensure Numba is warm
    print("Pre-warmup...")
    PIMCEngine._warmup_numba()
    print("Post-warmup...")
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

def verify_pimc():
    print("=== Verifying PIMC Logic & Performance ===")
    
    # Setup Scenario: Tenpai Hand (Waiting for 6s/9s)
    # Hand: 1,2,3m 4,5,6p 7,8s 1,1z (East) + 1 free
    # Let's say we hold 7,8s. Waiting 6s, 9s.
    # We hold 13 tiles + draw.
    
    # 123m
    hand_34 = [0]*34
    hand_34[0]=1; hand_34[1]=1; hand_34[2]=1
    # 456p
    hand_34[12]=1; hand_34[13]=1; hand_34[14]=1
    # 78s
    hand_34[24]=1; hand_34[25]=1
    # 111z (East pon) - Assume hidden for PIMC simplicity or just in hand
    hand_34[27]=3
    # Pair 2z
    hand_34[28]=2
    
    # Total 13 tiles.
    # Draw: 5s (useless neighbor)
    hand_34[22] = 1 
    
    # Discard Candidates: 
    # 1. Discard 5s (Back to Tenpai 6-9s) -> Should have high score
    # 2. Discard 2z (Break pair) -> No Tenpai -> Low score
    
    visible_counts = [0]*34
    # Set some visible
    visible_counts[27] = 1 # 1 East out
    
    hands_arr = [[0]*34 for _ in range(4)]
    hands_arr[3] = hand_34
    
    candidates = [
        {"idx": 22, "label": "5s (Tenpai)", "base_conf": 0.9},
        {"idx": 28, "label": "2z (Break)", "base_conf": 0.1}
    ]
    
    doras_ids = [4] # 5m
    
    NUM_WORLDS = 1000
    
    print(f"\nRunning {NUM_WORLDS} simulations...")
    s = time.time()
    
    results = PIMCEngine.run(hands_arr, visible_counts, candidates, doras_ids, num_worlds=NUM_WORLDS)
    
    e = time.time()
    print(f"\nTotal Time: {e-s:.4f}s")
    
    print("\n=== Results ===")
    for c in results:
        print(f"Cand: {c['label']:<15} | Score: {c.get('pimc_score', 0):.2f}")
        
    # Check Logic
    score_tenpai = next((c['pimc_score'] for c in results if c['idx'] == 22), 0)
    score_break = next((c['pimc_score'] for c in results if c['idx'] == 28), 0)
    
    if score_tenpai > score_break:
        print("\n[SUCCESS] Logic Check Passed: Tenpai candidate has higher score.")
    else:
        print("\n[FAILURE] Logic Check Failed: Tenpai score is not higher.")
        
    if score_tenpai == 0 and score_break == 0:
        print("[FAILURE] All scores are ZERO. Simulation might be broken.")
    else:
        print("[SUCCESS] Non-zero scores detected.")

if __name__ == "__main__":
    verify_pimc()
