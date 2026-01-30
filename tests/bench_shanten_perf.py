
import time
import numpy as np
import sys
import os

# Insert path to allow imports
sys.path.insert(0, r"E:\AI_Project_Hub\Mahjong_Maker")

try:
    from mahjong_logic import ShantenUtils
    from numba_utils import calculate_shanten_jit
except ImportError:
    print("Could not import modules. Make sure you are in the correct directory.")
    sys.exit(1)

def benchmark():
    print("Benchmarking Shanten Calculation...")
    
    # Create a complex hand (1-shanten or 2-shanten usually triggers more search)
    # A hand with many possibilities is best to stress test
    # Chinitsu-like shape logic might trigger deep recursion
    hand = [0]*34
    # Manzu fullish
    for i in range(9): hand[i] = 1
    # Pinzu
    for i in range(9, 14): hand[i] = 1
    
    # Total 14 tiles
    # hand is 123456789m 12345p
    
    np_hand = np.array(hand, dtype=np.int8)
    
    # Warmup
    print("Warming up...")
    start = time.time()
    for _ in range(100):
        ShantenUtils.calculate_shanten(hand)
    print(f"Warmup done: {time.time() - start:.4f}s")
    
    ITER = 10000
    
    # Test 1: Numba JIT (directly calling numba function)
    print(f"Testing Numba JIT check ({ITER} iter)...")
    s = time.time()
    for _ in range(ITER):
        calculate_shanten_jit(np_hand)
    e = time.time()
    print(f"Numba JIT Time: {e-s:.4f}s (Avg: {(e-s)/ITER*1000:.4f} ms)")
    
    # Test 2: Python (Split Suit Memoization)
    # We need to hack USE_NUMBA to False to test the Python path
    import mahjong_logic
    mahjong_logic.USE_NUMBA = False
    
    print(f"Testing Python Logic ({ITER} iter)...")
    s = time.time()
    for _ in range(ITER):
        ShantenUtils.calculate_shanten(hand)
    e = time.time()
    print(f"Python Logic Time: {e-s:.4f}s (Avg: {(e-s)/ITER*1000:.4f} ms)")

if __name__ == "__main__":
    benchmark()
