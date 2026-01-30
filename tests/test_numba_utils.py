import sys
import os
import numpy as np
os.environ['NUMBA_THREADING_LAYER'] = 'workqueue'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from numba_utils import calculate_shanten_jit, is_agari_jit
    print("Imported numba_utils")
except Exception as e:
    print(f"Import failed: {e}")
    exit(1)

def test_utils():
    # Test Agari
    print("Testing is_agari_jit...")
    hand = np.zeros(34, dtype=np.int8)
    # 111 222 333 444 55 (Agari)
    hand[0:3] = 3
    hand[3:6] = 3
    hand[6:9] = 3
    hand[9:12] = 3
    hand[12] = 2
    
    res = is_agari_jit(hand)
    print(f"is_agari result: {res}")
    
    # Test Shanten
    print("Testing calculate_shanten_jit...")
    hand2 = np.zeros(34, dtype=np.int8)
    hand2[0:3] = 3
    hand2[3:6] = 3
    hand2[6:9] = 3
    hand2[9:11] = 2 # 11
    hand2[12] = 1 # 1
    # 111 222 333 44 5 (Wait for 4 or 5) -> Tenpai (0 shanten)
    
    res2 = calculate_shanten_jit(hand2)
    print(f"calculate_shanten result: {res2}")

if __name__ == "__main__":
    test_utils()
