
import sys
import os
import numpy as np

sys.path.insert(0, r"E:\AI_Project_Hub\Mahjong_Maker")

try:
    from numba_utils import simulate_chunk_jit
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def test_compile():
    print("Testing Numba Compilation...")
    seed = 123
    mh = np.zeros(34, dtype=np.int8)
    vc = np.zeros(34, dtype=np.int8)
    ci = np.array([0, 1], dtype=np.int32)
    cr = np.array([False, False], dtype=np.bool_)
    di = np.array([4], dtype=np.int8)
    
    print("Invoking JIT function...")
    try:
        res = simulate_chunk_jit(seed, mh, vc, ci, cr, 10, di)
        print("Success!")
    except Exception as e:
        print(f"Compilation Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_compile()
