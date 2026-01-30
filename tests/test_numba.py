import os
# Force safe threading layer
os.environ['NUMBA_THREADING_LAYER'] = 'workqueue'
import numpy as np

try:
    from numba import njit
    print("Numba imported successfully.")
except ImportError:
    print("Numba import failed.")
    exit(1)

@njit(nogil=True, cache=False) # Disable cache for testing
def test_jit(x):
    return x * 2

def run_test():
    print("Running compiled function...")
    res = test_jit(10)
    print(f"Result: {res}")
    
    arr = np.array([1, 2, 3], dtype=np.int8)
    @njit(nogil=True, cache=False)
    def test_arr(a):
        return a[0] + a[1]
    
    print("Running array function...")
    res2 = test_arr(arr)
    print(f"Result2: {res2}")

if __name__ == "__main__":
    run_test()
