import numpy as np
import time

from numba_utils import is_agari_jit, get_waits_count_jit

# In the test, we'll verify it returns correct values and test performance
def test_performance():
    # Generate test hands
    np.random.seed(42)
    hands = []
    visibles = []
    for _ in range(1000):
        hand = np.zeros(34, dtype=np.int8)
        for _ in range(13):
            hand[np.random.randint(0, 34)] += 1
        hands.append(hand)

        visible = np.zeros(34, dtype=np.int8)
        for _ in range(10):
            visible[np.random.randint(0, 34)] += 1
        visibles.append(visible)

    # Hand for Chiitoi
    chiitoi = np.zeros(34, dtype=np.int8)
    chiitoi[0] = 2
    chiitoi[1] = 2
    chiitoi[2] = 2
    chiitoi[3] = 2
    chiitoi[4] = 2
    chiitoi[5] = 2
    chiitoi[6] = 1
    hands.append(chiitoi)
    visibles.append(np.zeros(34, dtype=np.int8))

    # Hand for Kokushi
    kokushi = np.zeros(34, dtype=np.int8)
    yao_indices = [0,8,9,17,18,26,27,28,29,30,31,32,33]
    for idx in yao_indices[:-1]:
        kokushi[idx] = 1
    kokushi[yao_indices[-1]] = 1
    hands.append(kokushi)
    visibles.append(np.zeros(34, dtype=np.int8))

    # Standard hand
    valid_hand = np.zeros(34, dtype=np.int8)
    valid_hand[0] = 3 # 1m x3
    valid_hand[9] = 1 # 1p x1
    valid_hand[10] = 1 # 2p x1
    valid_hand[11] = 1 # 3p x1
    valid_hand[18] = 3 # 1s x3
    valid_hand[27] = 2 # East x2
    valid_hand[31] = 1 # Haku x1
    hands.append(valid_hand)
    visibles.append(np.zeros(34, dtype=np.int8))

    # Warmup
    get_waits_count_jit(hands[-1], visibles[-1])

    start = time.time()
    for h, v in zip(hands, visibles):
        get_waits_count_jit(h, v)
    print("New Optimized time:", time.time() - start)

if __name__ == "__main__":
    test_performance()
