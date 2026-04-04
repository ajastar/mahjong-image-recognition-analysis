from numba import njit
import numpy as np
import time

@njit(nogil=True, cache=False)
def is_agari_original_jit(hand_34):
    for head_idx in range(34):
        if hand_34[head_idx] >= 2:
            hand_34[head_idx] -= 2
            if _is_all_mentsu_jit(hand_34):
                hand_34[head_idx] += 2
                return True
            hand_34[head_idx] += 2
    return False

@njit(nogil=True, cache=False)
def is_agari_optimized_jit(hand_34):
    sum_m = np.sum(hand_34[0:9])
    sum_p = np.sum(hand_34[9:18])
    sum_s = np.sum(hand_34[18:27])
    sum_z = np.sum(hand_34[27:34])

    mod_m = sum_m % 3
    mod_p = sum_p % 3
    mod_s = sum_s % 3
    mod_z = sum_z % 3

    if (mod_m == 1 or mod_p == 1 or mod_s == 1 or mod_z == 1) or (mod_m + mod_p + mod_s + mod_z != 2):
        return False

    for head_idx in range(34):
        if hand_34[head_idx] >= 2:
            suit = head_idx // 9
            if suit == 0 and mod_m != 2: continue
            if suit == 1 and mod_p != 2: continue
            if suit == 2 and mod_s != 2: continue
            if suit >= 3 and mod_z != 2: continue

            hand_34[head_idx] -= 2
            if _is_all_mentsu_jit(hand_34):
                hand_34[head_idx] += 2
                return True
            hand_34[head_idx] += 2
    return False

@njit(nogil=True, cache=False)
def _is_all_mentsu_jit(hand):
    if not _decompose_suit_boolean(hand, 0): return False
    if not _decompose_suit_boolean(hand, 9): return False
    if not _decompose_suit_boolean(hand, 18): return False
    for i in range(27, 34):
        if hand[i] % 3 != 0: return False
    return True

@njit(nogil=True, cache=False)
def _decompose_suit_boolean(hand, start_idx):
    counts = hand[start_idx : start_idx+9].copy()
    return _recurse_boolean(counts, 0)

@njit(nogil=True, cache=False)
def _recurse_boolean(counts, idx):
    if idx >= 9:
        for i in range(9):
            if counts[i] != 0: return False
        return True
    if counts[idx] == 0: return _recurse_boolean(counts, idx + 1)
    if counts[idx] >= 3:
        counts[idx] -= 3
        if _recurse_boolean(counts, idx): return True
        counts[idx] += 3
    if idx < 7:
        if counts[idx] > 0 and counts[idx+1] > 0 and counts[idx+2] > 0:
            counts[idx] -= 1; counts[idx+1] -= 1; counts[idx+2] -= 1
            if _recurse_boolean(counts, idx): return True
            counts[idx] += 1; counts[idx+1] += 1; counts[idx+2] += 1
    return False

# Generate some hands
np.random.seed(42)
hands = []
for _ in range(100000):
    hand = np.zeros(34, dtype=np.int8)
    for _ in range(14):
        hand[np.random.randint(0, 34)] += 1
    hands.append(hand)

# Valid hand
valid_hand = np.zeros(34, dtype=np.int8)
valid_hand[0] = 3 # 1m x3
valid_hand[9] = 1 # 1p x1
valid_hand[10] = 1 # 2p x1
valid_hand[11] = 1 # 3p x1
valid_hand[18] = 3 # 1s x3
valid_hand[27] = 3 # East x3
valid_hand[31] = 2 # Haku x2
hands.append(valid_hand)

# Warmup
is_agari_original_jit(hands[0])
is_agari_optimized_jit(hands[0])

start = time.time()
for h in hands:
    is_agari_original_jit(h)
print("Original time:", time.time() - start)

start = time.time()
for h in hands:
    is_agari_optimized_jit(h)
print("Optimized time:", time.time() - start)
