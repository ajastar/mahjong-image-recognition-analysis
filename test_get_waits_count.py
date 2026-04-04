from numba import njit
import numpy as np
import time

# Copy dependencies
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

@njit(nogil=True, cache=False)
def is_agari_jit(hand_34):
    # Simplified just for performance testing of the loop logic
    for head_idx in range(34):
        if hand_34[head_idx] >= 2:
            hand_34[head_idx] -= 2
            if _is_all_mentsu_jit(hand_34):
                hand_34[head_idx] += 2
                return True
            hand_34[head_idx] += 2
    return False

# ----------------- Target Functions -----------------

@njit(nogil=True, cache=False)
def get_waits_count_original(hand_34_arr, visible_34_arr):
    total_waits = 0
    temp_hand = hand_34_arr.copy()
    for t_id in range(34):
        if temp_hand[t_id] >= 4: continue
        temp_hand[t_id] += 1
        if is_agari_jit(temp_hand):
            left = 4 - (temp_hand[t_id] - 1) - visible_34_arr[t_id]
            if left > 0: total_waits += left
        temp_hand[t_id] -= 1
    return total_waits

@njit(nogil=True, cache=False)
def get_waits_count_optimized(hand_34_arr, visible_34_arr):
    total_waits = 0
    temp_hand = hand_34_arr.copy()

    # Pre-calculate suit sums
    sum_m = np.sum(temp_hand[0:9])
    sum_p = np.sum(temp_hand[9:18])
    sum_s = np.sum(temp_hand[18:27])
    sum_z = np.sum(temp_hand[27:34])

    for t_id in range(34):
        if temp_hand[t_id] >= 4: continue

        # The trick: adding a tile changes the mod3 of its suit.
        # Check if the new sum matches the suit criteria before copying.
        new_mod_m = (sum_m + (1 if t_id < 9 else 0)) % 3
        new_mod_p = (sum_p + (1 if 9 <= t_id < 18 else 0)) % 3
        new_mod_s = (sum_s + (1 if 18 <= t_id < 27 else 0)) % 3
        new_mod_z = (sum_z + (1 if t_id >= 27 else 0)) % 3

        # For a 14-tile hand, exact one suit has mod3 == 2, others == 0.
        # This translates to mod_m + mod_p + mod_s + mod_z == 2
        # And none of them are 1.
        if (new_mod_m == 1 or new_mod_p == 1 or new_mod_s == 1 or new_mod_z == 1) or \
           (new_mod_m + new_mod_p + new_mod_s + new_mod_z != 2):
            continue

        temp_hand[t_id] += 1
        if is_agari_jit(temp_hand):
            left = 4 - (temp_hand[t_id] - 1) - visible_34_arr[t_id]
            if left > 0: total_waits += left
        temp_hand[t_id] -= 1
    return total_waits

# Generate test data
np.random.seed(42)
hands = []
visibles = []
for _ in range(10000):
    hand = np.zeros(34, dtype=np.int8)
    # Tenpai hand (13 tiles)
    for _ in range(13):
        hand[np.random.randint(0, 34)] += 1
    hands.append(hand)

    visible = np.zeros(34, dtype=np.int8)
    for _ in range(10):
        visible[np.random.randint(0, 34)] += 1
    visibles.append(visible)

# Warmup
get_waits_count_original(hands[0], visibles[0])
get_waits_count_optimized(hands[0], visibles[0])

start = time.time()
for h, v in zip(hands, visibles):
    get_waits_count_original(h, v)
print("Original time:", time.time() - start)

start = time.time()
for h, v in zip(hands, visibles):
    get_waits_count_optimized(h, v)
print("Optimized time:", time.time() - start)

for h, v in zip(hands, visibles):
    assert get_waits_count_original(h, v) == get_waits_count_optimized(h, v)

print("All tests passed!")
