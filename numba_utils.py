import numpy as np
from numba import njit
import random

# ----------------------------------------------------------------------
# ⚡ Numba Accelerated Mahjong Logic (Full PIMC Loop)
# ----------------------------------------------------------------------

@njit(nogil=True, cache=False)
def simulate_chunk_jit(seed, my_hand_34, visible_counts_34, cand_indices, cand_is_riichi, sim_limit, doras_ids):
    """
    PIMC Simulation Loop completely in Numba
    
    Returns:
        results_wins: [num_candidates] (int)
        results_scores: [num_candidates] (float/int)
    """
    np.random.seed(seed)
    
    num_cands = len(cand_indices)
    
    # Results accumulators
    res_wins = np.zeros(num_cands, dtype=np.int32)
    res_scores = np.zeros(num_cands, dtype=np.float64)
    # res_turns = np.zeros(num_cands, dtype=np.int32) # Not strictly needed for logic, but maybe for analysis?
    
    # 34 types of tiles
    # Create wall pool template
    wall_pool_template = np.zeros(136, dtype=np.int8) # Max 136
    wall_len = 0
    for t_id in range(34):
        rem = 4 - visible_counts_34[t_id]
        if rem > 0:
            for _ in range(rem):
                wall_pool_template[wall_len] = t_id
                wall_len += 1
                
    # Reuse arrays for performance
    # Opponent hands: 3 opponents * 34 tiles
    # But for logic we often just need counts.
    # We can use (3, 34) array.
    
    for sim_idx in range(sim_limit):
        # 1. Shuffle Wall
        current_wall = wall_pool_template[:wall_len].copy()
        np.random.shuffle(current_wall)
        
        # Wall pointer (stack index)
        wall_ptr = 0
        
        # 2. Distribute Hands
        # 3 Opponents, 13 tiles each = 39 tiles
        if wall_ptr + 39 > wall_len:
            continue # Should not happen unless extremely low tiles
        
        opp_hands = np.zeros((3, 34), dtype=np.int8)
        
        # Distribute 13 tiles to each opp
        for i in range(3):
            for _ in range(13):
                t = current_wall[wall_ptr]
                wall_ptr += 1
                opp_hands[i, t] += 1
                
        # 3. Simulation per Candidate
        for c_i in range(num_cands):
            c_discard = cand_indices[c_i]
            # Verify validity: Do we have this tile?
            # Creating my hand copy
            temp_my_hand = my_hand_34.copy()
            if temp_my_hand[c_discard] <= 0:
                # Invalid candidate (e.g. tsumogiri but tile not present? though logic usually handles)
                # Just skip or penalize?
                # Assuming valid input.
                pass
            
            temp_my_hand[c_discard] -= 1
            is_riichi = cand_is_riichi[c_i]
            
            # Copy simulation state
            sim_opp_hands = opp_hands.copy()
            sim_wall_ptr = wall_ptr
            
            # Game Loop
            turn = 0
            is_win = False
            final_score = 0
            
            # Opponent reach status
            opp_reach = np.zeros(3, dtype=np.bool_)
            
            # 17 turns limit or wall end
            while turn < 17 and sim_wall_ptr < wall_len:
                turn += 1
                
                # --- Opponents ---
                opp_dealt_in = False
                
                for i in range(3):
                    if sim_wall_ptr >= wall_len: break
                    draw_opp = current_wall[sim_wall_ptr]
                    sim_wall_ptr += 1
                    
                    sim_opp_hands[i, draw_opp] += 1
                    
                    # Win Check (Sloppy check: assume full hand)
                    if is_agari_jit(sim_opp_hands[i]):
                        # Opponent Tsumo
                        # End sim, we lost (0 points or negative?)
                        # Typically we only care if we win or deal in.
                        # If opp tsumo, result is 0 for us.
                        sim_wall_ptr = wall_len + 999 # Force break
                        break
                        
                    discard_opp = -1
                    if opp_reach[i]:
                        discard_opp = draw_opp # Tsumogiri
                    else:
                        # Strict Opponent Logic
                        # 1. Tenpai Check (Restored Accuracy)
                        # Only check if turn > 3 to save time (early turns rarely tenpai)
                        can_reach = False
                        if turn > 3 and not opp_reach[i]:
                             # Check Shanten
                             s = calculate_shanten_jit(sim_opp_hands[i])
                             if s <= 0: # Tenpai or Agari
                                 can_reach = True
                        
                        if can_reach:
                             # Decide Riichi
                             if np.random.random() < 0.6: # 60% Riichi if Tenpai
                                 opp_reach[i] = True
                        
                        # 2. Discard Selection
                        if opp_reach[i]:
                            discard_opp = draw_opp # Tsumogiri
                        else:
                            # Improved Discard Logic:
                            # 20% chance to swap tile (discard random from hand) to simulate hand progression
                            # 80% Tsumogiri (simulate maintaining improved hand)
                            # This is a heuristic simulation of "Drawing Useless Tile" vs "Improving"
                            if np.random.random() < 0.2:
                                # Pick random from hand
                                avail = []
                                for t in range(34):
                                    if sim_opp_hands[i, t] > 0: avail.append(t)
                                if len(avail) > 0:
                                     # Fast random pick
                                     # idx = np.random.randint(0, len(avail)) # Numba randint is inclusive? check docs. 
                                     # standard random.randint(a,b) is inclusive. numpy.random.randint(low, high) is exclusive.
                                     # We used np.random before.
                                     r_idx = np.random.randint(0, len(avail))
                                     discard_opp = avail[r_idx]
                                else:
                                     discard_opp = draw_opp
                            else:
                                discard_opp = draw_opp

                    
                    # Ron Check (Did we deal in? No, Opponent discarded)
                    # Did we win on Opponent discard? (Ron)
                    # Use discard_opp.
                    temp_my_hand[discard_opp] += 1
                    if is_agari_jit(temp_my_hand):
                         # Ron!
                         is_win = True
                         # Score calc
                         d_cnt = 0
                         for d in doras_ids:
                             if d < 34: d_cnt += temp_my_hand[d]
                         
                         base = 1500 + (d_cnt * 1000)
                         if is_riichi: base += 1000
                         final_score = base
                         
                         temp_my_hand[discard_opp] -= 1 # Restore
                         sim_wall_ptr = wall_len + 999
                         break
                    temp_my_hand[discard_opp] -= 1
                    
                if sim_wall_ptr >= wall_len: break
                    
                # --- Self ---
                draw = current_wall[sim_wall_ptr]
                sim_wall_ptr += 1
                
                temp_my_hand[draw] += 1
                
                # Tsumo Check
                if is_agari_jit(temp_my_hand):
                    is_win = True
                    d_cnt = 0
                    for d in doras_ids:
                         if d < 34: d_cnt += temp_my_hand[d]
                    base = 1500 + (d_cnt * 1000)
                    if is_riichi: base += 1000
                    final_score = base # Tsumo usually slightly higher but simplify
                    break
                    
                # Discard Logic
                # If Riichi, must discard draw
                if is_riichi:
                    chosen_discard = draw
                else:
                    # Logic: Maintain Shanten or Improve
                    # Current Shanten
                    cur_shanten = calculate_shanten_jit(temp_my_hand)
                    if cur_shanten == -1: # Already agari? (Handled above)
                         chosen_discard = draw 
                    else:
                        # Try discarding draw first
                        temp_my_hand[draw] -= 1
                        if calculate_shanten_jit(temp_my_hand) <= cur_shanten:
                            chosen_discard = draw
                            temp_my_hand[draw] += 1
                        else:
                            temp_my_hand[draw] += 1
                            # Try random 3
                            possible = []
                            for t in range(34):
                                if temp_my_hand[t] > 0: possible.append(t)
                            
                            # Random shuffle manually or pick
                            # np.random.shuffle is only 1D array.
                            # Just pick 3 random
                            chosen_discard = draw # Default
                            
                            idxs = np.random.choice(len(possible), min(3, len(possible)), replace=False)
                            for p_idx in idxs:
                                t = possible[p_idx]
                                temp_my_hand[t] -= 1
                                if calculate_shanten_jit(temp_my_hand) <= cur_shanten:
                                    chosen_discard = t
                                    temp_my_hand[t] += 1
                                    break
                                temp_my_hand[t] += 1
                                
                temp_my_hand[chosen_discard] -= 1
                
                # Did we deal in?
                for i in range(3):
                    # Check if opp wins on chosen_discard
                    sim_opp_hands[i, chosen_discard] += 1
                    if is_agari_jit(sim_opp_hands[i]):
                        if opp_reach[i]:
                            # Dealt into Riichi
                            final_score = -5200 # Average deal-in cost
                            sim_wall_ptr = wall_len + 999
                            break
                        else:
                            # Dama deal-in (Lower prob)
                            if np.random.random() < 0.2: # 20% Dama rate
                                final_score = -5200
                                sim_wall_ptr = wall_len + 999
                                break
                    sim_opp_hands[i, chosen_discard] -= 1
                
                if final_score < 0: break
                
            # End of Game Loop
            if final_score != 0:
                res_scores[c_i] += final_score
            if is_win:
                res_wins[c_i] += 1
                if final_score > 0:
                     # Turn bonus
                     res_scores[c_i] += (18 - turn) * 100
                     
    return res_wins, res_scores


# ----------------------------------------------------------------------
# Helper functions (Shanten / Agari) copied from previous step
# ----------------------------------------------------------------------

@njit(nogil=True, cache=False)
def get_waits_count_jit(hand_34_arr, visible_34_arr):
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
def is_agari_jit(hand_34):
    # Kokushi
    yao_indices = np.array([0,8,9,17,18,26,27,28,29,30,31,32,33], dtype=np.int8)
    has_pair = False
    yao_count = 0
    for i in range(13):
        idx = yao_indices[i]
        if hand_34[idx] == 0: break 
        if hand_34[idx] >= 2: has_pair = True
        yao_count += 1
    if yao_count == 13 and has_pair: return True

    # Chiitoi
    pair_count = 0
    for i in range(34):
        if hand_34[i] >= 2: pair_count += 1
    if pair_count == 7: return True

    # Normal
    for head_idx in range(34):
        if hand_34[head_idx] >= 2:
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

@njit(nogil=True, cache=False)
def calculate_shanten_jit(hand_34):
    # Kokushi
    yao_indices = np.array([0,8,9,17,18,26,27,28,29,30,31,32,33], dtype=np.int8)
    unique_yao = 0
    has_pair_k = False
    for i in range(13):
        idx = yao_indices[i]
        if hand_34[idx] > 0: unique_yao += 1
        if hand_34[idx] >= 2: has_pair_k = True
    shanten_kokushi = 13 - unique_yao - (1 if has_pair_k else 0)
    
    # Chiitoi
    pairs = 0
    unique = 0
    for i in range(34):
        if hand_34[i] >= 2: pairs += 1
        if hand_34[i] >= 1: unique += 1
    shanten_chiitoi = 6 - pairs
    if unique < 7: shanten_chiitoi += (7 - unique)
    
    # Normal (Split-Suit)
    pat_buf_m = np.zeros((64, 3), dtype=np.int8)
    pat_buf_p = np.zeros((64, 3), dtype=np.int8)
    pat_buf_s = np.zeros((64, 3), dtype=np.int8)
    pat_buf_z = np.zeros((64, 3), dtype=np.int8)
    
    cnt_m = _get_suit_patterns_jit(hand_34, 0, 9, pat_buf_m)
    cnt_p = _get_suit_patterns_jit(hand_34, 9, 18, pat_buf_p)
    cnt_s = _get_suit_patterns_jit(hand_34, 18, 27, pat_buf_s)
    cnt_z = _get_suit_patterns_honor_jit(hand_34, 27, 34, pat_buf_z)
    
    min_shanten_normal = 8
    
    for i_m in range(cnt_m):
        m1, t1, h1 = pat_buf_m[i_m]
        for i_p in range(cnt_p):
            m2, t2, h2 = pat_buf_p[i_p]
            for i_s in range(cnt_s):
                m3, t3, h3 = pat_buf_s[i_s]
                for i_z in range(cnt_z):
                    m4, t4, h4 = pat_buf_z[i_z]
                    
                    total_m = m1 + m2 + m3 + m4
                    total_t = t1 + t2 + t3 + t4
                    total_h = h1 + h2 + h3 + h4
                    
                    has_head = 0
                    if total_h > 0:
                        has_head = 1
                        total_t += (total_h - 1)
                        
                    avail_shuntsu = 4 - total_m
                    if avail_shuntsu < 0: avail_shuntsu = 0
                    
                    used_tatsu = total_t
                    if used_tatsu > avail_shuntsu: used_tatsu = avail_shuntsu
                        
                    temps = 8 - (2 * total_m) - used_tatsu - has_head
                    if temps < min_shanten_normal: min_shanten_normal = temps
                        
    return min(shanten_kokushi, shanten_chiitoi, min_shanten_normal)

@njit(nogil=True, cache=False)
def _get_suit_patterns_jit(hand, start, end, result_buf):
    counts = hand[start:end].copy()
    counter = np.zeros(1, dtype=np.int32)
    _recurse_pattern(counts, 0, 0, 0, 0, result_buf, counter)
    return counter[0]

@njit(nogil=True, cache=False)
def _recurse_pattern(counts, idx, m, t, h, res_buf, counter):
    if idx >= 9:
        c = counter[0]
        if c < 64:
            res_buf[c, 0] = m
            res_buf[c, 1] = t
            res_buf[c, 2] = h
            counter[0] += 1
        return
    if counts[idx] == 0:
        _recurse_pattern(counts, idx + 1, m, t, h, res_buf, counter)
        return
    if counts[idx] >= 3:
        counts[idx] -= 3
        _recurse_pattern(counts, idx, m + 1, t, h, res_buf, counter)
        counts[idx] += 3
    if idx < 7:
        if counts[idx] > 0 and counts[idx+1] > 0 and counts[idx+2] > 0:
            counts[idx] -= 1; counts[idx+1] -= 1; counts[idx+2] -= 1
            _recurse_pattern(counts, idx, m + 1, t, h, res_buf, counter)
            counts[idx] += 1; counts[idx+1] += 1; counts[idx+2] += 1
    if counts[idx] >= 2:
        counts[idx] -= 2
        _recurse_pattern(counts, idx, m, t, h + 1, res_buf, counter)
        counts[idx] += 2
    if idx < 8:
        if counts[idx] > 0 and counts[idx+1] > 0:
             counts[idx] -= 1; counts[idx+1] -= 1
             _recurse_pattern(counts, idx, m, t + 1, h, res_buf, counter)
             counts[idx] += 1; counts[idx+1] += 1
    if idx < 7:
        if counts[idx] > 0 and counts[idx+2] > 0:
             counts[idx] -= 1; counts[idx+2] -= 1
             _recurse_pattern(counts, idx, m, t + 1, h, res_buf, counter)
             counts[idx] += 1; counts[idx+2] += 1
    _recurse_pattern(counts, idx + 1, m, t, h, res_buf, counter)

@njit(nogil=True, cache=False)
def _get_suit_patterns_honor_jit(hand, start, end, result_buf):
    counts = hand[start:end].copy()
    counter = np.zeros(1, dtype=np.int32)
    _recurse_pattern_honor(counts, 0, 0, 0, 0, result_buf, counter)
    return counter[0]

@njit(nogil=True, cache=False)
def _recurse_pattern_honor(counts, idx, m, t, h, res_buf, counter):
    if idx >= 7:
        c = counter[0]
        if c < 64:
            res_buf[c, 0] = m
            res_buf[c, 1] = t
            res_buf[c, 2] = h
            counter[0] += 1
        return
    if counts[idx] == 0:
        _recurse_pattern_honor(counts, idx + 1, m, t, h, res_buf, counter)
        return
    if counts[idx] >= 3:
        counts[idx] -= 3
        _recurse_pattern_honor(counts, idx, m + 1, t, h, res_buf, counter)
        counts[idx] += 3
    if counts[idx] >= 2:
        counts[idx] -= 2
        _recurse_pattern_honor(counts, idx, m, t, h + 1, res_buf, counter)
        counts[idx] += 2
    _recurse_pattern_honor(counts, idx + 1, m, t, h, res_buf, counter)
