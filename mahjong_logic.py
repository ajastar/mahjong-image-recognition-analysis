import time
import math
import random
import collections
import concurrent.futures
import functools
import itertools

TILE_MAP = {
    "1m":0, "2m":1, "3m":2, "4m":3, "5m":4, "6m":5, "7m":6, "8m":7, "9m":8,
    "1p":9, "2p":10, "3p":11, "4p":12, "5p":13, "6p":14, "7p":15, "8p":16, "9p":17,
    "1s":18, "2s":19, "3s":20, "4s":21, "5s":22, "6s":23, "7s":24, "8s":25, "9s":26,
    "1z":27, "2z":28, "3z":29, "4z":30, "5z":31, "6z":32, "7z":33
}
Tile = int # Type Alias
INV_TILE_MAP = {v: k for k, v in TILE_MAP.items()}
TILE_MAP["0m"] = 4; TILE_MAP["0p"] = 13; TILE_MAP["0s"] = 22
NAKI_LABELS = ["スルー", "チー", "ポン", "カン", "リーチ"]

# Enable Numba if available
try:
    import numpy as np
    from numba_utils import calculate_shanten_jit, is_agari_jit, get_waits_count_jit
    USE_NUMBA = True
    print("[INFO] Numba Acceleration Enabled")
except ImportError:
    USE_NUMBA = False
    print("[WARN] Numba not found, running in Python mode (Slow)")

def get_suji(tile):
    suji = []
    if tile >= 27: return suji
    num = tile % 9
    if num >= 3: suji.append(tile - 3)
    if num <= 5: suji.append(tile + 3)
    return suji


class ShantenUtils:
    @staticmethod
    def calculate_shanten(hand_34):
        """
        手牌(34枚配列)のシャンテン数を計算する。
        戻り値: min(通常, 七対子, 国士無双)
        """
        if USE_NUMBA:
            return calculate_shanten_jit(np.array(hand_34, dtype=np.int8))

        # 1. 国士無双 (Kokushi)
        shanten_kokushi = ShantenUtils._shanten_kokushi(hand_34)
        
        # 2. 七対子 (Chiitoi)
        shanten_chiitoi = ShantenUtils._shanten_chiitoi(hand_34)
        
        # 3. 一般手 (Normal) - Optimized with Split-Suit Memoization
        shanten_normal = ShantenUtils._shanten_normal(hand_34)
        
        return min(shanten_kokushi, shanten_chiitoi, shanten_normal)

    @staticmethod
    def _shanten_kokushi(hand_34):
        yao_indices = [0,8,9,17,18,26,27,28,29,30,31,32,33]
        unique_yao = 0
        has_pair = False
        for idx in yao_indices:
            if hand_34[idx] > 0:
                unique_yao += 1
            if hand_34[idx] >= 2:
                has_pair = True
        return 13 - unique_yao - (1 if has_pair else 0)

    @staticmethod
    def _shanten_chiitoi(hand_34):
        pairs = 0
        unique_tiles = 0
        for c in hand_34:
            if c >= 2: pairs += 1
            if c >= 1: unique_tiles += 1
        shanten = 6 - pairs
        if unique_tiles < 7: # 7種必要
            shanten += (7 - unique_tiles)
        return shanten

    @staticmethod
    def _shanten_normal(hand_34):
        # Split-Suit Memoization Approach
        # 各色の(mentsu, tatsu, pair_candidates)の組み合わせを取得し、統合する
        
        # 1. 各色のパターンを取得 (Tuple化してCacheを利用 -> Bytes化で高速化)
        # 0-4の値を想定。255以下なのでbytesでOK
        patterns_man = ShantenUtils._get_suit_patterns(bytes(hand_34[0:9]))
        patterns_pin = ShantenUtils._get_suit_patterns(bytes(hand_34[9:18]))
        patterns_sou = ShantenUtils._get_suit_patterns(bytes(hand_34[18:27]))
        patterns_hon = ShantenUtils._get_suit_patterns_honor(bytes(hand_34[27:34]))
        
        # 2. 組み合わせ探索
        # シャンテン数 = 8 - 2*M - T - H
        # 制約: M + T + H <= 5 (Total Blocks)
        # ただしH(Head)は最大1。H=0なら5ブロック(M+T<=5)も可だが、H=1ならM+T<=4
        
        min_shanten = 8
        
        # 4つのリストの直積を回すが、要素数が少ないので高速
        for m_m, t_m, h_m in patterns_man:
            for m_p, t_p, h_p in patterns_pin:
                for m_s, t_s, h_s in patterns_sou:
                    for m_z, t_z, h_z in patterns_hon:
                        
                        total_m = m_m + m_p + m_s + m_z
                        total_t = t_m + t_p + t_s + t_z
                        total_h = h_m + h_p + h_s + h_z
                        
                        # Head Handling
                        has_head = 0
                        if total_h > 0:
                            has_head = 1
                            # 複数の候補がHeadになれるが、採用するのは1つだけ
                            # Headとして使わない余剰な対子はTatsu(搭子)として扱える
                            total_t += (total_h - 1)
                        
                        # Block Constraint (4面子1雀頭 = 5ブロック)
                        # 公式: 8 - 2M - T - H
                        # T(Tatsu)は、残り枠 (4 - M) までしかカウントできない
                        
                        avail_shuntsu = 4 - total_m
                        if avail_shuntsu < 0: avail_shuntsu = 0
                        
                        used_tatsu = total_t
                        if used_tatsu > avail_shuntsu:
                            used_tatsu = avail_shuntsu
                                
                        shanten = 8 - (2 * total_m) - used_tatsu - has_head
                        
                        if shanten < min_shanten:
                            min_shanten = shanten
                            
        return min_shanten

    @staticmethod
    @functools.lru_cache(maxsize=8192)
    def _get_suit_patterns(counts):
        """
        1色分の手牌(tuple)を受け取り、可能な(Mentsu, Tatsu, HeadCount)の組み合わせリストを返す
        """
        results = set() # 重複排除
        
        # 再帰関数内でリストを操作するが、結果は不変
        def _recurse(idx, c_counts, m, t, h):
            if idx >= 9:
                results.add((m, t, h))
                return

            # Skip empty
            if c_counts[idx] == 0:
                _recurse(idx + 1, c_counts, m, t, h)
                return

            # Try Mentsu (Koutsu)
            if c_counts[idx] >= 3:
                c_counts[idx] -= 3
                _recurse(idx, c_counts, m + 1, t, h)
                c_counts[idx] += 3
            
            # Try Mentsu (Shuntsu)
            if idx < 7 and c_counts[idx] > 0 and c_counts[idx+1] > 0 and c_counts[idx+2] > 0:
                c_counts[idx] -= 1; c_counts[idx+1] -= 1; c_counts[idx+2] -= 1
                _recurse(idx, c_counts, m + 1, t, h)
                c_counts[idx] += 1; c_counts[idx+1] += 1; c_counts[idx+2] += 1
                
            # Try Head (Pair)
            if c_counts[idx] >= 2:
                c_counts[idx] -= 2
                _recurse(idx, c_counts, m, t, h + 1)
                c_counts[idx] += 2
                
            # Try Tatsu (Ryanmen/Kanchan/Penchan)
            if idx < 8 and c_counts[idx] > 0 and c_counts[idx+1] > 0: # Ryanmen/Penchan
                 c_counts[idx] -= 1; c_counts[idx+1] -= 1
                 _recurse(idx, c_counts, m, t + 1, h)
                 c_counts[idx] += 1; c_counts[idx+1] += 1
            
            if idx < 7 and c_counts[idx] > 0 and c_counts[idx+2] > 0: # Kanchan
                 c_counts[idx] -= 1; c_counts[idx+2] -= 1
                 _recurse(idx, c_counts, m, t + 1, h)
                 c_counts[idx] += 1; c_counts[idx+2] += 1
            
            # Skip (Use as isolated)
            # ここでは「何も作らない」選択肢も探索しないと全パターン網羅できない
            # ただし、牌を減らして次に進む
            # c_counts[idx]を減らすのではなく、idxを進める
            # しかし、c_counts[idx]が残ってると次にShuntsu等で使われる可能性がないならIsolation確定
            # Shuntsuの始点または構成要素になりうるか？
            # 簡略化: idx番目の牌を「使わない」場合、Shuntsuの構成員としても使わないことはできない(Shuntsu logic handles usage)
            # ここでは「これ以上このidxを始点とするグループを作らない」として進む
            _recurse(idx + 1, c_counts, m, t, h)

        # Cast to list for mutation
        _recurse(0, list(counts), 0, 0, 0)
        return list(results)

    @staticmethod
    @functools.lru_cache(maxsize=8192)
    def _get_suit_patterns_honor(counts):
        """
        字牌用のパターン抽出(順子がないので簡単)
        """
        results = set()
        
        def _recurse(idx, c_counts, m, t, h):
            if idx >= 7:
                results.add((m, t, h))
                return
            
            if c_counts[idx] == 0:
                _recurse(idx + 1, c_counts, m, t, h)
                return
                
            # Koutsu
            if c_counts[idx] >= 3:
                c_counts[idx] -= 3
                _recurse(idx, c_counts, m + 1, t, h)
                c_counts[idx] += 3
            
            # Pair
            if c_counts[idx] >= 2:
                c_counts[idx] -= 2
                _recurse(idx, c_counts, m, t, h + 1)
                c_counts[idx] += 2
                
            # No Tatsu for Honors (Simplified) - Honor Pair is Tatsu? 
            # No, standard definition: Tatsu is 2 tiles waiting for 1. 2 Honors is a Pair (Head) or Pung-wait.
            # In shanten formula, Pair can be Tatsu. Here we count as Head candidates.
            # If we don't use it as Head, it becomes Tatsu in combiner.
            
            # Skip
            _recurse(idx + 1, c_counts, m, t, h)
            
        _recurse(0, list(counts), 0, 0, 0)
        return list(results)

# -----------------------------------------------------------------------------
# 📈 Ukeire & Value Engine (Advanced SOTA Version)
# -----------------------------------------------------------------------------
class UkeireUtils:
    @staticmethod
    def get_ukeire_count(hand_34, current_shanten, visible_tiles_34):
        """
        現在のシャンテン数を下げる「有効牌」の種類と合計枚数を返す (完全版)
        visible_tiles_34: 全体で見えている牌の個数カウント (0-4)
        """
        effective_tiles = []
        total_count = 0
        
        # 34種すべてについて「1枚足してシャンテン数が下がるか」確認
        for t_id in range(34):
            if hand_34[t_id] >= 4: continue
            
            # 手持ち + 場に見えている数が4枚なら、もう引けない (Ghost Ukeire排除)
            if (hand_34[t_id] + visible_tiles_34[t_id]) >= 4:
                continue

            hand_34[t_id] += 1
            new_shanten = ShantenUtils.calculate_shanten(hand_34)
            hand_34[t_id] -= 1
            
            if new_shanten < current_shanten:
                # 有効牌発見
                # 残り枚数 = 4 - 手持ち - 場に見えている分
                left_num = 4 - hand_34[t_id] - visible_tiles_34[t_id]
                if left_num > 0:
                    effective_tiles.append(t_id)
                    total_count += left_num
                    
        return total_count, effective_tiles

class SafetyUtils:
    @staticmethod
    def _estimate_tenpai_prob(turn, melds_count, is_riichi):
        """
        Estimate the probability that an opponent is Tenpai.
        World-Class Heuristic:
        - Riichi: 100%
        - Turn < 6: Very Low (unless 2+ melds)
        - Turn 6-12: Increasing
        - Turn 13+: High
        - Melds increase probability significantly.
        """
        if is_riichi:
            return 1.0
        
        # Base probability by turn (sigmoid-ish)
        # Turn 0-5: 5%
        # Turn 6-9: 10-25%
        # Turn 10-13: 30-50%
        # Turn 14+: 60-80%
        if turn < 5: base = 0.05
        elif turn < 9: base = 0.15 + (turn-5)*0.03
        elif turn < 13: base = 0.30 + (turn-9)*0.05
        else: base = 0.60 + (turn-13)*0.05
        
        # Meld Boost
        # 1 Meld: +10%, 2 Melds: +30%, 3 Melds: +60%
        if melds_count == 1: base += 0.10
        elif melds_count == 2: base += 0.30
        elif melds_count >= 3: base += 0.60
            
        return min(base, 0.95) # Cap at 95% for non-riichi

    @staticmethod
    def _detect_honitsu(river):
        """
        Detect Honitsu using Statistical Hypothesis Testing (Z-test).
        Null Hypothesis: Discards are uniformly distributed (p=1/3).
        If Z-score of a suit is significantly NEGATIVE (fewer discards than random), 
        we reject Null Hypothesis and assume Honitsu.
        """
        if len(river) < 6: return None
        
        counts = [0, 0, 0] # Man, Pin, Sou
        total_terminals = 0
        
        for t in river:
            if t < 27:
                counts[t // 9] += 1
                total_terminals += 1
                
        if total_terminals < 5: return None
        
        # Statistical Test
        # Expected Count = N * 1/3
        # Variance = N * p * (1-p) = N * 1/3 * 2/3 = N * 2/9
        # StdDev = sqrt(N * 2/9)
        
        import math
        expected = total_terminals / 3.0
        std_dev = math.sqrt(total_terminals * 2.0 / 9.0)
        
        for s in range(3):
            # Z-score = (Observed - Expected) / StdDev
            z_score = (counts[s] - expected) / std_dev
            
            # If Z < -1.28 (p < 0.10 one-tailed), significant shortage.
            # Relaxed to 10% significance as requested.
            if z_score < -1.28: 
                return s
        return None

    @staticmethod
    def _detect_tanyao_threat(river, melds):
        """
        Detect if player is pushing Tanyao.
        Logic: Discards are overwhelmingly Honors/Terminals.
        And opened Meld is Tanyao (Simples).
        """
        if not river: return False
        
        # Check Melds first
        for m in melds:
            for t in m:
                if t >= 27: return False # Honor in meld -> Not Tanyao
                if (t%9) in [0, 8]: return False # Terminal in meld -> Not Tanyao
        
        # Check River
        yaochu_count = 0
        total = len(river)
        if total < 5: return False
        
        for t in river:
            if t >= 27 or (t < 27 and (t%9 in [0, 8])):
                yaochu_count += 1
        
        if yaochu_count / total > 0.6: # >60% discards are Yaochu
            return True
        return False

    @staticmethod
    def _detect_kokushi_threat(river):
        """
        Detect Kokushi Musou.
        Logic: Anomalous discarding of Middle Tiles (2-8) early.
        Zero or near-zero Yaochu discards.
        """
        if len(river) < 6: return False
        
        yaochu_count = 0
        for t in river:
            if t >= 27 or (t < 27 and (t%9 in [0, 8])):
                yaochu_count += 1
        
        # Kokushi player hoards Yaochu. So River has very FEW Yaochu.
        if yaochu_count == 0: return True
        return False

    @staticmethod
    def get_danger_score(tile_idx, river_hist, riichi_status, visible_tiles_34, turn=10, melds_arr=None, doras=None, riichi_indices=None):
        """
        World-Class Danger Assessment (Genbutsu, Suji, Kabe, TenpaiProb, Yomi, Dora, OneChance)
        Params:
          turn: Current turn count (approx)
          melds_arr: List of melds for each player (to count melds)
          doras: List of actual Dora tile IDs
          riichi_indices: List of [index, index, index, index] indicating WHEN player declared Riichi.
                          -1 if not Riichi. Used for Post-Riichi Safety (Awase-uchi).
        """
        max_danger = 0
        
        if melds_arr is None: melds_arr = [[] for _ in range(4)]
        if doras is None: doras = []
        if riichi_indices is None: riichi_indices = [-1]*4
        
        for r_player in range(4):
            # Evaluate danger against EACH opponent
            if r_player == 3: continue 
            
            is_riichi_p = riichi_status[r_player]
            
            # 0. Tenpai Probability Scaling
            melds_c = len(melds_arr[r_player])
            tenpai_prob = SafetyUtils._estimate_tenpai_prob(turn, melds_c, is_riichi_p)
            
            # If probability is negligible, ignore this player
            if tenpai_prob < 0.10: continue
            
            river = river_hist[r_player] # Get player's river
            
            # 1. Genbutsu (Absolute Safety)
            # A. Own Discards
            if tile_idx in river:
                continue # Safety 100% -> Danger 0
            
            # B. Post-Riichi Discards (Awase-uchi by others)
            # If r_player declared Riichi at index R, any tile discarded by ANYONE at index > R is safe against r_player.
            r_idx = riichi_indices[r_player]
            if is_riichi_p and r_idx != -1:
                is_safe_awase = False
                for p_check in range(4):
                    # Check recent discards of everyone
                    # If tile_idx was discarded by p_check AFTER r_idx
                    target_river = river_hist[p_check]
                    # We assume lists are chronological.
                    # Safety condition: Exists at index k where k > r_idx
                    if len(target_river) > r_idx + 1:
                        # Check passed tiles AFTER riichi
                        # Optimization: Iterate backwards
                        for k in range(len(target_river)-1, r_idx, -1):
                            if target_river[k] == tile_idx:
                                is_safe_awase = True; break
                    if is_safe_awase: break
                
                if is_safe_awase:
                    continue # Safety 100% (Passed against Riichi)

            base_danger = 100.0 # Default High
            
                
            
            base_danger = 100.0 # Default High
            
            # --- YOMI: Advanced Role Reading ---
            
            # A. Kokushi Check (Dangerous Yaochu)
            if SafetyUtils._detect_kokushi_threat(river):
                if tile_idx >= 27 or (tile_idx < 27 and (tile_idx%9 in [0, 8])):
                    base_danger *= 2.0 # Critical Danger
                else:
                    base_danger *= 0.1 # Middle tiles are safe
            
            # B. Tanyao Check (Safe Yaochu)
            elif SafetyUtils._detect_tanyao_threat(river, melds_arr[r_player]):
                if tile_idx >= 27 or (tile_idx < 27 and (tile_idx%9 in [0, 8])):
                    base_danger *= 0.2 # Yaochu Safe
                else:
                    base_danger *= 1.2 # Middle Dangerous
            
            # C. Honitsu Reading (Flush)
            target_suit = SafetyUtils._detect_honitsu(river)
            
            if target_suit is not None:
                if tile_idx < 27: # Only applies to suit tiles
                    my_suit = tile_idx // 9
                    if my_suit == target_suit:
                        base_danger *= 1.5 # Danger Boost for Target Suit
                    else:
                        base_danger *= 0.5 # Safety Boost for Non-Target Suit
                        if not is_riichi_p: # If not Riichi, non-target suit is often safe against Honitsu
                             base_danger *= 0.5
            
            # 2. Honors Logic
            if tile_idx >= 27:
                visible = visible_tiles_34[tile_idx]
                if visible >= 3: base_danger = 0 # Safe (Hell wait only)
                elif visible == 2: base_danger = 10 # Low
                elif visible == 1: base_danger = 40 # Medium
                elif visible == 0: base_danger = 80 # Dangerous (Yaku-hai?)
                # Guest winds are safer than Dragons/Round/Self
                pass
                
            # 3. Suji Logic (Ryanmen negation)
            elif tile_idx < 27:
                is_suji = False
                color = tile_idx // 9
                num = tile_idx % 9 + 1
                river_set = set(river) # Use local river vars
                
                # Check Standard Suji (4->1,7 etc)
                suji_safe = False
                if num in [1, 9]:
                    if (num==1 and (tile_idx+3) in river_set) or (num==9 and (tile_idx-3) in river_set): suji_safe = True
                elif num in [2, 8]:
                    if (num==2 and (tile_idx+3) in river_set) or (num==8 and (tile_idx-3) in river_set): suji_safe = True
                elif num in [3, 7]:
                     if (num==3 and (tile_idx+3) in river_set) or (num==7 and (tile_idx-3) in river_set): suji_safe = True
                elif num in [4, 5, 6]:
                    # Nakasuji requires BOTH sides
                    # e.g. 4 -> 1 and 7
                    s1 = (tile_idx - 3) in river_set
                    s2 = (tile_idx + 3) in river_set
                    if s1 and s2: suji_safe = True
                    # Katasuji (Half Safe)
                    elif s1 or s2: base_danger *= 0.6 
                    
                if suji_safe:
                    base_danger = 15 
                
                # Ryanmen Tatsu Otoshi Check
                # If opponent dropped 2m then 3m, then 1m, 4m are safe.
                # Check for adjacent discards in river
                # Naive scan of river tuples
                for i in range(len(river)-1):
                    t1, t2 = river[i], river[i+1]
                    if t1 >= 27 or t2 >= 27: continue
                    if t1//9 != t2//9: continue
                    dist = abs(t1 - t2)
                    if dist == 1:
                         # Found sequence drop (e.g. 2,3)
                         # Safe Suji: (min-1, max+1)
                         mi, ma = min(t1,t2), max(t1,t2)
                         # e.g. 2,3 (idx 1,2) -> Safe 1 (0) and 4 (3)
                         safe_a = mi - 1
                         safe_b = ma + 1
                         if tile_idx == safe_a or tile_idx == safe_b:
                             base_danger *= 0.4 # Significant reduction (Sequence broken)
                    
                # 4. Kabe (Wall) Logic (No-Chance)
                # If 4 is Dead (4 visible), then 2,3 are safe from (23, 34) wait?
                # Wait: 23 waits on 1,4. If 4 is dead, 23 must wait on 1. (Penchan)
                # Logic:
                # To wait on 'Tile', opponent needs adjacent tiles.
                # Ryanmen wait on T requires (T-1, T-2) OR (T+1, T+2).
                # If (T-1) is dead (4 copies visible), then (T-1, T-2) ryanmen is impossible.
                # If (T+1) is dead, then (T+1, T+2) ryanmen is impossible.
                
                is_kabe_safe = False
                # Check Lower Kabe (e.g. tile 3, need 4 to be dead for 3-4 wait? No.)
                # Wait T=3.
                # Pattern: 1-2 (wait 3), 2-4 (wait 3), 4-5 (wait 3,6), 3-3 (wait 3)
                # If 4 is dead?
                # 2-4 impossible. 4-5 impossible.
                # So only 1-2 (Penchan) or 3-3 (Tanki/Shanpon) left.
                # Ryanmen (4-5) is killed.
                
                # Upper Neighbor Kabe
                if num < 9:
                    cnt = visible_tiles_34[tile_idx+1]
                    if cnt >= 4: base_danger *= 0.5 # No-Chance (Strong Safe)
                    elif cnt == 3: base_danger *= 0.7 # One-Chance (Slightly Safe)

                # Lower Neighbor Kabe
                if num > 1:
                    cnt = visible_tiles_34[tile_idx-1]
                    if cnt >= 4: base_danger *= 0.5 # No-Chance
                    elif cnt == 3: base_danger *= 0.7 # One-Chance
            
            # 5. Dora Logic (Bonus Danger)
            if tile_idx in doras:
                base_danger *= 1.3 # Dora itself is hot
            
            # Dora Neighbors (Soba) check
            # If 5m is Dora, 4m/6m are dangerous.
            is_dora_neighbor = False
            for d in doras:
                if d < 27 and tile_idx < 27:
                    if d // 9 == tile_idx // 9: # Same suit
                        dist = abs((d%9) - (tile_idx%9))
                        if dist == 1: is_dora_neighbor = True
            
            if is_dora_neighbor:
                base_danger *= 1.15

            # Final calculation for this player
            # Apply Tenpai Probability
            player_danger = base_danger * tenpai_prob
            
            # Keep max danger (we are afraid of the most dangerous opponent)
            max_danger = max(max_danger, player_danger)
            
        # Convert continuous danger to score (0, 15, 50, etc for compatibility)
        # But returning float is fine for sorting logic.
        return max_danger

# -----------------------------------------------------------------------------
# 🗣️ Naki (Call) Logic
# -----------------------------------------------------------------------------
class NakiUtils:
    @staticmethod
    def check_calls(hand_34, target_tile, from_offset, doras, turn):
        """
        Check if we should call (Pon, Chi, Kan, Ron).
        from_offset: 1=Right(Shimo), 2=Top(Toimen), 3=Left(Kami)
        Returns: Best Action string or None.
        """
        actions = []
        
        # 0. Ron Check
        hand_copy = hand_34[:]
        hand_copy[target_tile] += 1
        if AgariUtils.is_agari(hand_copy):
            # Calculate value (simplified)
            return "RON! (Win)"
        
        # 1. Pon Check (Any opponent)
        if hand_34[target_tile] >= 2:
            # Pon is possible. Should we?
            # Criteria:
            # - Dora involved?
            # - Yaku likely? (Yakuhai, Tanyao, Toitoi)
            # - Speed boost?
            score = 0
            if target_tile >= 27:
                if target_tile in [31, 32, 33]: score += 100 # Yakuhai
            else:
                 # Tanyao heuristic
                 terminals = sum(hand_34[t] for t in [0,8,9,17,18,26] + list(range(27,34)))
                 if terminals == 0: score += 50
            
            if target_tile in doras: score += 150 
            if turn > 12: score += 50 # Late game speed
            
            # Simple strategic threshold
            if score > 0:
                actions.append(f"PON ({target_tile})")

        # 2. Chi Check (Only from Left / offset=3)
        if from_offset == 3 and target_tile < 27:
            # Check L-M-R
            # Possible patterns using target T:
            # (T-2, T-1), (T-1, T+1), (T+1, T+2)
            c = target_tile
            n = c % 9 + 1
            
            # Left Chi: T, T+1, T+2 (Need T+1, T+2)
            if n <= 7 and hand_34[c+1] > 0 and hand_34[c+2] > 0:
                actions.append(f"CHI (L)")
            
            # Middle Chi: T-1, T, T+1 (Need T-1, T+1)
            if n >= 2 and n <= 8 and hand_34[c-1] > 0 and hand_34[c+1] > 0:
                 actions.append(f"CHI (M)")
                 
            # Right Chi: T-2, T-1, T (Need T-2, T-1)
            if n >= 3 and hand_34[c-2] > 0 and hand_34[c-1] > 0:
                 actions.append(f"CHI (R)")

        # Select Best
        if not actions: return None
        if any("RON" in a for a in actions): return "RON! (Win)"
        if any("PON" in a for a in actions): return actions[0] # Priority to Pon
        return actions[0] 

class NakiPlanner:
    """
    Analyzes a 13-tile hand (after hypothetical discard) for Open-Hand Potential.
    Uses EXACT structural decomposition via StructureUtils for rigorous evaluation.
    """
    @staticmethod
    def analyze_naki_potential(hand_34, doras):
        """
        Returns a score (0.0 to 30.0) indicating Open-Hand Potential.
        Uses StructureUtils.get_all_partitions for EXACT structural analysis.
        Evaluates ALL Yaku on EACH partition and returns the MAXIMUM potential.
        """
        # Get all possible partitions (DFS exhaustive)
        partitions = StructureUtils.get_all_partitions(hand_34)
        
        if not partitions:
            return 0.0
        
        max_score = 0.0
        terminals = [0,8,9,17,18,26] + list(range(27,34))
        
        for part in partitions:
            part_score = 0.0
            
            mentsu = part['mentsu']
            tatsu = part['tatsu']
            pairs = part['pair']
            iso = part['iso']
            
            # --- 1. Block Count Bonus (Speed) ---
            # More complete blocks = faster to tenpai
            block_count = len(mentsu) + len(tatsu) + len(pairs)
            part_score += block_count * 1.0
            
            # --- 2. Yakuhai Check (Exact) ---
            # Dragons: 31=Haku, 32=Hatsu, 33=Chun
            # Winds: 27=East, 28=South, 29=West, 30=North
            for t in [31, 32, 33]: # Dragons (Confirmed Yakuhai)
                # In pairs (potential Pon)
                if t in pairs: part_score += 4.0
                # In Mentsu (already Koutsu? shouldn't be in 13-tile)
                for m in mentsu:
                    if m[0] == 'koutsu' and m[1] == t: part_score += 4.0
            
            for t in [27, 28, 29, 30]: # Winds (50% chance)
                if t in pairs: part_score += 2.0
            
            # --- 3. Tanyao Check (Exact) ---
            is_tanyao = True
            for m in mentsu:
                if m[0] == 'koutsu':
                    if m[1] in terminals: is_tanyao = False
                elif m[0] == 'shunta':
                    if m[1] % 9 == 0 or m[1] % 9 == 6: is_tanyao = False
            for t in tatsu:
                idx = t[1]
                if idx in terminals or (idx+1) in terminals or (idx+2 if t[0]=='kanchan' else idx+1) in terminals:
                    is_tanyao = False
            for p in pairs:
                if p in terminals: is_tanyao = False
            for i in iso:
                if i in terminals: is_tanyao = False
            
            if is_tanyao: part_score += 3.0
            
            # --- 4. Honitsu / Chinitsu Check (Exact) ---
            suit_counts = [0, 0, 0]
            honor_count = 0
            for m in mentsu:
                idx = m[1]
                if idx >= 27: honor_count += 3
                else: suit_counts[idx // 9] += 3
            for t in tatsu:
                idx = t[1]
                suit_counts[idx // 9] += 2
            for p in pairs:
                if p >= 27: honor_count += 2
                else: suit_counts[p // 9] += 2
            for i in iso:
                if i >= 27: honor_count += 1
                else: suit_counts[i // 9] += 1
            
            total_suit_tiles = sum(suit_counts)
            
            for s in range(3):
                # Chinitsu: All tiles in one suit (no honors)
                if suit_counts[s] == total_suit_tiles + honor_count and honor_count == 0:
                    part_score += 6.0
                # Honitsu: One suit + honors
                elif suit_counts[s] >= 8 and sum(1 for x in suit_counts if x > 0) == 1 and honor_count > 0:
                    part_score += 4.0
            
            # --- 5. Sanshoku Doujun Check (Exact) ---
            # Check if shuntsu in different suits share same number
            shunta_nums = {}
            for m in mentsu:
                if m[0] == 'shunta':
                    idx = m[1]
                    suit = idx // 9
                    num = idx % 9
                    if num not in shunta_nums:
                        shunta_nums[num] = set()
                    shunta_nums[num].add(suit)
            
            # Check tatsu for potential sanshoku
            for t in tatsu:
                if t[0] in ['ryanmen', 'kanchan']:
                    idx = t[1]
                    suit = idx // 9
                    num = idx % 9
                    if num not in shunta_nums:
                        shunta_nums[num] = set()
                    shunta_nums[num].add(suit)
            
            for num, suits in shunta_nums.items():
                if len(suits) == 3: part_score += 3.0 # Confirmed Sanshoku
                elif len(suits) == 2: part_score += 1.0 # Potential Sanshoku
            
            # --- 6. Ittsu (Straight) Check (Exact) ---
            for s in range(3):
                has_123 = any(m[0]=='shunta' and m[1]==s*9+0 for m in mentsu) or \
                          any(t[1]==s*9+0 or t[1]==s*9+1 for t in tatsu)
                has_456 = any(m[0]=='shunta' and m[1]==s*9+3 for m in mentsu) or \
                          any(t[1]==s*9+3 or t[1]==s*9+4 for t in tatsu)
                has_789 = any(m[0]=='shunta' and m[1]==s*9+6 for m in mentsu) or \
                          any(t[1]==s*9+6 or t[1]==s*9+7 for t in tatsu)
                
                if has_123 and has_456 and has_789:
                    part_score += 3.0
                elif (has_123 and has_456) or (has_456 and has_789) or (has_123 and has_789):
                    part_score += 1.0
            
            # --- 7. Toitoi Check (Exact) ---
            koutsu_count = sum(1 for m in mentsu if m[0] == 'koutsu')
            pair_count = len(pairs)
            if koutsu_count >= 2 and pair_count >= 2:
                part_score += 2.0
            
            # --- 8. Chanta / Junchan Check (Exact) ---
            all_terminal_blocks = True
            for m in mentsu:
                if m[0] == 'koutsu':
                    if m[1] not in terminals: all_terminal_blocks = False
                elif m[0] == 'shunta':
                    if m[1] % 9 not in [0, 6]: all_terminal_blocks = False
            for p in pairs:
                if p not in terminals: all_terminal_blocks = False
            
            if all_terminal_blocks and len(mentsu) >= 2:
                part_score += 2.0
            
            # --- 9. Dora Bonus (Exact Count) ---
            dora_count = 0
            for d in doras:
                if d < 34:
                    dora_count += hand_34[d]
            
            if dora_count >= 3: part_score += 4.0 + dora_count
            elif dora_count >= 2: part_score += 2.0 + dora_count
            elif dora_count == 1: part_score += 1.0
            elif dora_count == 0 and part_score < 3.0:
                part_score -= 1.0 # Penalty for trash hand
                
            max_score = max(max_score, part_score)
        
        return max_score 

# -----------------------------------------------------------------------------
# 🀄 Agari Utilities (Win Check & Waits)
# -----------------------------------------------------------------------------
class AgariUtils:
    @staticmethod
    def get_waits_count(hand_34, visible_tiles_34):
        """
        手牌(34枚配列)を受け取り、聴牌しているなら「有効な待ち枚数の合計」を返す。
        ノーテンなら0を返す。
        """
        if USE_NUMBA:
            h = np.array(hand_34, dtype=np.int8)
            v = np.array(visible_tiles_34, dtype=np.int8)
            return get_waits_count_jit(h, v)

        waits = []
        # 34種を1つずつ足してアガリ形になるか試す
        for t_id in range(34):
            if hand_34[t_id] >= 4: continue
            
            hand_34[t_id] += 1 
            if AgariUtils.is_agari(hand_34):
                print(f"DEBUG: Agari Match T:{t_id}", flush=True)
                # アガリ形になるなら、その牌の残り枚数を計算
                # (4枚 - 自分の手牌 - 場に見えている枚数 + 今足した1枚戻し)
                left = 4 - hand_34[t_id] - visible_tiles_34[t_id] + 1 
                if left < 0: left = 0
                waits.append(left)
            hand_34[t_id] -= 1 # 戻す

        return sum(waits)

    @staticmethod
    def is_agari(hand_34, visible_tiles_34=None):
        """ 手牌が和了形（一般手、七対子、国士無双）か判定する """
        # print(f"DEBUG: is_agari called. USE_NUMBA={USE_NUMBA}")
        if USE_NUMBA:
            return is_agari_jit(np.array(hand_34, dtype=np.int8))
            
        # print("DEBUG: is_agari Python Mode")
        
        # 1. 国士無双 (13種14牌)
        # 1,9,字牌の種数を確認
        yao_indices = [0,8,9,17,18,26,27,28,29,30,31,32,33]
        is_kokushi = True
        has_pair = False
        for idx in yao_indices:
            if hand_34[idx] == 0:
                is_kokushi = False
                break
            if hand_34[idx] >= 2:
                has_pair = True
        
        if is_kokushi and has_pair:
            return True # 国士無双アガリ

        # 2. 七対子 (7個の対子)
        # 4枚使いは通常「2対子」として扱わないため、単純に種類のカウントで判定
        pair_count = sum(1 for c in hand_34 if c >= 2)
        if pair_count == 7:
            return True

        # 3. 一般手 (4面子1雀頭)
        # まず雀頭（2枚以上ある牌）の候補を探す
        for head_idx in range(34):
            if hand_34[head_idx] >= 2:
                hand_34[head_idx] -= 2 # 雀頭を抜く
                
                # 残りの12枚がすべて面子（順子or刻子）で構成されているか判定
                if AgariUtils._is_all_mentsu(hand_34):
                    hand_34[head_idx] += 2 # 戻してTrue
                    return True
                
                hand_34[head_idx] += 2 # ダメなら戻して次へ
        
        return False

    @staticmethod
    def _is_all_mentsu(hand_34):
        """
        雀頭を除いた残りの牌が、すべて面子(3枚組)に分解できるか判定。
        色ごとに分けて判定することで高速化・正確化。
        """
        # 萬子 (0-8)
        if not AgariUtils._decompose_suit_memo(bytes(hand_34[0:9])): return False
        # 筒子 (9-17)
        if not AgariUtils._decompose_suit_memo(bytes(hand_34[9:18])): return False
        # 索子 (18-26)
        if not AgariUtils._decompose_suit_memo(bytes(hand_34[18:27])): return False
        # 字牌 (27-33) - 字牌は順子がないので判定が簡単
        for i in range(27, 34):
            if hand_34[i] % 3 != 0: return False
            
        return True

    @staticmethod
    @functools.lru_cache(maxsize=16384)
    def _decompose_suit_memo(counts_bytes):
        """
        指定された範囲（一色分、0-8）の牌が面子だけで構成されているか再帰判定。 Memoized version.
        bytes inputs (faster hash).
        """
        # print(f"DEBUG: decompose_suit_memo called with {counts_tuple}")
        return AgariUtils._recursion_decompose(list(counts_bytes), 0)

    @staticmethod
    def _recursion_decompose(counts, idx):
        # print(f"DEBUG: decompose idx={idx} counts={counts}")
        # 基底条件: 最後まで到達したら成功（残りがすべて0なら）
        # Optimization: Check trailing zeros first if needed, but recursion handles it
        # Actually logic below ensures we only proceed if we consumed tiles.
        # But we need to ensure all tiles are consumed.
        # If we just skipped 0s, and reached end, it's fine.
        # If we skipped non-0s? The logic below:
        # if counts[idx]==0 -> next.
        # if >0 -> try consume. if fail -> return False.
        # So if we reach end, we are good.
        if idx >= len(counts):
            return True 

        if counts[idx] == 0:
            return AgariUtils._recursion_decompose(counts, idx + 1)
            
        # 刻子トライ
        if counts[idx] >= 3:
            counts[idx] -= 3
            if AgariUtils._recursion_decompose(counts, idx): return True
            counts[idx] += 3
            
        # 順子トライ (9番目・8番目は順子の先頭になれない)
        if idx < 7:
            if counts[idx] > 0 and counts[idx+1] > 0 and counts[idx+2] > 0:
                counts[idx] -= 1; counts[idx+1] -= 1; counts[idx+2] -= 1
                if AgariUtils._recursion_decompose(counts, idx): return True
                counts[idx] += 1; counts[idx+1] += 1; counts[idx+2] += 1
                
        return False

# -----------------------------------------------------------------------------
# 📊 Rank & Score Manager (Strategic Awareness)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 🏆 Full Yaku Logic (World Class Precision)
# -----------------------------------------------------------------------------
class YakuUtils:
    @staticmethod
    def get_hand_han(hand_34, melds, ron_tile, is_riichi, is_tsumo, round_wind, self_wind, doras_count):
        """
        Calculates the exact Han count for a given winning hand.
        Returns (han_total, yaku_list)
        melds: list of {'type':'pon'/'chi'/'kan', 'tiles':[...]}
        """
        # 1. Base Setup
        is_menzen = (len(melds) == 0)
        
        yaku_list = []
        han_total = 0
        
        # --- Pre-Yaku ---
        if is_riichi:
            yaku_list.append("Riichi")
            han_total += 1
            # Ippatsu/Ura not calculated here (handled by EV prob)
            
        if is_menzen and is_tsumo:
            yaku_list.append("Menzen Tsumo")
            han_total += 1
            
        # Dora
        if doras_count > 0:
            # Dora adds generic Han but is not a "Yaku" for Yaku-requirement.
            # But the return value is total Han.
            # CAUTION: If only Dora and No Yaku -> Cannot win. 
            pass 

        # 3. Special Hands
        # Chiitoi Check
        pairs = sum(1 for c in hand_34 if c >= 2)
        if pairs == 7:
            # Chiitoi Logic
            yaku_han = 2 # Chiitoi base
            yaku_names = ["Chiitoi"]
            
            # Tanyao Chiitoi
            is_tanyao = True
            terminals = [0,8,9,17,18,26] + list(range(27,34))
            for i in range(34):
                if hand_34[i] > 0 and i in terminals: is_tanyao = False; break
            if is_tanyao: yaku_names.append("Tanyao"); yaku_han += 1
            
            # Honitsu Chiitoi
            suits = set()
            has_honor = False
            for i in range(34):
                if hand_34[i] > 0:
                    if i >= 27: has_honor = True
                    else: suits.add(i // 9)
            if len(suits) == 1 and has_honor: yaku_names.append("Honitsu"); yaku_han += 3
            if len(suits) == 1 and not has_honor: yaku_names.append("Chinitsu"); yaku_han += 6
            
            # Tsuiso (All Honors)
            if len(suits) == 0 and has_honor: yaku_names.append("Tsuiso"); yaku_han += 13 # Yakuman
            
            han_total += yaku_han
            yaku_list.extend(yaku_names)
            return han_total + doras_count, yaku_list

        # 4. Standard Form Decomposition
        # We need optimal partition to maximize Han.
        # Reuse StructureUtils to get all partitions.
        partitions = StructureUtils.get_all_partitions(hand_34)
        
        max_yaku_han = 0
        best_yaku_list = []
        
        for part in partitions:
            # Must have 4 mentsu + 1 pair
            # part: {'mentsu':[(type, idx)], 'tatsu':[], 'pair':[idx], 'iso':[]}
            # Filter valid Agari partitions
            if len(part['mentsu']) != 4 or len(part['pair']) != 1:
                continue
                
            current_yaku_han = 0
            current_yaku_names = []
            
            if is_riichi: current_yaku_han += 1; current_yaku_names.append("Riichi")
            if is_menzen and is_tsumo: current_yaku_han += 1; current_yaku_names.append("Tsumo")
            
            blocks = []
            for m in part['mentsu']: blocks.append({'type':m[0], 'start':m[1]}) 
            pair_idx = part['pair'][0]
            
            # --- Yaku Checks ---
            
            # Tanyao
            is_tanyao = True
            terminals = [0,8,9,17,18,26] + list(range(27,34))
            if pair_idx in terminals: is_tanyao = False
            for b in blocks:
                s = b['start']
                if b['type'] == 'koutsu':
                    if s in terminals: is_tanyao = False
                elif b['type'] == 'shunta':
                    if s % 9 == 0 or s % 9 == 6: is_tanyao = False
            if is_tanyao: current_yaku_han += 1; current_yaku_names.append("Tanyao")
            
            # Yakuhai
            # Dragons
            for b in blocks:
                if b['type'] == 'koutsu':
                    if b['start'] in [31, 32, 33]: # White, Green, Red
                        current_yaku_han += 1; current_yaku_names.append("Yakuhai (Dragon)")
            # Winds
            round_tile = 27 + (round_wind % 4)
            for b in blocks:
                if b['type'] == 'koutsu' and b['start'] == round_tile:
                    current_yaku_han += 1; current_yaku_names.append("Yakuhai (Round)")
            self_tile = 27 + (self_wind % 4)
            for b in blocks:
                if b['type'] == 'koutsu' and b['start'] == self_tile:
                    current_yaku_han += 1; current_yaku_names.append("Yakuhai (Self)")
            
            # Pinfu (Menzen Only)
            if is_menzen:
                is_pinfu = True
                for b in blocks:
                    if b['type'] == 'koutsu': is_pinfu = False; break
                
                if pair_idx in [31, 32, 33]: is_pinfu = False
                if pair_idx == round_tile: is_pinfu = False
                if pair_idx == self_tile: is_pinfu = False
                
                # Wait Validity Check (Heuristic)
                # Ideally check if ron_tile completes a Ryanmen wait
                if is_pinfu: current_yaku_han += 1; current_yaku_names.append("Pinfu")
            
            # Iipeiko (Pure Double Chow) - Menzen Only
            if is_menzen:
                shuntsu_starts = sorted([b['start'] for b in blocks if b['type'] == 'shunta'])
                import collections
                ctr = collections.Counter(shuntsu_starts)
                iipeiko_count = 0
                for k, v in ctr.items():
                    if v >= 2: iipeiko_count += 1
                if iipeiko_count == 1: current_yaku_han += 1; current_yaku_names.append("Iipeiko")
                if iipeiko_count == 2: current_yaku_han += 3; current_yaku_names.append("Ryanpeiko")
            
            # Sanshoku Doujun
            shunta_list = [b['start'] for b in blocks if b['type'] == 'shunta']
            found_sanshoku = False
            for num in range(7): 
                if (num in shunta_list) and ((num+9) in shunta_list) and ((num+18) in shunta_list):
                    found_sanshoku = True; break
            if found_sanshoku: 
                val = 2 if is_menzen else 1
                current_yaku_han += val; current_yaku_names.append("Sanshoku")
                
            # Itsu (Straight)
            found_itsu = False
            for base in [0, 9, 18]:
                if (base in shunta_list) and ((base+3) in shunta_list) and ((base+6) in shunta_list):
                    found_itsu = True; break
            if found_itsu:
                val = 2 if is_menzen else 1
                current_yaku_han += val; current_yaku_names.append("Itsu")
            
            # Honitsu / Chinitsu
            suits = set()
            has_honor = False
            for i in range(34):
                if hand_34[i] > 0:
                    if i >= 27: has_honor = True
                    else: suits.add(i // 9)
            if len(suits) == 1:
                if has_honor:
                    val = 3 if is_menzen else 2
                    current_yaku_han += val; current_yaku_names.append("Honitsu")
                else:
                    val = 6 if is_menzen else 5
                    current_yaku_han += val; current_yaku_names.append("Chinitsu")
            
            # Toitoi
            koutsu_count = sum(1 for b in blocks if b['type'] == 'koutsu')
            if koutsu_count == 4:
                current_yaku_han += 2; current_yaku_names.append("Toitoi")
            
            # Sananko (Assume all closed for now if Menzen)
            if is_menzen and koutsu_count >= 3:
                current_yaku_han += 2; current_yaku_names.append("Sananko")
                
            # Update Best
            if current_yaku_han > max_yaku_han:
                max_yaku_han = current_yaku_han
                best_yaku_list = current_yaku_names
        
        # Final Verification
        # If no Yaku (max_yaku_han == 0) -> Return 0 (even if Doras exist)
        # Unless it's just a raw han calculation request. 
        # But for decision making, we need to know if we CAN win.
        
        return max_yaku_han + doras_count, best_yaku_list


class RankManager:
    @staticmethod
    def get_strategy(scores, round_val, honba, my_seat, doras_count):
        """
        現在の点数状況と局から、最適な戦略（リスク許容度、重視項目）を算出する。
        """
        # 1. 現状分析
        my_score = scores[my_seat]
        ranked_scores = sorted([(s, i) for i, s in enumerate(scores)], reverse=True)
        my_rank = -1
        for r, (sc, p_idx) in enumerate(ranked_scores):
            if p_idx == my_seat:
                my_rank = r + 1 # 1-4位
                break
        
        # ターゲットとの点差
        diff_to_top = ranked_scores[0][0] - my_score
        diff_to_upper = 0
        diff_to_lower = 0
        
        if my_rank > 1:
            diff_to_upper = ranked_scores[my_rank-2][0] - my_score
        if my_rank < 4:
            diff_to_lower = my_score - ranked_scores[my_rank][0]

        # デフォルト戦略
        strategy = {
            "mode": "Balanced",
            "risk_tolerance": 1.0, # 1.0=Standard, 0.5=Safe, 2.0=Push
            "speed_weight": 0.0,   # 追加評価点
            "value_weight": 0.0,   # 追加評価点
            "min_score_needed": 0,
            "must_riichi": False
        }

        # 2. 進行状況判定
        # round_val: 0-3(East), 4-7(South)
        is_east = (round_val < 4)
        is_south = (round_val >= 4)
        is_oras = (round_val == 7) # South 4
        
        # 3. 戦略決定ロジック
        
        # --- A. 序盤・東場 (Early Game) ---
        if is_east:
            # 基本はフラットだが、極端な点差だけケア
            if my_score < 15000: # 飛び寸前
                strategy["mode"] = "Survival"
                strategy["value_weight"] = 0.2
                strategy["risk_tolerance"] = 1.2 # 攻めるしかない
            elif my_score > 40000: # 大量リード
                strategy["mode"] = "Solid"
                strategy["risk_tolerance"] = 0.7 # 守り気味
        
        # --- B. 中盤・南1-3局 (Middle Game) ---
        elif is_south and not is_oras:
            # 順位固めフェーズ
            if my_rank == 1:
                strategy["mode"] = "Maintain Top"
                strategy["risk_tolerance"] = 0.6
                strategy["speed_weight"] = 0.3 # 局回し優先
            elif my_rank == 4:
                strategy["mode"] = "Avoid Last"
                strategy["risk_tolerance"] = 1.3
                strategy["value_weight"] = 0.3 # 手を作りに行く
                
        # --- C. 終盤・オーラス (All Last) ---
        elif is_oras:
            # 勝利条件・生存条件の厳密適用
            
            # Case 1: Top (逃げ切り)
            if my_rank == 1:
                strategy["mode"] = "Oras Top Escape"
                strategy["risk_tolerance"] = 0.2 # ほぼベタオリ推奨
                strategy["speed_weight"] = 2.0 # 安くてもアガれば勝ち
                # 2位との差が2000点未満ならそこそこ攻める必要があるが、基本は守り
                if diff_to_lower < 2000:
                    strategy["risk_tolerance"] = 0.8
            
            # Case 2: Last (ラス回避必須)
            elif my_rank == 4:
                strategy["mode"] = "Oras Last Desperate"
                strategy["risk_tolerance"] = 2.0 # 全ツッパ
                
                # 3位をまくる条件計算
                target_diff = diff_to_upper
                # 直撃なら差の半分、ツモなら差の1/3... 簡易的に「差分」を必要打点とする
                # 本場(honba) * 300点などを考慮すべきだが、まずは粗点
                strategy["min_score_needed"] = target_diff
                strategy["value_weight"] = 1.5 # 打点至上主義
                
                # 条件を満たさないならリーチ強制
                strategy["must_riichi"] = True 

            # Case 3: 2nd/3rd (上位狙い)
            else:
                strategy["mode"] = "Oras Challenge"
                if diff_to_top < 8000: # 満貫圏内
                    strategy["speed_weight"] = 0.5
                    strategy["value_weight"] = 0.5
                    strategy["min_score_needed"] = diff_to_top
                else:
                    # 無理せず順位キープ
                    strategy["risk_tolerance"] = 0.8

        return strategy

# -----------------------------------------------------------------------------
# 🧠 Parallel PIMC Engine (World Class Logic)
# -----------------------------------------------------------------------------


def _pimc_worker_entry(args):
    """
    Parallel PIMC Worker Entry (Calling Numba JIT Loop)
    args: (seed, my_hand_34, visible_counts_34, cand_indices, cand_is_riichi, sim_limit, doras_ids)
    """
    try:
        from numba_utils import simulate_chunk_jit
        
        # Unpack arguments
        seed, my_hand, visible, c_indices, c_riichi, sim_limit, doras = args
        
        # Call Numba JIT function (Heavy Lifting)
        res_wins, res_scores = simulate_chunk_jit(seed, my_hand, visible, c_indices, c_riichi, sim_limit, doras)
        
        # Pack results back to dictionary
        results = {}
        for i, idx in enumerate(c_indices):
             results[idx] = {
                 "wins": int(res_wins[i]),
                 "score_sum": float(res_scores[i])
             }
        return results

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {}


# Global Singleton Executor
_PIMC_EXECUTOR = None

class PIMCEngine:
    @staticmethod
    def _warmup_numba():
        """
        Warmup Numba (JIT compile)
        """
        try:
            print("[INFO] PIMC Engine: Warming up Numba JIT functions...", flush=True)
            if USE_NUMBA:
                from numba_utils import simulate_chunk_jit
                # Dummy Arguments
                seed = 123
                mh = np.zeros(34, dtype=np.int8)
                vc = np.zeros(34, dtype=np.int8)
                ci = np.array([0, 1], dtype=np.int32)
                cr = np.array([False, False], dtype=np.bool_)
                di = np.array([4], dtype=np.int8)
                simulate_chunk_jit(seed, mh, vc, ci, cr, 10, di)
            print("[INFO] PIMC Engine: Numba Warmup Complete.", flush=True)
        except Exception as e:
            print(f"[WARN] PIMC Engine: Numba Warmup Failed: {e}", flush=True)

    @staticmethod
    def _reset_executor():
        global _PIMC_EXECUTOR
        print("[WARN] PIMC Engine: Resetting ThreadPoolExecutor...", flush=True)
        if _PIMC_EXECUTOR:
            try:
                _PIMC_EXECUTOR.shutdown(wait=False, cancel_futures=True)
            except: pass
        _PIMC_EXECUTOR = None

    @staticmethod
    def run(hands_arr, visible_counts, candidates, doras_ids, num_worlds=20):
        """
        Parallel PIMC Execution (Threading + Numba Loop)
        """
        global _PIMC_EXECUTOR
        import concurrent.futures
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import random
        import os
        import time
        
        if _PIMC_EXECUTOR is None:
            PIMCEngine._warmup_numba()
            # Max workers = physical cores for compute bound Numba nogil
            max_w = os.cpu_count() or 8
            print(f"[INFO] PIMC Engine: Initializing ThreadPoolExecutor with {max_w} workers...", flush=True)
            _PIMC_EXECUTOR = ThreadPoolExecutor(max_workers=max_w)
            
        my_hand = np.array(hands_arr[3], dtype=np.int8)
        visible_counts_np = np.array(visible_counts, dtype=np.int8)
        doras_np = np.array(doras_ids, dtype=np.int8)
        
        # Prepare candidates arrays
        # Filter valid candidates (idx < 34)
        valid_cands = [c for c in candidates if c["idx"] < 34]
        
        if not valid_cands:
            return candidates # No valid candidates to simulate
            
        cand_indices = np.array([c["idx"] for c in valid_cands], dtype=np.int32)
        cand_is_riichi = np.array([c.get("is_riichi", False) for c in valid_cands], dtype=np.bool_)
        
        total_simulations = num_worlds 
        
        # Chunking: Reduce Python overhead (Submit calls)
        # 3000 sims / 20 threads = 150.
        # Chunk size 200 is good. Python overhead becomes negligible.
        CHUNK_SIZE = 200
        num_chunks = (total_simulations + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        tasks = []
        for i in range(num_chunks):
            sims_in_chunk = CHUNK_SIZE
            if i == num_chunks - 1:
                remainder = total_simulations % CHUNK_SIZE
                if remainder > 0: sims_in_chunk = remainder
            
            if sims_in_chunk <= 0: continue
            
            seed = random.randint(0, 2**31 - 1)
            # Pass numpy arrays. They are read-only (copied inside JIT usually or treated as such)
            # Note: Thread safety. Is passing the same numpy array safe if JIT reads only? Yes.
            tasks.append((seed, my_hand, visible_counts_np, cand_indices, cand_is_riichi, sims_in_chunk, doras_np))

        s_time = time.time()
        
        futures = [_PIMC_EXECUTOR.submit(_pimc_worker_entry, t) for t in tasks]
        
        final_res = {idx: {"wins": 0, "score_sum": 0} for idx in cand_indices}
        
        for f in as_completed(futures):
            res = f.result()
            if not res: continue
            for idx, val in res.items():
                final_res[idx]["wins"] += val["wins"]
                final_res[idx]["score_sum"] += val["score_sum"]
            
        e_time = time.time()
        print(f"DEBUG: PIMC Thread-Numba Time: {e_time - s_time:.4f}s for {total_simulations} sims", flush=True)
        
        # Update scores
        for c in candidates:
            if c["idx"] >= 34: 
                 c["pimc_score"] = 0
                 continue
            
            idx = c["idx"]
            if total_simulations > 0:
                 avg_score = final_res[idx]["score_sum"] / total_simulations
                 c["pimc_score"] = avg_score
            else:
                 c["pimc_score"] = 0
        
        candidates.sort(key=lambda x: x.get("pimc_score", 0), reverse=True)
        return candidates


# -----------------------------------------------------------------------------
# 🧩 Structure Engine (Hand Decomposition)
# -----------------------------------------------------------------------------
class StructureUtils:
    @staticmethod
    def get_all_partitions(hand_34):
        """
        手牌を「面子・対子・搭子」の組み合わせに全分解する (Recursive DFS)
        戻り値: list of partitions
        partition = {'mentsu':[], 'tatsu':[], 'pair':[], 'iso':[]}
        mentsu item: (type, first_tile_idx) e.g. ('shunta', 2) or ('koutsu', 5)
        """
        results = []
        StructureUtils._dfs_partition(hand_34[:], 0, {'mentsu':[], 'tatsu':[], 'pair':[], 'iso':[]}, results)
        return results

    @staticmethod
    def _dfs_partition(hand, idx, current_struct, results):
        if idx >= 34:
             # 残った牌はすべて孤立牌扱い
             # しかしここに来る前に処理するので通常は全て0になっているはず
             # 念のため残りをIsoに入れる
             final_struct = {
                 'mentsu': list(current_struct['mentsu']),
                 'tatsu': list(current_struct['tatsu']),
                 'pair': list(current_struct['pair']),
                 'iso': list(current_struct['iso'])
             }
             for i in range(34):
                 for _ in range(hand[i]):
                     final_struct['iso'].append(i)
             results.append(final_struct)
             return

        if hand[idx] == 0:
            StructureUtils._dfs_partition(hand, idx + 1, current_struct, results)
            return

        # 枝刈り: 5ブロック以上は不要だが、搭子オーバーなどもあり得るので緩めに
        
        # Priority 1: Mentsu (Koutsu)
        if hand[idx] >= 3:
            hand[idx] -= 3
            current_struct['mentsu'].append(('koutsu', idx))
            StructureUtils._dfs_partition(hand, idx, current_struct, results)
            current_struct['mentsu'].pop()
            hand[idx] += 3
            
        # Priority 2: Mentsu (Shuntsu)
        if idx < 27 and (idx % 9) < 7:
            if hand[idx] > 0 and hand[idx+1] > 0 and hand[idx+2] > 0:
                hand[idx] -= 1; hand[idx+1] -= 1; hand[idx+2] -= 1
                current_struct['mentsu'].append(('shunta', idx))
                StructureUtils._dfs_partition(hand, idx, current_struct, results)
                current_struct['mentsu'].pop()
                hand[idx] += 1; hand[idx+1] += 1; hand[idx+2] += 1

        # Priority 3: Pair (only 1 allowed usually, but for exhaustive search allow multiple then filter)
        # ここでは一般形のアガリを目指すため、雀頭は1つという制約は評価時に行う
        if hand[idx] >= 2:
            hand[idx] -= 2
            current_struct['pair'].append(idx)
            StructureUtils._dfs_partition(hand, idx, current_struct, results)
            current_struct['pair'].pop()
            hand[idx] += 2

        # Priority 4: Tatsu (Ryanmen/Kanchan/Penchan)
        if idx < 27:
            # Ryanmen/Penchan (idx, idx+1)
            if (idx % 9) < 8 and hand[idx] > 0 and hand[idx+1] > 0:
                hand[idx] -= 1; hand[idx+1] -= 1
                current_struct['tatsu'].append(('ryanmen', idx)) # or penchan
                StructureUtils._dfs_partition(hand, idx, current_struct, results)
                current_struct['tatsu'].pop()
                hand[idx] += 1; hand[idx+1] += 1
            
            # Kanchan (idx, idx+2)
            if (idx % 9) < 7 and hand[idx] > 0 and hand[idx+2] > 0:
                hand[idx] -= 1; hand[idx+2] -= 1
                current_struct['tatsu'].append(('kanchan', idx))
                StructureUtils._dfs_partition(hand, idx, current_struct, results)
                current_struct['tatsu'].pop()
                hand[idx] += 1; hand[idx+2] += 1

        # Priority 5: Isolated (Skip)
        hand[idx] -= 1
        current_struct['iso'].append(idx)
        StructureUtils._dfs_partition(hand, idx, current_struct, results)
        current_struct['iso'].pop()
        hand[idx] += 1

class ValueEstimator:
    @staticmethod
    def estimate_hand_value(hand_34, doras_ids):
        """
        手牌の価値を全パターン探索で厳密に評価する
        """
        # 1. 構造分解
        partitions = StructureUtils.get_all_partitions(hand_34[:])
        if not partitions: return 0
        
        max_score = 0
        
        for part in partitions:
            # 5ブロック制限チェック (Mentsu + Tatsu + Pair <= 5) など
            # ここでは純粋に役の可能性を足し合わせる
            
            score = 1000 # Base
            
            # --- 手役コンポーネント解析 ---
            
            mentsu_list = part['mentsu'] # list of (type, first_idx)
            tatsu_list = part['tatsu']
            pair_list = part['pair']
            
            all_blocks = []
            for m in mentsu_list: all_blocks.append({'type': m[0], 'idx': m[1], 'is_complete': True})
            for t in tatsu_list: all_blocks.append({'type': t[0], 'idx': t[1], 'is_complete': False})
            
            # --- Sanshoku (三色) ---
            # 順子(shunta)または搭子(ryanmen/kanchan)での三色ポテンシャル
            # マンズ(0-8), ピンズ(9-17), ソーズ(18-26)
            # idx % 9 が同じブロックが3色あるか
            counts_by_mod = [0]*9
            for b in all_blocks:
                t_idx = b['idx']
                if t_idx < 27:
                    counts_by_mod[t_idx % 9] += 1
            
            if any(c >= 3 for c in counts_by_mod):
                score += 2000 # Potentially Sanshoku
            
            # --- Ittsu (一気通貫) ---
            # 123, 456, 789 in same suit
            for color in range(3):
                # 必要な開始インデックス
                # 0, 3, 6 (+ color*9)
                base = color * 9
                has_123 = any(b['idx'] == base+0 for b in all_blocks)
                has_456 = any(b['idx'] == base+3 for b in all_blocks)
                has_789 = any(b['idx'] == base+6 for b in all_blocks)
                if has_123 and has_456 and has_789:
                    score += 2000
            
            # --- Tanyao (タンヤオ) ---
            # 全ブロックがタンヤオ牌で構成されているか
            is_tanyao = True
            if part['iso']: is_tanyao = False # 孤立牌も含めてチェックすべきだが厳密には完成時
            
            terminals = [0,8,9,17,18,26] + list(range(27,34))
            
            # ブロックチェック
            for b in all_blocks:
                # Shunta/Tatsu start at idx.
                # If idx is terminal? No, check content.
                # Shunta(idx): idx, idx+1, idx+2.
                start = b['idx']
                if b['type'] in ['shunta', 'ryanmen', 'kanchan']:
                     # 順子系: 123(0) -> 0,1,2 contains 0(Terminal)
                     # 234(1) -> 1,2,3 OK
                     # ...
                     # 789(6) -> 6,7,8 contains 8(Terminal)
                     if start % 9 == 0 or start % 9 == 6: # 123 or 789
                         is_tanyao = False
                     if b['type'] == 'ryanmen' and (start % 9 == 0 or start % 9 == 7): # 12 or 89
                         is_tanyao = False
                else: # Koutsu / Pair
                     if start in terminals: is_tanyao = False
            
            for p in part['pair']:
                if p in terminals: is_tanyao = False
                
            if is_tanyao: score += 1500
            
            # --- Yakuhai (役牌) ---
            for m in mentsu_list:
                if m[0] == 'koutsu' and m[1] >= 27:
                    score += 1200
            for p in pair_list:
                if p >= 27: score += 400
                
            # --- Dora ---
            dora_score = 0
            # 手元のドラ枚数(固定なのでPartition依存しないが、使い切れているか見るのも手)
            # ここでは単純加算
            for d_id in doras_ids:
                dora_score += hand_34[d_id] * 1200
            score += dora_score

            # Update Max
            if score > max_score:
                max_score = score

        return max_score

