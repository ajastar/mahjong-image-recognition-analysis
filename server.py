import os
import io
import time
import datetime 
import numpy as np
import cv2
import math
import collections
import glob
import torch
import logging
import sys
import hashlib

# --- Numba Stability Configuration for Windows ---
# Disable Intel SVML to prevent potential crashes in multiprocessing
os.environ['NUMBA_DISABLE_INTEL_SVML'] = '1'
# Force workqueue threading layer (safest for Windows spawn)
os.environ['NUMBA_THREADING_LAYER'] = 'workqueue'

import torch.nn as nn
import re
from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
import easyocr
import math  
import concurrent.futures
import random
import copy
from mahjong_logic import TILE_MAP, INV_TILE_MAP, NAKI_LABELS, get_suji, ShantenUtils, UkeireUtils, SafetyUtils, AgariUtils, RankManager, PIMCEngine, StructureUtils, ValueEstimator, NakiUtils, NakiPlanner

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# ==========================================
# ⚙️ 設定
# ==========================================
weights_path = r"E:\AI_Project_Hub\Mahjong_Maker\runs\jantama_absolute_limit_1280px\weights\best.onnx"
if not os.path.exists(weights_path):
    weights_path = r"E:\AI_Project_Hub\Mahjong_Maker\runs\jantama_absolute_limit_1280px\best.pt"

BRAIN_MODEL_PATH = "./checkpoints_superhuman_blackwell/mahjong_god_elite_tuned.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

base_dir = r"E:\AI_Project_Hub\Mahjong_Maker"
template_dir = os.path.join(base_dir, "templates")
debug_save_path = os.path.join(base_dir, "debug_result.jpg")

# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# 📁 ここにデータ収集設定を追加します
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
COLLECT_DATA = True 
COLLECT_DIR = os.path.join(base_dir, "dataset_collector")

if COLLECT_DATA:
    os.makedirs(os.path.join(COLLECT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(COLLECT_DIR, "labels"), exist_ok=True)
    print(f"📂 Data Collector: ON -> {COLLECT_DIR}")


# ==========================================
# 🧠 Brain Definition
# ==========================================
class Partial_Conv1d(nn.Module):
    def __init__(self, dim, n_div=4, forward="split_cat"):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv1d(self.dim_conv3, self.dim_conv3, kernel_size=3, stride=1, padding=1, bias=False)
        self.forward = self.forward_split_cat if forward == "split_cat" else self.forward_slicing
    def forward_slicing(self, x):
        x = x.clone(); x[:, :self.dim_conv3, :] = self.partial_conv3(x[:, :self.dim_conv3, :]); return x
    def forward_split_cat(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1); return torch.cat((x1, x2), 1)

class FasterBlock(nn.Module):
    def __init__(self, dim, n_div=4, mlp_ratio=2., drop_path=0.):
        super().__init__()
        self.pconv = Partial_Conv1d(dim, n_div, forward="split_cat")
        self.conv1 = nn.Conv1d(dim, int(dim * mlp_ratio), kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(int(dim * mlp_ratio))
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(int(dim * mlp_ratio), dim, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(dim)
        self.drop_path = nn.Identity() if drop_path <= 0. else nn.Dropout(drop_path)
    def forward(self, x):
        return x + self.drop_path(self.bn2(self.conv2(self.act(self.bn1(self.conv1(self.pconv(x)))))))

class FasterMahjongNet(nn.Module):
    def __init__(self, in_chans=80, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
        super().__init__()
        self.downsample_layers = nn.ModuleList() 
        stem = nn.Sequential(nn.Conv1d(in_chans, dims[0], kernel_size=4, stride=1, padding=2), nn.BatchNorm1d(dims[0]))
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(nn.BatchNorm1d(dims[i]), nn.Conv1d(dims[i], dims[i+1], kernel_size=2, stride=1, padding=0)))
        self.stages = nn.ModuleList() 
        for i in range(4):
            self.stages.append(nn.Sequential(*[FasterBlock(dim=dims[i]) for j in range(depths[i])]))
        self.norm = nn.BatchNorm1d(dims[-1])
        self.head_act = nn.Linear(dims[-1], 35)
        self.head_naki = nn.Linear(dims[-1], 5)
    def forward(self, x):
        if x.dim() == 4: x = x.squeeze(-1)
        for i in range(4): x = self.stages[i](self.downsample_layers[i](x))
        x = self.norm(x.mean([-1]))
        return self.head_act(x), self.head_naki(x)

# ---------------------------------------------------------
# 🀄 TILE MAP
# ---------------------------------------------------------

def encode_state_god(hands, discards_hist, tsumogiri_hist, melds, scores, doras, 
                     round_info, honba, riichi_sticks, ippatsu_flags, 
                     p_idx, full_hands_ids, target_tile_id=-1):
    NUM_CHANNELS = 80
    obs = np.zeros((NUM_CHANNELS, 34, 1), dtype=np.float32)

    h = hands[p_idx]
    for t in range(34):
        cnt = h[t]
        for c in range(1, 5):
            if cnt >= c: obs[c-1, t, :] = 1

    for i in range(4):
        target_p = (p_idx + i) % 4
        hist = discards_hist[target_p]
        tsumo_dat = tsumogiri_hist[target_p]
        for j, t in enumerate(hist):
            if t >= 34: continue
            row = min(j // 6, 3)
            order_val = ((j % 6) + 1) / 6.0
            obs[4 + (i*4) + row, t, :] = order_val
            if j < len(tsumo_dat) and tsumo_dat[j]:
                obs[20 + i, t, :] = 1

    for i in range(4):
        target_p = (p_idx + i) % 4
        for group in melds[target_p]:
            for t in group:
                if t < 34: obs[24 + i, t, :] = 1

    for d in doras:
        if d < 34: obs[28, d, :] = 1

    # ★Fix: Pass round_wind and self_wind explicitly if possible, or derive.
    # actually round_val encodes it.
    
    for i in range(4):
        if riichi_sticks[(p_idx + i) % 4]: obs[29 + i, :, :] = 1

    obs[33, :, :] = min(scores[p_idx] / 50000.0, 1.0)
    rank = 1
    for s in scores:
        if s > scores[p_idx]: rank += 1
    obs[34, :, :] = rank / 4.0
    obs[35, :, :] = honba / 10.0
    obs[36, :, :] = len(doras) / 5.0

    cur_ids = full_hands_ids[p_idx]
    if 16 in cur_ids: obs[37, :, :] = 1 
    if 52 in cur_ids: obs[38, :, :] = 1
    if 88 in cur_ids: obs[39, :, :] = 1

    for i in range(4):
        target_p = (p_idx + i) % 4
        d_list = discards_hist[target_p]
        if riichi_sticks[target_p]:
            for t in d_list:
                if t < 34: 
                    obs[40 + i, t, :] = 1
                    for s in get_suji(t):
                        if s < 34: obs[44 + i, s, :] = 1

    visible = np.zeros(34, dtype=int)
    for t in range(34): visible[t] += hands[p_idx][t]
    for p in range(4):
        for t in discards_hist[p]:
            if t < 34: visible[t] += 1
        for g in melds[p]:
            for t in g:
                if t < 34: visible[t] += 1
    for d in doras:
        if d < 34: visible[d] += 1
    for t in range(34):
        cnt = visible[t]
        for c in range(1, 5):
            if cnt >= c: obs[48 + c - 1, t, :] = 1

    for i in range(4):
        if ippatsu_flags[(p_idx + i) % 4]: obs[52 + i, :, :] = 1

    for i in range(1, 4):
        target_p = (p_idx + i) % 4
        diff = scores[target_p] - scores[p_idx]
        obs[55 + i, :, :] = max(-1.0, min(1.0, diff / 50000.0))

    total_d = sum(len(d) for d in discards_hist)
    obs[60, :, :] = total_d / 70.0

    bakaze_code = (round_info // 4) % 4 
    bakaze_tile = 27 + bakaze_code
    if bakaze_tile < 34: obs[63, bakaze_tile, :] = 1
    
    dealer_idx = round_info % 4
    jikaze_code = (p_idx - dealer_idx + 4) % 4
    jikaze_tile = 27 + jikaze_code
    if jikaze_tile < 34: obs[64, jikaze_tile, :] = 1

    obs[79, :, :] = 1
    return obs

# ==========================================
# 🛠️ 初期化処理
# ==========================================
model_main = None
if os.path.exists(weights_path):
    model_main = YOLO(weights_path, task='detect')
    print("✅ YOLO Ready.")

reader = None
try:
    reader = easyocr.Reader(['en'], gpu=True)
    print("✅ OCR Ready.")
except: pass

brain_model = None
if os.path.exists(BRAIN_MODEL_PATH):
    print("🧠 Loading BRAIN...")
    # ★変更: .half() を追加してFP16モードにする
    brain_model = FasterMahjongNet(in_chans=80).to(DEVICE).half()
    try:
        state_dict = torch.load(BRAIN_MODEL_PATH, map_location=DEVICE)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                new_state_dict[k.replace("_orig_mod.", "")] = v 
            else:
                new_state_dict[k] = v
        brain_model.load_state_dict(new_state_dict)
        brain_model.eval()
        print("✅ Brain Ready (Fixed Prefix).")

        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # 🚀 [追加] JITコンパイル (高速化の魔法)
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        try:
            print("⚡ Optimizing Brain with TorchScript...")
            # ダミー入力を作成して、計算グラフを固定化する
            # (Batch_Size=1, Channels=80, Height=34, Width=1)
            dummy_input = torch.randn(1, 80, 34, 1).to(DEVICE).half()
            
            # トレース実行 (これでモデルがC++レベルにコンパイルされます)
            brain_model = torch.jit.trace(brain_model, dummy_input)
            
            # 内部の計算最適化を有効化
            torch.backends.cudnn.benchmark = True
            print("✅ Brain JIT Compiled & Optimized!")
        except Exception as e:
            print(f"⚠️ JIT Optimization Failed (Run as normal): {e}")
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    except Exception as e:
        print(f"❌ Brain Load Error: {e}")
else:
    print(f"❌ Brain model not found at {BRAIN_MODEL_PATH}")

class TemplateMatcher:
    def __init__(self, folder):
        self.templates = {}
        paths = glob.glob(os.path.join(folder, "*.*"))
        for p in paths:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is not None:
                label = os.path.splitext(os.path.basename(p))[0]
                self.templates[label] = img
    def match(self, target_img, threshold=0.8, candidates=None, return_best=False):
        if not self.templates: return None, 0.0
        best_score = -1.0; best_label = None
        h, w = target_img.shape[:2]
        
        items_to_check = self.templates.items()
        if candidates:
            items_to_check = [(k, v) for k, v in self.templates.items() if k in candidates]
            
        for label, temp_img in items_to_check:
            try:
                # Resize template to target size (Simple scaling)
                # Note: This assumes crop and template are roughly same aspect ratio
                res = cv2.matchTemplate(target_img, cv2.resize(temp_img, (w, h)), cv2.TM_CCOEFF_NORMED)
                score = np.max(res)
                # DEBUG: Print score for analysis
                if candidates: print(f"DEBUG_MATCH: {label} = {score:.4f}")
                if score > best_score: best_score = score; best_label = label
            except Exception as e: 
                pass
        
        if return_best and best_label:
            return best_label, best_score
        return (best_label, best_score) if best_score > threshold else (None, best_score)

matcher = TemplateMatcher(template_dir)

# ★★★ ステート管理クラス (複数人リーチ & パターンマッチ & 10分保持) ★★★
class GameStateManager:
    def __init__(self):
        self.memories = {}
    
    def update(self, opps, reach_status):
        now = time.time()
        
        # 1. 期限切れ削除 (10分)
        expired = [s for s, m in self.memories.items() if now - m['timestamp'] > 600]
        for s in expired: del self.memories[s]

        # 4人独立して判定
        for seat in range(4):
            curr_river = [t['label'] for t in opps[seat]['river_data']]
            is_matched = False
            
            # 2. メモリ照合 (パターンマッチ)
            if seat in self.memories:
                mem = self.memories[seat]
                snap = mem['snapshot']
                # 現在の河の先頭が、記憶している河と一致するか
                if len(curr_river) >= len(snap) and curr_river[:len(snap)] == snap:
                    is_matched = True
                    idx = mem['decl_idx']
                    if 0 <= idx < len(opps[seat]['river_data']):
                        opps[seat]['river_data'][idx]['is_declaration'] = True
                    self.memories[seat]['timestamp'] = now

            # 3. 新規リーチ検出 (まだ記憶していない、かつ現在リーチ棒がある場合)
            if reach_status[seat] and not is_matched:
                if len(curr_river) > 0:
                    self.memories[seat] = {
                        'snapshot': list(curr_river),
                        'decl_idx': len(curr_river) - 1, # 最後の牌を宣言牌とする
                        'timestamp': now
                    }
                    opps[seat]['river_data'][-1]['is_declaration'] = True

state_manager = GameStateManager()

# ==========================================
# 🧠 AI思考ロジック (Strategyクラス・食い替え防止＆ポン改善版)
# ==========================================
class Strategy:
    @staticmethod
    def decide_discard(hand_objs, melds_objs, doras_objs, hud_info, state_manager, brain_model, opps):
        print("DEBUG: decide_discard CALLED", flush=True)
        t_s = time.time()
        # 1. Input Parsing
        hand = [t['label'] for t in hand_objs]
        melds = [t['label'] for t in melds_objs]
        indicators = [t['label'] for t in doras_objs]
        
        hands_arr = [[0]*34 for _ in range(4)]
        full_hands_ids_dummy = [[], [], [], []]
        p_idx = 3 

        for t_str in hand:
            if t_str in TILE_MAP:
                hands_arr[p_idx][TILE_MAP[t_str]] += 1
                if t_str == "0m": full_hands_ids_dummy[p_idx].append(16)
                elif t_str == "0p": full_hands_ids_dummy[p_idx].append(52)
                elif t_str == "0s": full_hands_ids_dummy[p_idx].append(88)

        discards_hist = [[] for _ in range(4)]
        riichi_sticks = [False]*4
        scores = [25000]*4
        melds_arr = [[] for _ in range(4)]
        score_map = {0:"Score_Right", 1:"Score_Top", 2:"Score_Left", 3:"Score_Self"}

        honba_val = 0
        try: honba_val = int(hud_info.get("Info_Honba", "0"))
        except: pass
        
        wind_label = hud_info.get("Round_Wind", "East1")
        round_val = 0
        if "South" in wind_label or "S" in wind_label: round_val += 4
        num_match = re.search(r'\d+', wind_label)
        if num_match: round_val += int(num_match.group()) - 1 

        for i in range(4):
            riichi_sticks[i] = opps[i]['reach']
            try: scores[i] = int(hud_info.get(score_map[i], "25000"))
            except: pass
            for t_label in opps[i]['river']:
                if t_label in TILE_MAP: discards_hist[i].append(TILE_MAP[t_label])
            
            current_seat_melds = list(opps[i]['melds'])
            if i == 3: current_seat_melds.extend(melds) 
            for m_label in current_seat_melds:
                if m_label in TILE_MAP: melds_arr[i].append([TILE_MAP[m_label]])

        doras_ids = []
        for d in indicators:
            if d in TILE_MAP:
                nid = TILE_MAP[d]
                if nid < 27: nx = nid - 8 if nid % 9 == 8 else nid + 1
                elif nid < 31: nx = 27 if nid == 30 else nid + 1
                else: nx = 31 if nid == 33 else nid + 1
                doras_ids.append(nx)

        # 2. Inference
        best_move = "Wait"
        confidence = 0.0
        prediction_text = "スルー"
        riichi_advice = "🤫ダマ"
        top_2_candidates = []
        
        if brain_model and len(hand) > 0:
            try:
                # =====================================================
                # 🧮 共通変数の準備
                # =====================================================
                
                # ★追加: ドラ継続管理 (Global)
                global last_round_val, accumulated_doras
                if 'last_round_val' not in globals(): last_round_val = -1
                if 'accumulated_doras' not in globals(): accumulated_doras = set()

                # 局が変わったらドラリセット
                if round_val != last_round_val:
                    accumulated_doras = set()
                    last_round_val = round_val
                    print(f"🔄 Round Changed to {round_val}: Resetting Accumulated Doras.")

                # 今回見えているドラを追加
                for d in indicators:
                    accumulated_doras.add(d)

                # 実際にAI/表示に使うのは累積されたドラを利用
                final_indicators = list(accumulated_doras)
                
                # doras_ids 再生成
                doras_ids = []
                for d in final_indicators:
                    if d in TILE_MAP:
                        nid = TILE_MAP[d]
                        nx = 0
                        if nid < 27:
                            if nid % 9 == 8: nx = nid - 8
                            else: nx = nid + 1
                        elif nid < 31:
                            if nid == 30: nx = 27
                            else: nx = nid + 1
                        else:
                            if nid == 33: nx = 31
                            else: nx = nid + 1
                        doras_ids.append(nx)

                # Setup Common Variables
                tsumogiri_dummy = [[False]*len(d) for d in discards_hist]
                tensor_now_cpu = encode_state_god(
                    hands_arr, discards_hist, tsumogiri_dummy,
                    melds_arr, scores, doras_ids, 
                    round_val, honba_val, riichi_sticks, [False]*4, 3, full_hands_ids_dummy
                )
                base_tensor_gpu = torch.from_numpy(tensor_now_cpu).unsqueeze(0).to(DEVICE).half()
                my_hand_counts = np.array(hands_arr[3])

                # DEBUG: Check hand size
                # DEBUG: Check hand size
                hand_sum = sum(hands_arr[3])
                
                # ★Naki Mode (Call Check)
                if hand_sum % 3 == 1:
                     print(f"👀 Naki Check Mode (Tiles={hand_sum})")
                     # Identify Target Tile (Last discard from Opponents)
                     # Search L->T->R (Kami->Toimen->Shimo) priority?
                     # No, we check ALL recently active rivers.
                     # Simplified: Check last tile of Left(2), Top(1), Right(0)
                     # Priority: Left (allows Chi/Pon), others (Pon only)
                     
                     naki_advice = None
                     current_turn_approx = len(discards_hist[3])
                     
                     # 1. Check Left (offset 3)
                     if discards_hist[2]:
                         target = discards_hist[2][-1]
                         act = NakiUtils.check_calls(hands_arr[3], target, 3, doras_ids, current_turn_approx)
                         if act: naki_advice = f"[上家] {act}"
                     
                     # 2. Check Top (offset 2)
                     if not naki_advice and discards_hist[1]:
                         target = discards_hist[1][-1]
                         act = NakiUtils.check_calls(hands_arr[3], target, 2, doras_ids, current_turn_approx)
                         if act: naki_advice = f"[対面] {act}"
                         
                     # 3. Check Right (offset 1)
                     if not naki_advice and discards_hist[0]:
                         target = discards_hist[0][-1]
                         act = NakiUtils.check_calls(hands_arr[3], target, 1, doras_ids, current_turn_approx)
                         if act: naki_advice = f"[下家] {act}"
                         
                     if naki_advice:
                         print(f"💡 Naki Advice: {naki_advice}")
                         riichi_advice = naki_advice # Override Riichi advice field for Naki
                
                if hand_sum != 14 and hand_sum % 3 != 1:
                     print(f"⚠️ WARNING: Hand size is {hand_sum} (Expected 14 or 13). Logic might fail.")
                
                t_parse = time.time()
                    
                # (1) Intuition Phase
                with torch.no_grad():
                    logits, naki_now = brain_model(base_tensor_gpu)
                
                t_infer = time.time()
                
                # Reach Logic
                riichi_prob = 0.0
                if naki_now is not None:
                    riichi_prob = torch.softmax(naki_now[0], 0)[4].item()
                    if naki_now[0].argmax().item() == 4: riichi_advice = f"⚠️リーチ推奨 ({riichi_prob:.1%})"
                    elif riichi_prob > 0.20: riichi_advice = f"🤔リーチ考慮 ({riichi_prob:.1%})"

                mask = torch.full((35,), -float('inf'), device=DEVICE)
                for i in range(34):
                    if hands_arr[3][i] > 0: mask[i] = 0
                mask[34] = 0
                
                probs = torch.softmax(logits[0] + mask, 0)
                topk_vals, topk_indices = torch.topk(probs, 5)
                
                candidates = []
                for i in range(5):
                    idx = topk_indices[i].item()
                    val = float(topk_vals[i].item())
                    if val < 0.0001: continue
                    label = "ツモ切り" if idx == 34 else INV_TILE_MAP[idx]
                    candidates.append({"label": label, "idx": idx, "base_conf": val})

                # DEBUG: Check what the AI sees as the hand
                print(f"DEBUG: AI Hand Array = {hands_arr[3]}")

                # ★Logic Guard: 孤立字牌・孤立一九牌の救済 (Rescue Isolated Honors/Terminals)
                # モデルの自信がない場合(Top<20%)や、モデルが見落とした場合に備え、孤立牌を候補に加える
                # これによりUkeire/Valueロジックが正しく評価できるようになる
                existing_ind = {c["idx"] for c in candidates}
                
                # Check Honors (27..33)
                for t_id in range(27, 34):
                    if hands_arr[3][t_id] == 1: # Isolated Honor
                        if t_id not in existing_ind:
                            label = INV_TILE_MAP[t_id]
                            candidates.append({"label": label, "idx": t_id, "base_conf": 0.05}) # Low conf, trusted to logic
                            existing_ind.add(t_id)
                
                # Check Terminals (0, 8, 9, 17, 18, 26)
                for t_id in [0, 8, 9, 17, 18, 26]:
                    if hands_arr[3][t_id] == 1: # Isolated Terminal
                        # Check neighbors to ensure truly isolated (simple check)
                        is_iso = True
                        if t_id % 9 == 0 and hands_arr[3][t_id+1] > 0: is_iso = False # 1 -> 2
                        if t_id % 9 == 8 and hands_arr[3][t_id-1] > 0: is_iso = False # 9 -> 8
                        
                        if is_iso and t_id not in existing_ind:
                            label = INV_TILE_MAP[t_id]
                            candidates.append({"label": label, "idx": t_id, "base_conf": 0.05})
                            existing_ind.add(t_id)

                # ★Logic Upgrade: 候補拡張 (Logical Candidate Expansion)
                # モデルが見落とした「シャンテン数が下がる打牌」があれば強制的に候補に入れる
                current_shanten = ShantenUtils.calculate_shanten(hands_arr[3])
                existing_candidate_indices = {c["idx"] for c in candidates}
                
                # 手持ちの牌全種をチェック
                for t_id in range(34):
                    if hands_arr[3][t_id] > 0:
                        # 試行: これを切ったらシャンテン下がる？
                        temp_h = hands_arr[3][:]
                        temp_h[t_id] -= 1
                        next_s = ShantenUtils.calculate_shanten(temp_h)
                        
                        if next_s < current_shanten:
                            if t_id not in existing_candidate_indices:
                                # 発見！モデルが見落とした良手
                                label = INV_TILE_MAP[t_id]
                                # base_confは低いが、後のLogic Boostで評価されるはず
                                candidates.append({"label": label, "idx": t_id, "base_conf": 0.05})
                                existing_candidate_indices.add(t_id)

                # (2) 思考フェーズ (Deep Thought Implementation)
                sim_targets = list(range(34))
                
                # ★修正: 聴牌ボーナスは元に戻す(微小値)
                # その代わり、シミュレーション精度(ROBUST_ITER)と未来予測のウェイト(w_future)を状況に応じて変化させる
                WAIT_BONUS_FACTOR = 0.10 
                
                # 自信度判定
                top_conf_val = candidates[0]['base_conf'] if candidates else 0.0
                
                # リーチ推奨が強い、または自信がある場合は高速モード
                if riichi_prob > 0.4 or top_conf_val > 0.6:
                    ROBUST_ITER = 10 # Optimized: 10 is enough for High Conf
                    w_base = 0.70; w_future = 0.30
                else:
                    # 自信がない(迷っている)場合は「Deep Thought」モード
                    # 50は重すぎたので30に調整 (Batch Size削減)
                    ROBUST_ITER = 30 
                    w_base = 0.30; w_future = 0.70

                sim_tasks = []

                # ... (Simulation Loop) ...
                sim_tasks = []
                for c_i, cand in enumerate(candidates):
                    if cand["idx"] == 34: continue
                    for next_draw in sim_targets:
                        # まだ持てる(count<4)、または今切る牌(cand["idx"])と同じ牌を引く(擬似的に戻ってくる)場合
                        if my_hand_counts[next_draw] < 4 or (cand["idx"] == next_draw):
                            sim_tasks.append((c_i, cand["idx"], next_draw))

                if sim_tasks:
                    num_sims = len(sim_tasks)
                    sim_tensor = base_tensor_gpu.repeat(num_sims, 1, 1, 1)
                    
                    batch_indices = torch.arange(num_sims, device=DEVICE)
                    discards = torch.tensor([x[1] for x in sim_tasks], device=DEVICE)
                    draws = torch.tensor([x[2] for x in sim_tasks], device=DEVICE)
                    curr_counts = torch.tensor(my_hand_counts, device=DEVICE)
                    
                    target_ch_rem = curr_counts[discards] - 1
                    valid_rem = target_ch_rem >= 0
                    sim_tensor[batch_indices[valid_rem], target_ch_rem[valid_rem], discards[valid_rem], 0] = 0

                    counts_after_discard = curr_counts[draws]
                    is_same = (discards == draws)
                    counts_after_discard[is_same] -= 1
                    target_ch_add = counts_after_discard 
                    valid_add = target_ch_add < 4
                    sim_tensor[batch_indices[valid_add], target_ch_add[valid_add], draws[valid_add], 0] = 1

                    final_input = sim_tensor.repeat_interleave(ROBUST_ITER, dim=0)
                    final_input += torch.randn_like(final_input) * 0.005 

                    with torch.no_grad():
                        logits_huge, _ = brain_model(final_input)
                        probs_huge = torch.max(torch.softmax(logits_huge, dim=1), dim=1)[0]
                        avg_potentials = torch.mean(probs_huge.view(num_sims, ROBUST_ITER), dim=1).cpu().numpy()

                    cand_totals = collections.defaultdict(float)
                    cand_counts = collections.defaultdict(int)
                    for i, (c_i, _, _) in enumerate(sim_tasks):
                        cand_totals[c_i] += float(avg_potentials[i])
                        cand_counts[c_i] += 1
                    
                    final_results = []
                    for c_i, cand in enumerate(candidates):
                        if cand["idx"] == 34:
                            # ツモ切りもスコア計算対象（ベーススコアそのまま）
                            cand["score"] = cand["base_conf"]
                            final_results.append(cand)
                            continue
                        
                        avg_future = cand_totals[c_i] / cand_counts[c_i] if cand_counts[c_i] > 0 else cand["base_conf"]
                        sim_hands = [row[:] for row in hands_arr]
                        sim_hands[3][cand["idx"]] -= 1
                        wait_count = AgariUtils.get_waits_count(sim_hands[3], [0]*34) 
                        bonus = math.log(1 + wait_count) * 0.08 if wait_count > 0 else 0
                        
                        # 重み付け反映
                        score = float((cand["base_conf"] * w_base) + (avg_future * w_future) + bonus)
                        cand["score"] = score
                        final_results.append(cand)
                    
                    final_results.sort(key=lambda x: x.get("score", x["base_conf"]), reverse=True)
                    candidates = final_results

                    # (2.5) Rank Strategy Calculation
                    rank_strategy = RankManager.get_strategy(scores, round_val, honba_val, 3, 0)
                    
                    t_deep = time.time()
                    
                    # (2.6) Parallel PIMC Execution (World Class Logic)
                    try:
                        # PIMC用のVisible Counts作成
                        pimc_visible = [0]*34
                        # 捨て牌
                        for i in range(4):
                            for t in discards_hist[i]: pimc_visible[t] += 1
                            for m_list in melds_arr[i]:
                                for t in m_list: pimc_visible[t] += 1
                        # ドラ表示牌
                        for d in indicators:
                            if d in TILE_MAP: pimc_visible[TILE_MAP[d]] += 1
                        # 自分の手牌
                        for t in hands_arr[3]: pimc_visible[t] += 1
                        
                        # PIMC実行 (300並列世界)
                        # ProcessPoolで高速並列化: 300世界でも 0.5~1.0s 程度で完了するはず
                        s_time = time.time()
                        
                        # 特殊候補(idx>=34)除外およびRiichi判定
                        pimc_candidates = []
                        for c in candidates:
                            if c["idx"] >= 34: continue
                            
                            # 聴牌チェック: この牌を切ってテンパイなら「リーチ宣言」としてシミュレーションする
                            # (本来はダマ/リーチ分岐すべきだが、積極性重視でリーチ扱いにする)
                            temp_h = hands_arr[3][:]
                            temp_h[c["idx"]] -= 1
                            if ShantenUtils.calculate_shanten(temp_h) == 0:
                                c["is_riichi"] = True
                            else:
                                c["is_riichi"] = False
                                
                            pimc_candidates.append(c)
                            
                        skipped_candidates = [c for c in candidates if c["idx"] >= 34]
                        
                        if pimc_candidates:
                            # 5000 -> 3000 (Sweet Spot for i5-14500)
                            # 3000 sims gives 99% of the gain of 5000 but 40% faster
                            pimc_results = PIMCEngine.run(hands_arr, pimc_visible, pimc_candidates, doras_ids, num_worlds=3000)
                            # 結果をマージして再ソート
                            candidates = pimc_results + skipped_candidates
                            candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
                        
                        t_pimc_end = time.time()
                        print(f"DEBUG: PIMC Time = {t_pimc_end - s_time:.3f}s")
                        print(f"DEBUG: AI Breakdown >> Parse: {t_parse - t_s:.3f}s | Infer: {t_infer - t_parse:.3f}s | DeepThought: {t_deep - t_infer:.3f}s | PIMC: {t_pimc_end - s_time:.3f}s")
                    except Exception as e:
                        print(f"PIMC Error: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # (3) EV-based Decision Logic (World Class Logic)
                    # ----------------------------------------------------------------
                    # シミュレーション結果と「明示的な期待値計算」を統合する
                    
                    final_candidates = []
                    # riichi_candidate = None # Removed global tracking
                    
                    # 計算済みcandidatesを走査
                    for cand in candidates:
                        cand["riichi_advice"] = "" # Init advice
                        if cand["idx"] == 34: 
                            final_candidates.append(cand)
                            continue

                        # 現在のsim_scoreを取得
                        sim_score = cand.get("score", cand["base_conf"])
                        
                        # Rank Logic: Speed/Value Boost
                        # 受け入れ枚数や打点予測は後で厳密に計算するが、ここでは簡易的なBoost
                        # ※厳密な計算は各分岐内で行う
                        
                        # -----------------------------------------------
                        # 🧮 テンパイ・期待値(EV)計算
                        # -----------------------------------------------
                        # 打牌後の手牌を作成
                        temp_hand = hands_arr[3][:]
                        temp_hand[cand["idx"]] -= 1
                        
                        # 待ち枚数計算 (AgariUtils)
                        waits_count = AgariUtils.get_waits_count(temp_hand, [0]*34) # visible簡略化(0)
                        
                        ev_score = sim_score 
                        
                        # PIMC結果を加算 (World Class Logic Boost)
                        if "pimc_score" in cand:
                            ev_score += cand["pimc_score"] * 5.0
                        
                        if waits_count > 0:
                            # 聴牌している場合: 厳密なEV計算を行う
                            print(f"DEBUG: Candidate {cand['label']} is TENPAI. Waits: {waits_count}")
                            
                            # 1. アガリ確率 (Win Prob)
                            win_prob = min(waits_count / 70.0, 0.9)
                            
                            # 2. 推定打点 (Estimated Score via YakuUtils)
                            # ----------------------------------------------------------------
                            # Define helper to get score
                            # Need round_wind, self_wind
                            # round_val: 0(E1),1(E2)..4(S1)..
                            r_wind = 0 if round_val < 4 else 1 # 0:East, 1:South (Simplified)
                            # Actually YakuUtils expects 0-3 int. 
                            # If round_val 0-3 -> East(0). round_val 4-7 -> South(1).
                            
                            # HUD Info "Self_Wind_Tile" -> 'E'/'S'/'W'/'N' or '東(East)' etc.
                            # Updated map to handle all formats
                            s_wind_map = {'1z':0, '2z':1, '3z':2, '4z':3,
                                          'E':0, 'S':1, 'W':2, 'N':3,
                                          '東(East)':0, '南(South)':1, '西(West)':2, '北(North)':3}
                            # Default to West(2) if unknown (common guest pos?) or North(3). safer to say West.
                            s_wind_str = hud_info.get("Self_Wind_Tile", "3z") 
                            if "East" in s_wind_str: s_wind_str = "E" # Normalization
                            elif "South" in s_wind_str: s_wind_str = "S"
                            elif "West" in s_wind_str: s_wind_str = "W"
                            elif "North" in s_wind_str: s_wind_str = "N"
                                          
                            s_wind = s_wind_map.get(s_wind_str, 2)
                            
                            # Calculate Han for Riichi
                            # Note: temp_hand is the hand assuming we Discarded 'cand'.
                            # Wait... No. 'temp_hand' is `hands_arr[3]` MINUS `cand`?
                            # decice_discard loop: for cand in candidates... 
                            # We need to assess the value of the HAND that remains.
                            # But Han is calculated on Agari.
                            # We are Tenpai. We need "Predicted Han".
                            # YakuUtils calculates current shape. 
                            # If we are Tenpai, the hand + RonTile = Agari.
                            # We should test with a dummy Ron Tile (one of our waits).
                            # Ideally average over all waits? 
                            # For speed, pick the first wait or best wait.
                            
                            predicted_han_riichi = 0
                            predicted_han_dama = 0
                            
                            # Check waits
                            waits_list = [] # Need actual waits to test Yaku
                            # We have waits_count but not list? 
                            # AgariUtils.get_waits_count returns count only.
                            # We need to know WHAT we are waiting for to call YakuUtils.
                            # Re-calc waits? (Fast enough for 1 cand? maybe)
                            # Or just assume "If I win, I have these Yaku".
                            # YakuUtils can check "Hand + DummyTile".
                            
                            # Let's quickly find 1 valid wait tile
                            dummy_ron_tile = -1
                            temp_h_chk = temp_hand.copy() # Temp hand after discard
                            for t_id in range(34):
                                if temp_h_chk[t_id] < 4:
                                    temp_h_chk[t_id] += 1
                                    if AgariUtils.is_agari(temp_h_chk):
                                        dummy_ron_tile = t_id
                                        break
                                    temp_h_chk[t_id] -= 1
                            
                            if dummy_ron_tile != -1:
                                temp_h_chk = temp_hand.copy()
                                temp_h_chk[dummy_ron_tile] += 1
                                
                                # Riichi Han
                                # melds=[] (assuming Menzen for Riichi decision mostly, unless called)
                                # Actually we should use actual melds.
                                # melds_objs is visual format. We need Logic format?
                                # StateManager has melds? Or `melds_objs`.
                                # Conversion needed if we want strict check.
                                # For now assume Menzen if state_manager says so.
                                # If we have melds, we can't Riichi anyway!
                                # Logic: If not Menzen, Riichi prob is 0 from model usually.
                                # But if Model says Riichi (weird), we should block it.
                                
                                # Strict Menzen Check
                                is_menzen_hand = state_manager.is_menzen(3)
                                
                                if is_menzen_hand:
                                    # Calc Dama
                                    han_d, yaku_d = YakuUtils.get_hand_han(
                                        temp_h_chk, [], dummy_ron_tile, 
                                        is_riichi=False, is_tsumo=False, 
                                        round_wind=r_wind, self_wind=s_wind, 
                                        doras_count=0 # Add separately
                                    )
                                    predicted_han_dama = han_d
                                    
                                    # Calc Riichi (Add 1 Han for Riichi)
                                    han_r, yaku_r = YakuUtils.get_hand_han(
                                        temp_h_chk, [], dummy_ron_tile, 
                                        is_riichi=True, is_tsumo=False,
                                        round_wind=r_wind, self_wind=s_wind,
                                        doras_count=0
                                    )
                                    predicted_han_riichi = han_r
                                else:
                                    # Not Menzen -> Can't Riichi.
                                    # Dama only (check Yaku for valid ron)
                                    # Need to parse melds?
                                    # Simplified: Assume 1 Han if called (Tanyao/Yakuhai likely)
                                    # or 0 if Bad.
                                    predicted_han_riichi = 0
                                    predicted_han_dama = 1 # Fallback
                            
                            
                            # Dora Count
                            dora_cnt = 0
                            for d_id in doras_ids:
                                 dora_cnt += temp_hand[d_id]
                            # +1 for logic (reds etc not fully tracked, approx)
                            
                            # Score Conversion (Approximation)
                            # 1 Han = 1000, 2=2000, 3=3900, 4=7700, 5=8000...
                            def han_to_score(h, d):
                                total = h + d
                                if total <= 0: return 0
                                if total == 1: return 1000
                                if total == 2: return 2000
                                if total == 3: return 3900
                                if total == 4: return 7700
                                if total >= 5: return 8000 + (total-5)*1000 # Rough Mangan+
                                return 1000
                            
                            score_dama = han_to_score(predicted_han_dama, dora_cnt)
                            score_riichi = han_to_score(predicted_han_riichi, dora_cnt)
                            
                            # If No Yaku Dama -> Score is 0
                            # YakuUtils returns 0 if no yaku names.
                            if predicted_han_dama == 0: score_dama = 0
                            
                            # 3. EV Calculation (Strict)
                            # ----------------------------------------------------------------
                            win_prob_dama = min(waits_count / 80.0, 0.8) # Base win rate
                            win_prob_riichi = min(waits_count / 90.0, 0.7) # Lower win rate due to defense
                            
                            # Riichi Bonus: Uradora/Ippatsu chance -> Avg Score Boost +1500?
                            # Also Score Riichi is just Base.
                            # If we Riichi, we expect higher average score.
                            score_riichi_expected = score_riichi + 1500
                            
                            # Risk
                            # Riichi Risk: 1000 stick + Increased Deal-in
                            # Dama Risk: Low deal-in
                            risk_cost_riichi = 1000 + (2000 * 0.15) # Stick + 15% deal-in change
                            risk_cost_dama   = (2000 * 0.05)
                            
                            ev_riichi = (win_prob_riichi * score_riichi_expected) - risk_cost_riichi
                            ev_dama = (win_prob_dama * score_dama) - risk_cost_dama
                            
                            ev_diff = ev_riichi - ev_dama
                            
                            # 4. Decision
                            # ----------------------------------------------------------------
                            
                            advice_msg = ""
                            force_riichi = rank_strategy["must_riichi"]
                            
                            # Threshold: Positive EV or Strategy Force
                            # If Dama Score is 0 (No Yaku), ev_dama is negative (risk only).
                            # ev_diff will be huge (Positive - Negative). -> Riichi recommended naturally.
                            
                            if (ev_diff > 2000) or (score_dama == 0 and score_riichi > 0):
                                is_riichi_better = True
                                advice_msg = f"🚀強リーチ (EV差:{int(ev_diff)})"
                            elif ev_diff > 300:
                                is_riichi_better = True
                                advice_msg = f"⚠️リーチ有利 (EV差:{int(ev_diff)})"
                            elif force_riichi:
                                is_riichi_better = True
                                advice_msg = f"🔥攻めリーチ (条件重視)"
                            else:
                                is_riichi_better = False
                                if score_dama == 0:
                                    advice_msg = "🛑ダマ不可(役無) -> 降りor崩し?"
                                    # But if we are here we are Tenpai.
                                    # If not Riichi and No Yaku -> We are effectively Folding.
                                    # But current logic is for "Discard Candidate".
                                    # If we discard this, we are Tenpai.
                                    # If we don't Riichi, we have no Yaku.
                                    # Should we advise Riichi anyway?
                                    # If EV(Riichi) is bad too (e.g. -500), and EV(Dama) is -100.
                                    # Then Dama (Fold) is better.
                                    # But user often wants to win.
                                    if ev_riichi > -200: # If Riichi is not Terrible
                                         is_riichi_better = True
                                         advice_msg = f"🚀役無リーチ (Dama=0)"
                                else:
                                    advice_msg = f"🤫ダマ (Wait:{waits_count}, {predicted_han_dama}翻)"

                            cand["riichi_advice"] = advice_msg
                            if is_riichi_better:
                                ev_score += 1.5
                                print(f"DEBUG: {advice_msg} HanR:{predicted_han_riichi} HanD:{predicted_han_dama}")
                            else:
                                ev_score += 0.5
                                
                            cand["final_score"] = ev_score
                            final_candidates.append(cand)                            

                    
                    if final_candidates:
                        final_candidates.sort(key=lambda x: x.get("final_score", 0), reverse=True)
                        candidates = final_candidates
                
                # (4) Shanten-based Logic (Backtracking Check & Menzen Verify)
                # ----------------------------------------------------------------
                # AIが「シャンテン数を戻す打牌」を選ばないように監視する
                # 11億局面の学習データがあるとはいえ、局所的な見落としを防ぐための論理ガード
                
                if candidates:
                    current_shanten = ShantenUtils.calculate_shanten(hands_arr[3])
                    
                    # 敵のリーチ有無を確認 (自分以外)
                    enemy_has_riichi = any(riichi_sticks[i] for i in range(4) if i != 3)

                    for cand in candidates:
                        # Initialize final_score safely
                        cand["final_score"] = cand.get("final_score", cand["base_conf"])

                    for cand in candidates:
                        # ツモ切りはスルー（現状維持）
                        if cand["idx"] == 34: continue
                        
                        # 打牌後のシャンテン数を計算
                        temp_hand = hands_arr[3][:]
                        temp_hand[cand["idx"]] -= 1
                        next_shanten = ShantenUtils.calculate_shanten(temp_hand)
                        
                        # シャンテン数が悪化している場合 (例: 1シャンテン -> 2シャンテン)
                        # 戻る動きは原則禁止だが、以下の例外を除く:
                        # 1. 敵がリーチしている場合 (ベタオリ許容) -> ペナルティなし
                        # 2. 圧倒的な自信(0.9以上)がある (手を崩してでもドラを引きに行く等)
                        if next_shanten > current_shanten:
                             is_emergency = enemy_has_riichi
                             if not is_emergency and cand["base_conf"] < 0.9:
                                 cand["final_score"] = cand.get("final_score", cand["base_conf"]) - 10.0 # 強力なペナルティ
                                 cand["note"] = "Backtracking Penalized"

                        # (4.1) リーチ推奨時の門前チェック
                        # AIがリーチを推奨していても、鳴いていたらダマにする
                        if "リーチ" in cand.get("label", "") or "Riichi" in cand.get("label", ""):
                             if len(melds_arr[3]) > 0: # Check if Self has melds (simplest Menzen check)
                                 cand["final_score"] = cand.get("final_score", cand["base_conf"]) - 5.0
                                 cand["label"] = cand["label"].replace("リーチ", "ダマ(鳴き)")

                    # 再度ソート            
                    candidates.sort(key=lambda x: x.get("final_score", x["base_conf"]), reverse=True)

                # (5) Speed vs Value Logic (Ukeire & Estimator)
                # ----------------------------------------------------------------
                # 聴牌以前の段階で、「受け入れ最大化」と「打点重視」のバランスを取る
                
                if candidates and candidates[0]["idx"] != 34:
                     # Top 3候補のみ詳細計算 (計算コスト削減)
                     top_cands = candidates[:3]
                     current_shanten = ShantenUtils.calculate_shanten(hands_arr[3])
                     
                     # ★Global Visible Tiles Construction (for Perfect Counting)
                     visible_tiles_34 = [0]*34
                     # 1. 自分の手牌
                     for i in range(34): visible_tiles_34[i] += hands_arr[3][i]
                     # 2. 全員の捨て牌
                     for p in range(4):
                         for t in discards_hist[p]:
                             visible_tiles_34[t] += 1
                     # 3. 全員の副露
                     for p in range(4):
                         for m in melds_arr[p]:
                             for t in m:
                                 visible_tiles_34[t] += 1
                     # 4. ドラ表示牌
                     for d in doras_ids:
                         visible_tiles_34[d] += 1
                     # Cap at 4
                     for i in range(34):
                         if visible_tiles_34[i] > 4: visible_tiles_34[i] = 4

                     # 敵リーチ中なら無視（守備優先済み）
                     enemy_has_riichi = any(riichi_sticks[i] for i in range(4) if i != 3)
                     
                     # Extract Riichi Declaration Indices for Post-Riichi Safety (Awase-uchi)
                     # opps is available in this scope? Yes, state_manager.update(opps, ...) was called.
                     # But opps is passed to analyze.
                     riichi_indices = [-1]*4
                     for p in range(4):
                         if riichi_sticks[p]:
                             # Find declaration index from opps
                             # opps structure: opps[p]['river_data'] is list of dicts.
                             # dict has 'is_declaration' key.
                             # opps variable name in analyze function is local?
                             # analyze signature: analyze(img_path='..')
                             # In analyze, opps is reconstructed. 
                             if 'opps' in locals():
                                 r_data = opps[p].get('river_data', [])
                                 for i, tile_d in enumerate(r_data):
                                     if tile_d.get('is_declaration', False):
                                         riichi_indices[p] = i
                                         break
                     
                     if not enemy_has_riichi:
                          # 基準となる候補（暫定1位）
                          # 比較ロジックを廃止し、絶対評価(Absolute Scoring)に移行
                          # これにより、Deep Learningの自信が低い場合(Tsumogiri bias)でも、
                          # ロジック的に正しい打牌が必ず選ばれるようにする。
                          
                          for cand in top_cands:
                              if cand["idx"] == 34: continue
                              
                              # 打牌後の手牌
                              temp_hand = hands_arr[3][:]
                              temp_hand[cand["idx"]] -= 1
                              
                              # 1. 受け入れ枚数 (Efficiency)
                              next_shanten = ShantenUtils.calculate_shanten(temp_hand)
                              c_ukeire, _ = UkeireUtils.get_ukeire_count(temp_hand, next_shanten, visible_tiles_34)
                              cand["ukeire"] = c_ukeire
                              
                              # 2. 打点ポテンシャル (Value)
                              est_val = ValueEstimator.estimate_hand_value(temp_hand, doras_ids)
                              
                              # 3. 鳴きシミュレーション (Naki)
                              naki_score = NakiPlanner.analyze_naki_potential(temp_hand, doras_ids)
                              # Naki Bonus: 1.0 score ~= 1000 pts equivalent
                              naki_val = naki_score * 1000.0
                              
                              cand["est_value"] = est_val + naki_val
                              cand["note"] = cand.get("note", "") + f" Val:{est_val} Naki:{naki_score:.1f}"
                              
                              # 4. Final Score Calculation (Absolute)
                              # Brain Prob (0~1) mixed with Logic.
                              # User Request: Increase weight of trained model.
                              # Legacy Logic Score = 5.0 + (Val/1000) + (Ukeire/5).
                              # New: + (BaseConf * 3.0) -> If conf is 0.5, adds 1.5. If 0.1, adds 0.3.
                              # Tsumogiri only has BaseConf. It needs to lose unless valid moves are terrible.
                              
                              logic_score = 5.0 + (cand["est_value"] / 1000.0) + (c_ukeire / 5.0)
                              model_bonus = cand["base_conf"] * 2.5 # Weight 2.5
                              
                              cand["final_score"] = logic_score + model_bonus
                              cand["note"] = f"Logic:{logic_score:.1f} Brain:{model_bonus:.2f} " + cand["note"]

                          # No comparison loop needed. Sorting handles it.
                          candidates.sort(key=lambda x: x.get("final_score", x["base_conf"]), reverse=True)
                     
                     else:
                         # (6) Safety Logic (Full Scale Defense)
                         # 敵リーチ時は、危険度が高い牌を徹底的に避ける
                         current_riichi_status = [riichi_sticks[i] for i in range(4)]
                         current_turn = max([len(d) for d in discards_hist], default=1)
                         
                         for cand in top_cands:
                             if cand["idx"] == 34: continue
                             
                             # Full Scale Danger Score (inc. Suji, Kabe, Tenpai Prob, Dora, OneChance)
                             # Added riichi_indices for Post-Riichi Safety (Awase-uchi)
                             danger = SafetyUtils.get_danger_score(cand["idx"], discards_hist, current_riichi_status, visible_tiles_34, turn=current_turn, melds_arr=melds_arr, doras=doras_ids, riichi_indices=riichi_indices)
                             cand["danger"] = danger
                             
                             if danger == 0:
                                 cand["final_score"] = cand.get("final_score", cand["base_conf"]) + 3.0 
                                 cand["note"] = "Genbutsu/Post-R Safe (+3.0)"
                             elif danger <= 15: # Suji-level safety
                                 cand["final_score"] = cand.get("final_score", cand["base_conf"]) + 1.0
                                 cand["note"] = "Suji Safe (+1.0)" 
                             elif danger >= 50:
                                 cand["final_score"] = cand.get("final_score", cand["base_conf"]) - 3.0
                                 cand["note"] = "Danger High (-3.0)"
                                 
                         candidates.sort(key=lambda x: x.get("final_score", x["base_conf"]), reverse=True)

                if candidates:
                    best_cand = candidates[0]
                    best_move = best_cand["label"]
                    confidence = float(best_cand["base_conf"])
                    riichi_advice = best_cand.get("riichi_advice", "") # Extract from best
                    top_2_candidates = candidates[:2]

                # ---------------------------------------------------------
                # (3) 鳴き判断ロジック (統合・厳選版)
                # ---------------------------------------------------------
                # 🛡️ ルールガード: リーチ中または海底なら鳴き禁止
                is_riichi = riichi_sticks[3]
                try: tiles_left_num = int(hud_info.get("Info_TilesLeft", "10"))
                except: tiles_left_num = 10
                is_houtei = (tiles_left_num == 0)

                if not is_riichi and not is_houtei:
                    left_river_len = len(discards_hist[2])
                    left_row = min(left_river_len // 6, 3)
                    left_col_val = ((left_river_len % 6) + 1) / 6.0
                    left_river_ch_base = 12 + left_row 

                    my_hand_counts_sim = np.array(hands_arr[3]) # コピー
                    
                    # 鳴いた後に切る予定の牌
                    planned_discard = candidates[0]["idx"]
                    if planned_discard == 34:
                         # 34の場合は暫定的に手出しの最有力候補を使う
                         for c in candidates:
                            if c["idx"] != 34:
                                planned_discard = c["idx"]
                                break
                    
                    if planned_discard != 34 and my_hand_counts_sim[planned_discard] > 0:
                        my_hand_counts_sim[planned_discard] -= 1

                    naki_candidates = []
                    for t_id in range(34):
                        can_chi = False; can_pon = False; can_kan = False
                        
                        if my_hand_counts_sim[t_id] >= 2: can_pon = True
                        if my_hand_counts_sim[t_id] >= 3: can_kan = True
                        
                        # 喰い替え防止付きチー判定
                        if t_id < 27:
                            idx = t_id % 9
                            # 下チー (3,4 で 2 をチー)
                            if idx >= 2 and my_hand_counts_sim[t_id-2] > 0 and my_hand_counts_sim[t_id-1] > 0:
                                if planned_discard == t_id: pass
                                elif (idx <= 5) and (planned_discard == t_id + 3): pass
                                else: can_chi = True
                            # 嵌張チー (4,6 で 5 をチー)
                            if 1 <= idx <= 7 and my_hand_counts_sim[t_id-1] > 0 and my_hand_counts_sim[t_id+1] > 0:
                                if planned_discard == t_id: pass
                                else: can_chi = True
                            # 上チー (6,7 で 8 をチー)
                            if idx <= 6 and my_hand_counts_sim[t_id+1] > 0 and my_hand_counts_sim[t_id+2] > 0:
                                if planned_discard == t_id: pass
                                elif (idx >= 3) and (planned_discard == t_id - 3): pass
                                else: can_chi = True

                        if can_chi or can_pon:
                            naki_candidates.append((t_id, {'chi': can_chi, 'pon': can_pon, 'kan': can_kan}))

                    if naki_candidates:
                        num_naki = len(naki_candidates)
                        naki_base_tensor = base_tensor_gpu.clone()
                        
                        # 自分のターンの切り出し情報を消す（鳴き想定は相手ターン直後なので）
                        # ※厳密には「直前に誰かが切った」状態を作るべきだが、簡易的に自分の予定打牌を打ち消す
                        if best_move != "Wait" and candidates[0]["idx"] != 34:
                             d_idx = candidates[0]["idx"]
                             # 自分の手牌チャネルを戻す操作は複雑なので、
                             # ここでは「手牌から減らす前のベーステンソル」を使っているためOK
                             pass

                        naki_tensor = naki_base_tensor.repeat(num_naki, 1, 1, 1)
                        batch_indices = torch.arange(num_naki, device=DEVICE)
                        target_tiles = torch.tensor([x[0] for x in naki_candidates], device=DEVICE)
                        naki_tensor[batch_indices, left_river_ch_base, target_tiles, 0] = left_col_val

                        # ロバスト化: ノイズを加えて複数回推論
                        S_DENSITY = 64
                        naki_input = naki_tensor.repeat_interleave(S_DENSITY, dim=0)
                        naki_input += torch.randn_like(naki_input) * 0.005

                        with torch.no_grad():
                            L_D, L_C = brain_model(naki_input) # L_D: Discard Logits (35), L_C: Naki Logits (5)
                            C_PROBS = torch.softmax(L_C, dim=1)
                            D_PROBS = torch.softmax(L_D, dim=1) # Discard Probability

                        # Reshape to average across robust samples
                        probs_reshaped = C_PROBS.view(num_naki, S_DENSITY, 5)
                        avg_probs = torch.mean(probs_reshaped, dim=1)
                        
                        d_probs_reshaped = D_PROBS.view(num_naki, S_DENSITY, 35)
                        avg_d_probs = torch.mean(d_probs_reshaped, dim=1)
                        
                        temp_results_list = []
                        for i, (t_id, legal) in enumerate(naki_candidates):
                            probs = avg_probs[i]
                            d_probs = avg_d_probs[i]
                            
                            p_pass = float(probs[0].item())
                            best_action = "Pass"; best_score = p_pass
                            
                            if legal['chi'] and probs[1].item() > best_score: best_action="Chi"; best_score=float(probs[1].item())
                            if legal['pon'] and probs[2].item() > best_score: best_action="Pon"; best_score=float(probs[2].item())
                            if legal['kan'] and probs[3].item() > best_score: best_action="Kan"; best_score=float(probs[3].item())
                            
                            if best_action != "Pass":
                                # -------------------------------------------------
                                # 🛡️ 推論後の打牌チェック (正確な食い替え判定)
                                # -------------------------------------------------
                                
                                # 1. 鳴きで消費する牌を特定して、手牌から減らす
                                # (これをしないと、チーした牌をそのまま切る等の不正打牌を予測してしまう)
                                consumed = []
                                if best_action == "Pon":
                                    consumed = [t_id, t_id]
                                elif best_action == "Kan":
                                    consumed = [t_id, t_id, t_id]
                                elif best_action == "Chi":
                                    # 簡易: 両面優先で消費牌を特定
                                    base_idx = t_id % 9
                                    # my_hand_counts_sim は既に planned_discard 分が引かれている可能性があるが、
                                    # uniqueな牌なら0か1。
                                    # ここでは my_hand_counts_sim を使う
                                    
                                    # 判定用に一時的に+100とかしないとダメか？
                                    # いや、my_hand_counts_simは「自分の手番が来る前」の状態。
                                    # つまり t_id は他家からの牌。手牌には含めない。
                                    # 消費するのは手牌にある牌。
                                    
                                    h_chk = my_hand_counts_sim
                                    if base_idx >= 2 and h_chk[t_id-2] > 0 and h_chk[t_id-1] > 0: consumed = [t_id-2, t_id-1]
                                    elif 1 <= base_idx <= 7 and h_chk[t_id-1] > 0 and h_chk[t_id+1] > 0: consumed = [t_id-1, t_id+1]
                                    elif base_idx <= 6 and h_chk[t_id+1] > 0 and h_chk[t_id+2] > 0: consumed = [t_id+1, t_id+2]

                                # 手牌コピーを作成して減算反映
                                current_hand_after_naki = my_hand_counts_sim.copy()
                                valid_consumption = True
                                for c in consumed:
                                    if current_hand_after_naki[c] > 0:
                                        current_hand_after_naki[c] -= 1
                                    else:
                                        valid_consumption = False
                                        
                                if not valid_consumption:
                                    continue # 手牌が足りない(あり得ないはずだが)

                                # 2. マスク作成: 手牌にない牌は切れない
                                # hand_for_naki = current_hand_after_naki
                                
                                # ツモ切り(34)は、鳴き直後にはあり得ない(手出しのみ)
                                d_probs_valid = d_probs[:34].clone()
                                
                                # 手牌に存在しない牌の確率は0にする
                                for ti in range(34):
                                    if current_hand_after_naki[ti] == 0:
                                        d_probs_valid[ti] = -1.0
                                
                                best_discard_idx = torch.argmax(d_probs_valid).item()
                                
                                # 食い替えチェック (Post-Hoc Check)
                                is_kuikae = False
                                
                                # ★復元: 「同牌チー」の禁止
                                if best_action == "Chi" and my_hand_counts_sim[t_id] > 0:
                                    is_kuikae = True

                                # ★修正: 厳格な食い替え判定 (Diff 0, 3 を禁止)
                                if best_action == "Chi" and t_id < 27:
                                    suit_t = t_id // 9
                                    suit_d = best_discard_idx // 9
                                    
                                    if suit_t == suit_d:
                                        idx_t = t_id % 9
                                        idx_d = best_discard_idx % 9
                                        diff = abs(idx_t - idx_d)
                                        # Diff 0: 現物食い替え (例: 5チーして5切り) -> 違法
                                        # Diff 3: 筋食い替え (例: 1チーして4切り) -> 違法
                                        if diff in [0, 3]:
                                            is_kuikae = True

                                elif best_action == "Pon":
                                    if best_discard_idx == t_id: is_kuikae = True 

                                if is_kuikae:
                                    continue # この候補は却下
                                
                                # ----------------------------------------------------------------
                                # 🛡️ Shanten Guard (シャンテン数悪化の禁止)
                                # ----------------------------------------------------------------
                                # 「テンパイしているのにチーしてシャンテン数を落とす」等の悪手を防ぐ
                                # 現在の最善メンゼン進行でのシャンテン数を計算
                                if 'best_menzen_shanten' not in locals():
                                    # メンゼンでの最善手（candidates[0]）を打った後のシャンテン数
                                    best_discard_cand = candidates[0]
                                    temp_h = hands_arr[3][:]
                                    if best_discard_cand["idx"] != 34:
                                         temp_h[best_discard_cand["idx"]] -= 1
                                    
                                    # 13枚状態でのシャンテン数
                                    best_menzen_shanten = ShantenUtils.calculate_shanten(temp_h)

                                # 鳴いた後の手牌(13枚)を作成してシャンテン数を確認
                                # check_hand は current_hand_after_naki を使えば良い
                                check_hand = current_hand_after_naki.tolist() 
                                # current_hand_after_naki は ndarray なので list 化
                                # 既に consumed は引かれている
                                
                                # さらに打牌(best_discard_idx)を手牌から引く
                                temp_check = check_hand[:]

                                # 打牌
                                if temp_check[best_discard_idx] > 0:
                                    temp_check[best_discard_idx] -= 1
                                    
                                    naki_shanten = ShantenUtils.calculate_shanten(temp_check)
                                    
                                    # シャンテン数が悪化するなら却下
                                    # (例: テンパイ(0) -> 1シャンテン(1) はNG)
                                    if naki_shanten > best_menzen_shanten:
                                        continue
                                    
                                    # 既にテンパイ(0)なら、鳴いてもテンパイ(0)維持が必須で、かつ明確な利点が必要
                                    if best_menzen_shanten <= 0 and naki_shanten <= 0:
                                         # PIMCにお任せするが、足切りラインを高める
                                         if best_score < 0.6: continue
                                
                                # ----------------------------------------------------
                                # 🎲 PIMC Integration for Naki (Simulation Boost)
                                # ----------------------------------------------------
                                # 鳴いた後の手牌を作成して、シミュレーションを実行する
                                try:
                                    # ベースのスコアチェック (足切り緩和: 10%)
                                    diff_val = best_score - p_pass
                                    if best_score > 0.10: 
                                        
                                        # 1. 仮想手牌の作成
                                        pimc_hands = [row[:] for row in hands_arr]
                                        pimc_melds = [row[:] for row in melds_arr]
                                        
                                        # 手牌から鳴き素材を削除
                                        # best_actionに応じた牌を除く
                                        # (t_idは他家から出る牌なので、自分の手牌にはない。自分の手にある関連牌を抜く)
                                        consumed_tiles = []
                                        
                                        if best_action == "Pon":
                                            # t_id を2枚抜く
                                            consumed_tiles = [t_id, t_id]
                                        
                                        elif best_action == "Chi":
                                            # t_idと組み合わせて順子になる牌を探す
                                            # モデルは具体的な構成を出さないので、手持ちから推測する
                                            # 優先度: 両面 > 嵌張 > ペンチャン (受け入れが広い方を使うと仮定)
                                            # target: t_id. candidates in hand.
                                            base_idx = t_id % 9
                                            # 候補: (t-2, t-1), (t-1, t+1), (t+1, t+2)
                                            
                                            cands_mat = []
                                            # Lower (t-2, t-1)
                                            if base_idx >= 2 and pimc_hands[3][t_id-2] > 0 and pimc_hands[3][t_id-1] > 0:
                                                cands_mat.append(([t_id-2, t_id-1], 0)) # Type 0
                                            # Middle (t-1, t+1)
                                            if 1 <= base_idx <= 7 and pimc_hands[3][t_id-1] > 0 and pimc_hands[3][t_id+1] > 0:
                                                cands_mat.append(([t_id-1, t_id+1], 1)) # Type 1
                                            # Upper (t+1, t+2)
                                            if base_idx <= 6 and pimc_hands[3][t_id+1] > 0 and pimc_hands[3][t_id+2] > 0:
                                                cands_mat.append(([t_id+1, t_id+2], 0)) # Type 0
                                            
                                            if cands_mat:
                                                # 簡易的に0番目を使う（または評価の良い方を選ぶべきだが）
                                                consumed_tiles = cands_mat[0][0]
                                        
                                        elif best_action == "Kan":
                                            # t_id を3枚抜く
                                            consumed_tiles = [t_id, t_id, t_id]

                                        # 手牌反映
                                        valid_consumption = True
                                        for ct in consumed_tiles:
                                            if pimc_hands[3][ct] > 0:
                                                pimc_hands[3][ct] -= 1
                                            else:
                                                valid_consumption = False
                                        
                                        if valid_consumption:
                                            # 2. PIMC実行 (Naki State)
                                            # 鳴き状態での候補手（打牌）を作成
                                            # 既に best_discard_idx はあるが、PIMCで再評価させる
                                            
                                            # 手牌にあるもの全てが候補
                                            naki_sim_candidates = []
                                            for h_t in range(34):
                                                if pimc_hands[3][h_t] > 0: # 鳴いた後の手牌に残っているもの
                                                    c_label = INV_TILE_MAP[h_t]
                                                    naki_sim_candidates.append({"label": c_label, "idx": h_t, "base_conf": 0.5}) # Dummy conf
                                            
                                            if naki_sim_candidates:
                                                # Visible Counts更新 (鳴きで晒した分 + t_id)
                                                # t_idは川から拾うのでVisibleは変わらない(既に川にある前提)
                                                # しかしconsumed_tilesは手から場(Meld)に移動する
                                                
                                                sim_visible = [0]*34
                                                # Reuse logic to build visible
                                                for pp in range(4):
                                                    for tt in discards_hist[pp]: sim_visible[tt] += 1
                                                    for mm in melds_arr[pp]: 
                                                        for mmt in mm: sim_visible[mmt] += 1
                                                for d in doras_ids: sim_visible[d] += 1
                                                for tt in pimc_hands[3]: sim_visible[tt] += 1
                                                
                                                # 新しいMeld分を追加
                                                for ct in consumed_tiles: sim_visible[ct] += 1
                                                sim_visible[t_id] += 1
                                                
                                                # Run PIMC (10 worlds is enough for naki check)
                                                naki_sim_candidates = PIMCEngine.run(pimc_hands, sim_visible, naki_sim_candidates, doras_ids, num_worlds=15)
                                                
                                                # PIMCで最も良かった打牌のスコアを取得
                                                best_sim_cand = naki_sim_candidates[0]
                                                sim_score_ev = best_sim_cand.get("pimc_score", 0)
                                                
                                                # ベーススコアをPIMC結果で上書き/補正
                                                # PIMCスコア(EV)が高いなら、自信度をブースト
                                                if sim_score_ev > 0.5: # 基準点
                                                    best_score = max(best_score, 0.8) # 確信
                                                    best_discard_idx = best_sim_cand["idx"] # 打牌も最適なものに変更
                                        
                                    # ----------------------------------------------------
                                    
                                    # 最終閾値チェック (緩和)
                                    # PIMCで裏付けがあれば低確率でも通す
                                    if best_score > 0.15: # diff check removed to allow proactive calls
                                        t_base = INV_TILE_MAP.get(t_id, "?")
                                        t_name = t_base
                                        if t_base.endswith('z'):
                                            map_z = {'1z':'東','2z':'南','3z':'西','4z':'北','5z':'白','6z':'發','7z':'中'}
                                            t_name = map_z.get(t_base, t_base)
                                        
                                        # 捨てる牌の名称
                                        d_name_base = INV_TILE_MAP.get(best_discard_idx, "?")
                                        d_name = d_name_base
                                        if d_name_base.endswith('z'):
                                            map_z = {'1z':'東','2z':'南','3z':'西','4z':'北','5z':'白','6z':'發','7z':'中'}
                                            d_name = map_z.get(d_name_base, d_name_base)
                                        
                                        if len(d_name) >= 2:
                                            if d_name[1] == 'm': d_name = f"{d_name[0]}萬"
                                            elif d_name[1] == 's': d_name = f"{d_name[0]}草"
                                            elif d_name[1] == 'p': d_name = f"{d_name[0]}筒"

                                        icon = {"Chi":"✅チー", "Pon":"🚀ポン", "Kan":"🎇カン"}[best_action]
                                        
                                        # 表示用テキスト作成
                                        # 表示用テキスト作成
                                        temp_results_list.append((best_score, f"{icon}[{t_name}]➡切:{d_name}"))
                                except Exception as e:
                                    print(f"Naki Logic Error: {e}")
                                    continue

                        if temp_results_list:
                            temp_results_list.sort(key=lambda x: x[0], reverse=True)
                            top_picks = temp_results_list[:3]
                            prediction_text = " ".join([x[1] for x in top_picks])            
            except Exception as e:
                print(f"AI Error: {e}")
                import traceback
                traceback.print_exc()
                best_move = "Error"
        
        decision_debug = {
            "best_move": best_move,
            "confidence": f"{confidence:.1%}",
            "riichi": riichi_advice,
            "pred": prediction_text,
            "candidates": top_2_candidates 
        }
        
        return best_move, decision_debug
# ==========================================
# 👁️ Vision Logic (判定基準緩和版)
# ==========================================
# ==========================================
# 🀄 高精度アガリ判定・待ち枚数計算
# ==========================================
# -----------------------------------------------------------------------------
# 🛠️ Shanten Calculator (Logic Engine)
# -----------------------------------------------------------------------------
class RobustGrid:
    @staticmethod
    def letterbox_image(img, target_size=(1920, 1080)):
        h, w = img.shape[:2]; tw, th = target_size; scale = min(tw/w, th/h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        dx, dy = (tw - nw) // 2, (th - nh) // 2
        canvas[dy:dy+nh, dx:dx+nw] = resized
        return canvas

    @staticmethod
    def scan_winds(img_std, matcher, cache_dict):
        # Fast template matching only
        areas = {
            "Round_Wind": (922, 414, 78, 30),
            "Self_Wind_Tile": (825, 500, 50, 45)
        }
        results = {}
        for key, (ax, ay, aw, ah) in areas.items():
            crop_img = img_std[int(ay):int(ay+ah), int(ax):int(ax+aw)]
            
            cands = ['E', 'S', 'W', 'N'] 
            label, score = matcher.match(crop_img, threshold=0.55, candidates=cands if key=="Self_Wind_Tile" else None, return_best=True)
            val = label if label else "Unk"
            
            if val == 'E': val = '東(East)'
            elif val == 'S': val = '南(South)'
            elif val == 'W': val = '西(West)'
            elif val == 'N': val = '北(North)'
            
            if key == "Self_Wind_Tile":
                 cv2.imwrite("debug_wind.jpg", crop_img)
            
            results[key] = val
            cache_dict[key] = val
        return results

    @staticmethod
    def scan_scores(img_std, reader, cache_dict):
        # Slow OCR operations
        if reader is None: return {}
        
        areas = {
            "Info_TilesLeft": (948, 449, 50, 26),
            "Score_Top": (922, 380, 78, 30), "Score_Left": (877, 415, 30, 60),
            "Score_Right": (1019, 415, 30, 60), "Score_Self": (920, 470, 78, 50),
            "Info_Kyoutaku": (212, 230, 30, 35), "Info_Honba": (324, 230, 30, 35)
        }
        results = {}
        
        for key, (ax, ay, aw, ah) in areas.items():
            x1, y1 = int(ax), int(ay)
            x2, y2 = int(ax+aw), int(ay+ah)
            crop_img = img_std[y1:y2, x1:x2]
            if crop_img.size == 0: continue
            
            # Cache check could be here, but we pass full image, so let's just OCR
            # To optimize, we could check hash here too if we wanted strict sync
            
            if key == "Score_Top": crop_img = cv2.rotate(crop_img, cv2.ROTATE_180)
            elif key == "Score_Left": crop_img = cv2.rotate(crop_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif key == "Score_Right": crop_img = cv2.rotate(crop_img, cv2.ROTATE_90_CLOCKWISE)
            
            val = "0"
            try: 
                text_list = reader.readtext(crop_img, detail=0, allowlist='0123456789x,')
                text = "".join(text_list).replace(',', '')
                val = text if text else "0"
            except: pass
            
            results[key] = val
            # Update shared cache (Thread-safe dict assignment in Python is atomic-ish, sufficient here)
            cache_dict[key] = val
            
        return results

    @staticmethod
    def scan_hud_info(img_std, reader, matcher, cache_dict=None):
        # Wrapper for backward compatibility or sync calls
        if cache_dict is None: cache_dict = {}
        w_res = RobustGrid.scan_winds(img_std, matcher, cache_dict)
        s_res = RobustGrid.scan_scores(img_std, reader, cache_dict)
        w_res.update(s_res)
        
        # Debug boxes (reconstruct for display)
        debug_boxes = [] # simplified
        return w_res, debug_boxes

    @staticmethod
    # 引数を img_bgr から hsv_img に変更
    def is_back_side_tile(hsv_crop):
        if hsv_crop is None or hsv_crop.size == 0: return False
        
        # ▼▼▼ ここでの cvtColor を削除し、直接 hsv_crop を使う ▼▼▼
        # hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV) <-- 削除
        
        h, w = hsv_crop.shape[:2]
        crop_h, crop_w = int(h*0.4), int(w*0.4)
        cy, cx = h//2, w//2
        center_region = hsv_crop[cy-crop_h//2 : cy+crop_h//2, cx-crop_w//2 : cx+crop_w//2]
        
        if center_region.size == 0: return False
        
        s_channel = center_region[:, :, 1]
        v_channel = center_region[:, :, 2]
        
        # NumPy演算の最適化（count_nonzeroはそのまま）
        white_pixels = np.count_nonzero((s_channel < 40) & (v_channel > 130))
        white_ratio = white_pixels / (center_region.shape[0] * center_region.shape[1])
        
        if white_ratio > 0.25: return False 
        return True

    @staticmethod
    def sort_river_tiles(tiles, seat):
        if not tiles: return []
        avg_h = np.median([t['h'] for t in tiles]) if tiles else 0
        avg_w = np.median([t['w'] for t in tiles]) if tiles else 0
        
        def cluster_and_sort(data, primary_key, secondary_key, threshold, reverse_primary, reverse_secondary):
            data.sort(key=lambda x: x[primary_key], reverse=reverse_primary)
            clustered = []
            if not data: return []
            current_cluster = [data[0]]
            for i in range(1, len(data)):
                if abs(data[i][primary_key] - current_cluster[0][primary_key]) < threshold:
                    current_cluster.append(data[i])
                else:
                    current_cluster.sort(key=lambda x: x[secondary_key], reverse=reverse_secondary)
                    clustered.extend(current_cluster)
                    current_cluster = [data[i]]
            current_cluster.sort(key=lambda x: x[secondary_key], reverse=reverse_secondary)
            clustered.extend(current_cluster)
            return clustered

        if seat == 3: return cluster_and_sort(tiles, 'cy', 'cx', avg_h * 0.5, False, False)
        elif seat == 1: return cluster_and_sort(tiles, 'cy', 'cx', avg_h * 0.5, True, True)
        elif seat == 2: return cluster_and_sort(tiles, 'cx', 'cy', avg_w * 0.5, True, False)
        elif seat == 0: return cluster_and_sort(tiles, 'cx', 'cy', avg_w * 0.5, False, True)
        return tiles

    @staticmethod
    def split_hand_and_melds(tiles):
        if not tiles: return [], []
        tiles.sort(key=lambda t: t['cx'])
        if len(tiles) < 2: return tiles, []
        avg_w = np.median([t['w'] for t in tiles])
        split_idx = -1
        for i in range(len(tiles) - 1):
            curr_t = tiles[i]
            next_t = tiles[i+1]
            dist = next_t['cx'] - curr_t['cx']
            
      
            # これにより少しでも隙間があれば鳴きとして分離されます
            if curr_t['nx'] > 0.45 and dist > avg_w * 1.5:
                split_idx = i + 1
                break 
        if split_idx != -1:
            return tiles[:split_idx], tiles[split_idx:]
        else:
            return tiles, []

    @staticmethod
    def get_hand_by_count(bottom_tiles, img_original):
        if not bottom_tiles: return []
        bottom_tiles.sort(key=lambda t: t['cx'])
        tiles_reversed = list(reversed(bottom_tiles))
        final_list = []
        total_count = 0
        i = 0
        while i < len(tiles_reversed):
            if total_count >= 14: break
            current_tile = tiles_reversed[i]
            is_kan = False
            
            # カン（4枚組）の判定ロジック
            if i + 3 < len(tiles_reversed):
                group_of_4 = tiles_reversed[i : i+4]
                avg_w = np.mean([t['w'] for t in group_of_4])
                width_span = group_of_4[0]['cx'] - group_of_4[3]['cx'] 
                
                # 4枚が密集している場合
                if width_span < avg_w * 4.5:
                    back_tile_count = 0
                    for t in group_of_4:
                        iy1, iy2 = int(t['bbox'][1]), int(t['bbox'][3])
                        ix1, ix2 = int(t['bbox'][0]), int(t['bbox'][2])
                        
                        if iy1 < iy2 and ix1 < ix2:
                            t_img = img_original[iy1:iy2, ix1:ix2]
                            
                            # ▼▼▼ 修正箇所: 暗カン判定用にここだけHSV変換する ▼▼▼
                            # is_back_side_tile が HSV画像を要求する仕様に変わったため
                            if t_img.size > 0:
                                t_img_hsv = cv2.cvtColor(t_img, cv2.COLOR_BGR2HSV)
                                if RobustGrid.is_back_side_tile(t_img_hsv): 
                                    back_tile_count += 1
                            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

                    if back_tile_count > 0: is_kan = True
                    else:
                        labels = [t['label'] for t in group_of_4]
                        if len(set(labels)) == 1: is_kan = True
            
            if is_kan:
                kan_group = tiles_reversed[i : i+4]
                final_list.extend(kan_group)
                total_count += 3
                i += 4
            else:
                final_list.append(current_tile)
                total_count += 1
                i += 1
                
        final_list.sort(key=lambda t: t['cx'])
        return final_list

    @staticmethod
    def check_reach_sticks_by_color(img_std):
        centers = {1: (944, 367), 2: (848, 460), 0: (1078, 460), 3: (930, 533)}
        MARGIN = 5; REF_BGR = np.array([87, 75, 78], dtype=np.float32); DISTANCE_THRESHOLD = 50.0
        reach_status = {0:False, 1:False, 2:False, 3:False}
        debug_areas = []
        for seat, (cx, cy) in centers.items():
            x1, y1 = cx - MARGIN, cy - MARGIN; x2, y2 = cx + MARGIN, cy + MARGIN
            roi = img_std[y1:y2, x1:x2]
            if roi.size == 0: continue
            mean_bgr = np.mean(roi, axis=(0, 1))
            dist = np.linalg.norm(mean_bgr - REF_BGR)
            is_reach = bool(dist > DISTANCE_THRESHOLD)
            reach_status[seat] = is_reach
            debug_areas.append((x1, y1, x2, y2, is_reach, f"D:{dist:.0f}"))
        return reach_status, debug_areas

    @staticmethod
    def parse_frame_dual(original_img, boxes, names, img_w, img_h):
        # ▼▼▼ 高速化: 最初に画像全体をHSV変換 ▼▼▼
        hsv_full_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        raw_detections = []
        base_center_x = img_w * 0.5; base_center_y = img_h * 0.42; R = img_h * 0.35
        bottom_anchor_y = base_center_y + R - (img_h * 0.04)
        anchors = {1: (base_center_x, base_center_y - R), 2: (base_center_x - R, base_center_y),
                   0: (base_center_x + R, base_center_y), 3: (base_center_x, bottom_anchor_y)}
        
        for box in boxes:
            cls_id = int(box.cls[0]); label = names[cls_id]
            cx, cy, w, h = box.xywh[0].tolist()
            x1, y1 = max(0, int(cx - w/2)), max(0, int(cy - h/2))
            x2, y2 = min(img_w, int(cx + w/2)), min(img_h, int(cy + h/2))
            conf = float(box.conf[0]); nx, ny = cx/img_w, cy/img_h
            
            if cy < anchors[3][1]: 
                iy1, iy2 = int(y1), int(y2); ix1, ix2 = int(x1), int(x2)
                if iy1 < iy2 and ix1 < ix2 and w > 5 and h > 5:
                    # ▼▼▼ 高速化: HSV画像から切り抜いて判定 ▼▼▼
                    tile_hsv_crop = hsv_full_img[iy1:iy2, ix1:ix2]
                    if RobustGrid.is_back_side_tile(tile_hsv_crop): continue
                    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

            raw_detections.append({'label': label, 'cx': cx, 'cy': cy, 'w': w, 'h': h, 'nx': nx, 'ny': ny, 'bbox': (x1, y1, x2, y2), 'conf': conf})
        
        raw_detections.sort(key=lambda x: x['conf'], reverse=True)
        detections = []
        for det in raw_detections:
            is_duplicate = False
            for saved in detections:
                dist = math.sqrt((det['cx'] - saved['cx'])**2 + (det['cy'] - saved['cy'])**2)
                threshold_dist = (det['w'] + det['h']) / 2 * 0.5 
                if dist < threshold_dist: is_duplicate = True; break
            if not is_duplicate: detections.append(det)
        
        # ▼▼▼ 消えてしまっていた後半のロジック（ここから復活） ▼▼▼
        my_hand_objs, my_melds_objs, doras_objs = [], [], []
        opps = [{'river':[], 'river_data':[], 'melds':[], 'reach':False} for _ in range(4)]
        table_center = (int(base_center_x), int(base_center_y))
        
        if not detections: return my_hand_objs, my_melds_objs, doras_objs, opps, table_center, anchors, []
        
        river_candidates = []; bottom_area_tiles = []
        for t in detections:
            if t['nx'] < 0.20 and t['ny'] < 0.25: doras_objs.append(t); continue
            if (t['ny'] < 0.17) and (0.18 < t['nx'] < 0.60): opps[1]['melds'].append(t['label']); continue
            if t['nx'] < 0.17: opps[2]['melds'].append(t['label']); continue
            if (t['nx'] > 0.71) and (t['ny'] < 0.35): opps[0]['melds'].append(t['label']); continue
            
            if t['cy'] > anchors[3][1]: bottom_area_tiles.append(t)
            else: river_candidates.append(t)
        
        real_hand_tiles, self_meld_tiles = RobustGrid.split_hand_and_melds(bottom_area_tiles)
        my_hand_objs = RobustGrid.get_hand_by_count(real_hand_tiles, original_img)
        my_melds_objs = self_meld_tiles 
        
        for t in river_candidates:
            best_seat = -1; min_dist = float('inf')
            for seat, (ax, ay) in anchors.items():
                dist = math.sqrt((t['cx'] - ax)**2 + (t['cy'] - ay)**2)
                if dist < min_dist: min_dist = dist; best_seat = seat
            if best_seat == 3 and t['cy'] > anchors[3][1]: continue
            if best_seat != -1: opps[best_seat]['river_data'].append(t)
        
        for seat in range(4):
            sorted_data = RobustGrid.sort_river_tiles(opps[seat]['river_data'], seat)
            opps[seat]['river_data'] = sorted_data
            opps[seat]['river'] = [t['label'] for t in sorted_data]
        
        reach_status, debug_areas = RobustGrid.check_reach_sticks_by_color(original_img)
        for seat in range(4):
            opps[seat]['reach'] = reach_status[seat]
            if reach_status[seat] and opps[seat]['river_data']:
                opps[seat]['river_data'][-1]['is_reach_tile'] = True
        
        return my_hand_objs, my_melds_objs, doras_objs, opps, table_center, anchors, debug_areas
        
        return my_hand_objs, my_melds_objs, doras_objs, opps, table_center, anchors, debug_areas
        for seat in range(4):
            opps[seat]['reach'] = reach_status[seat]
            if reach_status[seat] and opps[seat]['river_data']:
                opps[seat]['river_data'][-1]['is_reach_tile'] = True
        
        return my_hand_objs, my_melds_objs, doras_objs, opps, table_center, anchors, debug_areas

class Visualizer:
    @staticmethod
    # ★変更: 引数の最後に wind_roi_tuple=None を追加
    def save_debug_image(img, my_hand, my_melds, doras, opps, table_center, anchors, hud_debug_boxes=[], reach_debug_areas=[], wind_roi_tuple=None):
        debug_img = img.copy()
        h, w = debug_img.shape[:2]
        cv2.line(debug_img, (int(w/2), 0), (int(w/2), h), (255, 255, 255), 1)
        cv2.line(debug_img, (0, int(h/2)), (w, int(h/2)), (255, 255, 255), 1)
        
        if anchors:
            for seat, (ax, ay) in anchors.items():
                cv2.circle(debug_img, (int(ax), int(ay)), 10, (0, 0, 255), -1)
                cv2.putText(debug_img, str(seat), (int(ax)-5, int(ay)+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        def draw_box(obj_list, color, label_prefix=""):
            for t in obj_list:
                x1 = int(t['cx'] - t['w']/2); y1 = int(t['cy'] - t['h']/2)
                x2 = int(t['cx'] + t['w']/2); y2 = int(t['cy'] + t['h']/2)
                c = (0,0,255) if t.get('is_declaration') else color
                l = "REACH:" if t.get('is_declaration') else label_prefix
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), c, 2)
                cv2.putText(debug_img, f"{l}{t['label']}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)

        draw_box(my_hand, (0, 255, 0), "H:")
        draw_box(my_melds, (255, 0, 0), "M:")
        draw_box(doras, (0, 255, 255), "D:")
        for opp in opps: draw_box(opp['river_data'], (255, 200, 0), "R:")
            
        for (bx1, by1, bx2, by2, b_key, b_text) in hud_debug_boxes:
            cv2.rectangle(debug_img, (bx1, by1), (bx2, by2), (255, 0, 255), 1)
            cv2.putText(debug_img, str(b_text), (bx1, by1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)

        # ▼▼▼▼▼▼▼▼▼▼ 追加: 自風ROIの枠を描画 ▼▼▼▼▼▼▼▼▼▼
        if wind_roi_tuple is not None:
            wx1, wy1, wx2, wy2 = wind_roi_tuple
            # シアン色(BGR: 255, 255, 0)で太めの枠線を描画
            cv2.rectangle(debug_img, (wx1, wy1), (wx2, wy2), (255, 255, 0), 3)
            cv2.putText(debug_img, "WIND ROI", (wx1, wy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        cv2.imwrite(debug_save_path, debug_img)
# ==========================================
# ⚙️ グローバル変数（記憶用）
# ==========================================
last_ocr_time = 0
last_river_signature = ()  # 前回の捨て牌（最初の3枚）を記憶
cached_hud_info = {}       # 前回のHUD結果
cached_debug_boxes = []    # デバッグ枠のキャッシュ
hud_ocr_cache = {}         # ★追加: OCRのハッシュキャッシュ (Key: MD5, Val: Text)

# ==========================================
# 🌐 サーバー処理
# ==========================================
@app.route('/analyze', methods=['POST'])
def analyze():
    global last_ocr_time, last_river_signature, cached_hud_info, cached_debug_boxes, hud_ocr_cache

    # 🔥【計測1】処理開始
    t_start = time.time()
    
    # ユーザー定義のROI座標
    wind_roi_coords = (823, 497, 873, 542)

    # 1. 画像受け取り
    if 'file' not in request.files: return "No file", 400
    file = request.files['file']
    
    t_rx_start = time.time()
    in_mem = io.BytesIO(); file.save(in_mem)
    t_rx_end = time.time()
    
    data = np.frombuffer(in_mem.getvalue(), dtype=np.uint8)
    image_size_kb = len(data) / 1024
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    
    # 🔥【計測2】画像ロード完了
    t_img_loaded = time.time()
    print(f"DEBUG: Upload Wait: {t_rx_start - t_start:.3f}s, Rx Time: {t_rx_end - t_rx_start:.3f}s, Size: {image_size_kb:.1f}KB")

    # ★Optimization 1: Early Resize for High-Res Images
    # 4KなどをそのままYOLOに入れると重いので、長辺1920にダウンサイズしておく
    h, w = img.shape[:2]
    if w > 2560 or h > 2560:
        scale = min(1920/w, 1920/h)
        nw, nh = int(w * scale), int(h * scale)
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        print(f"DEBUG: Resized Large Image: {w}x{h} -> {nw}x{nh}")


    # 2. 前処理 (Letterbox)
    img_std = RobustGrid.letterbox_image(img, target_size=(1920, 1080))

    # 3. YOLO推論 (GPU)
    try:
        res = model_main.predict(
            source=img_std, imgsz=1280, conf=0.25, iou=0.45,
            augment=False, verbose=False, device=0
        )
    except Exception as e:
        return jsonify({"error": f"YOLO Error: {e}"}), 500

    # 4. ロジック解析
    try:
        hand_objs, melds_objs, doras_objs, opps, table_center, anchors, r_areas = \
            RobustGrid.parse_frame_dual(img_std, res[0].boxes, model_main.names, 1920, 1080)
    except Exception as e:
        print(f"⚠️ Parse Error: {e}")
        hand_objs, melds_objs, doras_objs = [], [], []
        opps = [{'river':[], 'river_data':[], 'melds':[], 'reach':False} for _ in range(4)]
        table_center, anchors, r_areas = (0,0), {}, []

    # 5. スマートOCR判定 (Async & Optimized)
    my_river_labels = opps[0]['river']
    current_river_signature = tuple(my_river_labels[:3])
    needs_ocr = False
    current_time = time.time()
    
    # Initialize executor if needed
    global ocr_executor
    if 'ocr_executor' not in globals():
        ocr_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    if len(current_river_signature) < 3:
        if current_river_signature != last_river_signature: needs_ocr = True
        elif current_time - last_ocr_time > 5.0: needs_ocr = True
    elif current_river_signature != last_river_signature:
        needs_ocr = True

    # Critical: Always scan Winds (Fast, Template)
    # Uses 1920x1080 img_std so coordinates are valid
    wind_info = RobustGrid.scan_winds(img_std, matcher, hud_ocr_cache)
    
    # Update local cache with latest winds immediately
    cached_hud_info.update(wind_info)
    
    # Optional: Scan Scores (Slow, OCR) -> Async
    if needs_ocr:
        print("DEBUG: Triggering Async Score Scan...")
        # Submit task (Clone image to ensure thread safety if needed, though read-only is fine)
        # We pass hud_ocr_cache which is thread-safe for dict updates in CPython
        ocr_executor.submit(RobustGrid.scan_scores, img_std.copy(), reader, hud_ocr_cache)
        
        last_ocr_time = current_time
        last_river_signature = current_river_signature
    
    # Merge latest cache (which might have been updated by background thread from previous turn)
    # The background thread updates 'hud_ocr_cache'. We merge it into our working copy.
    cached_hud_info.update(hud_ocr_cache)
    
    hud_info = cached_hud_info
    debug_boxes = [] # simplified
    
    # 6. ステート管理更新 (ここでリーチ宣言牌のフラグ is_declaration が立ちます)
    reach_status = {i: opps[i]['reach'] for i in range(4)}
    state_manager.update(opps, reach_status)

    # 7. AI思考 (Brain)
    t_ai_start = time.time()
    final_decision, decision_debug = Strategy.decide_discard(
        hand_objs, melds_objs, doras_objs, hud_info, state_manager, brain_model, opps
    )
    t_ai_end = time.time()

    # ----------------------------------------------------
    # ★追加: 表記変換ヘルパー (1m->1萬, 1s->1草, 1p->1筒, 6z->發)
    # ----------------------------------------------------
    def to_japanese_tile_code(code):
        if not code or len(code) < 2: return code
        if code == "ツモ切り": return "ツモ切り"
        
        # 字牌変換 (1z..7z)
        map_z = {'1z':'東','2z':'南','3z':'西','4z':'北','5z':'白','6z':'發','7z':'中'}
        if code in map_z: return map_z[code]
        
        # 数牌変換
        num = code[0]
        suit = code[1]
        if suit == 'm': return f"{num}萬"
        if suit == 's': return f"{num}草" # ユーザー要望: s->草
        if suit == 'p': return f"{num}筒"
        return code

    # 📱 クライアント用通知メッセージ作成
    final_decision_jp = to_japanese_tile_code(final_decision)
    confidence_str = decision_debug.get("confidence", "0%")
    riichi_advice = decision_debug.get("riichi", "")
    pred_text = decision_debug.get("pred", "スルー")
    top_candidates = decision_debug.get("candidates", [])

    # notification_text用メッセージ構築
    strategy_msg = ""
    enemy_riichi_seats = [i for i in range(4) if opps[i]['reach'] and i != 3]
    if enemy_riichi_seats:
        if final_decision in TILE_MAP:
            is_genbutsu = True
            for r_seat in enemy_riichi_seats:
                if final_decision not in opps[r_seat]['river']: is_genbutsu = False; break
            strategy_msg = "🛡️ベタオリ" if is_genbutsu else "⚔️全ツッパ"
        elif final_decision == "ツモ切り": strategy_msg = "⚔️ツモ切り"

    notification_msg = f"{final_decision_jp} ({confidence_str})"
    
    if len(top_candidates) >= 2:
        sec_conf = float(top_candidates[1].get('base_conf', 0))
        sec_label = top_candidates[1]['label']
        sec_label_jp = to_japanese_tile_code(sec_label)
        notification_msg += f" (次: {sec_label_jp} {sec_conf:.1%})"
        
    if strategy_msg: notification_msg += f" | {strategy_msg}"
    
    # ★UI Update: Always show Riichi/Dama Advice if available
    # Remove conditions, just append.
    if riichi_advice:
        # Check if advice indicates Riichi or Dama
        # riichi_advice example: "⚠️リーチ推奨 (Model:89%, EV:500)" or "🤫ダマ (Model:1%, Wait:8枚)"
        
        # Make short version for notification
        short_advice = riichi_advice
        if "推奨" in riichi_advice:
             short_advice = riichi_advice.split("(")[0].strip() # "⚠️リーチ推奨"
             if "リーチ" in short_advice: short_advice = "⚠️リーチ"
        elif "ダマ" in riichi_advice:
             short_advice = "🤫ダマ"
        
        # Add to message
        notification_msg += f" | {short_advice}"

    
    # 鳴き予測テキストも変換する
    if pred_text != "スルー":
        # pred_text は "✅チー[4s](+33.5%) ..." のような形式
        # 正規表現で [xx] の中身を取り出して変換する
        def replacer(match):
            return f"[{to_japanese_tile_code(match.group(1))}]"
        
        pred_text_jp = re.sub(r'\[([0-9][mspz])\]', replacer, pred_text)
        notification_msg += f" | 🔮{pred_text_jp}"
    else:
        pred_text_jp = "スルー"
    # レスポンスデータ
    response_data = {
        "best_move": final_decision_jp, # ★変更: ここも日本語化
        "confidence": confidence_str,
        "notification_text": notification_msg,
        "naki_advice": pred_text_jp,    # ★変更: ここも日本語化
        "riichi_advice": riichi_advice,
        "candidates": top_candidates,
        "my_hand": [t['label'] for t in hand_objs],
        "debug": decision_debug,
        "perf": {
            "vision": f"{t_ai_start - t_img_loaded:.3f}s",
            "ai": f"{t_ai_end - t_ai_start:.3f}s",
            "total": f"{t_ai_end - t_start:.3f}s"
        }
    }

    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    # 📺 コンソール表示ロジック (ドラ変換・リーチ宣言表示対応版)
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    try:
        print("\n" + "="*70)
        # --- 基本情報 ---
        wind = hud_info.get("Round_Wind", "?")
        left = hud_info.get("Info_TilesLeft", "?")
        honba = hud_info.get("Info_Honba", "0")

        # ★修正1: ドラ表示牌から「有効ドラ」を計算して表示
        real_doras = []
        for t in doras_objs:
            d_label = t['label']
            if d_label in TILE_MAP:
                nid = TILE_MAP[d_label]
                # ドラめくりロジック
                if nid < 27: nx = nid - 8 if nid % 9 == 8 else nid + 1
                elif nid < 31: nx = 27 if nid == 30 else nid + 1
                else: nx = 31 if nid == 33 else nid + 1
                real_doras.append(INV_TILE_MAP[nx])
            else:
                real_doras.append(d_label) # 不明な場合はそのまま

        print(f"🀄 場風: {wind} | 残: {left} | 本場: {honba}")
        print(f"💎 有効ドラ: {real_doras} (表示: {[t['label'] for t in doras_objs]})")
        print("-" * 70)

        # --- 各家の情報 ---
        # wind_label (e.g. "East1") -> Round Wind is East. 
        # But who is dealer?
        # Usually checking the "East" marker on screen is needed to know dealer.
        # But assuming "seat_labels" are fixed (Self=3, Right=0, Top=1, Left=2) is camera-relative.
        # The dealer rotates.
        # If we interpret "Round_Wind" as just "Ba-kaze", we need "Ji-kaze".
        # We don't have Dealer detection logic in scanned HUD explicitly. 
        # But we can infer from "Oya" marker if we had it.
        # We don't.
        # However, usually Top is Front, Left is Left.
        # Wind assignment depends on who is East.
        
        # Let's just print Round Wind and list Seat Names clearly.
        # User asked: "自分を起点として他者の風も記載するようにしてください"
        # Standard relative winds:
        # Self (Kamicha) -> Left -> Top -> Right -> Self
        # If Self is West, then Left=South, Top=East, Right=North
        
        # Since we don't know EXACTLY what wind Self is without scanning the East marker,
        # We can only print the fixed relative positions.
        # But wait, user asked "Show MY wind".
        # "Self_Wind_Tile" detection was added in scan_hud_info?
        # Yes! line 1316: "Self_Wind_Tile": (825, 500, 50, 45)
        
        self_wind_detected = hud_info.get("Self_Wind_Tile", "?")
        # map label to kanji
        # map label to kanji
        # Now supporting both 1z format and E/S/W/N format
        w_map = {'1z':'東', '2z':'南', '3z':'西', '4z':'北',
                 'E':'東', 'S':'南', 'W':'西', 'N':'北',
                 '東(East)':'東', '南(South)':'南', '西(West)':'西', '北(North)':'北'}
        
        # Clean up "東(East)" to just "東" for logic
        if "East" in self_wind_detected: myself_wind = "東"
        elif "South" in self_wind_detected: myself_wind = "南"
        elif "West" in self_wind_detected: myself_wind = "西"
        elif "North" in self_wind_detected: myself_wind = "北"
        else: myself_wind = w_map.get(self_wind_detected, '?')
        
        # Calculate others based on Self
        # Order (Kami -> Shimo): East -> South -> West -> North
        # Counter-Clockwise on table.
        # Index: 0=Right(Shimo), 1=Top(Toimen), 2=Left(Kami), 3=Self
        # If Self=East: Right=South, Top=West, Left=North
        # If Self=South: Right=West, Top=North, Left=East
        # If Self=West: Right=North, Top=East, Left=South
        # If Self=North: Right=East, Top=South, Left=West
        
        winds_order = ['東', '南', '西', '北']
        try:
            self_idx = winds_order.index(myself_wind)
            # Right(0) is +1 (South of East)
            # Top(1) is +2
            # Left(2) is +3 (North of East)
            w_right = winds_order[(self_idx + 1) % 4]
            w_top   = winds_order[(self_idx + 2) % 4]
            w_left  = winds_order[(self_idx + 3) % 4]
            
            seat_winds = {0: w_right, 1: w_top, 2: w_left, 3: myself_wind}
        except:
            seat_winds = {0: '?', 1: '?', 2: '?', 3: '?'}

        seat_labels = {1: "対面(Top)", 2: "上家(Left)", 0: "下家(Right)", 3: "自分(Self)"}
        score_map = {0:"Score_Right", 1:"Score_Top", 2:"Score_Left", 3:"Score_Self"}
        
        for seat in [1, 2, 0, 3]: 
            name = seat_labels[seat]
            sc = hud_info.get(score_map[seat], "25000")
            reach_mark = "⚠️リーチ" if opps[seat]['reach'] else "　　　"
            
            # Wind Display
            seat_w = seat_winds[seat]
            name_disp = f"{seat_w}:{name}"
            
            # 鳴きの表示
            m_list = list(opps[seat]['melds'])
            if seat == 3: m_list.extend([t['label'] for t in melds_objs])
            meld_str = f"🗣️ {m_list}" if m_list else ""

            # ★修正2: リーチ宣言牌に「*」を付けて表示
            river_display = []
            for t_data in opps[seat]['river_data']:
                label = t_data['label']
                # is_declaration フラグがあれば * を付ける
                if t_data.get('is_declaration'):
                    river_display.append(f"*{label}")
                else:
                    river_display.append(label)
            
            river_str = " ".join(river_display)
            if len(river_str) > 65: river_str = river_str[:65] + "..."

            if len(river_str) > 65: river_str = river_str[:65] + "..."

            # name_disp を使用
            print(f"{name_disp:<15} [点数: {sc:>6}] {reach_mark} | {meld_str}")
            print(f"   🌊 {river_str}")
            if seat != 3: print("-" * 30)

        print("-" * 70)
        hand_str = [t['label'] for t in hand_objs]
        print(f"✋ 手牌: {hand_str}")
        print(f"🤖 推奨: {final_decision} ({confidence_str}) | {riichi_advice}")
        if pred_text != "スルー": print(f"🔮 鳴き予測: {pred_text}")
        if strategy_msg: print(f"🛡️ 方針: {strategy_msg}")
        
        print("="*70)
        print(f"⏱️ [Perf] Vision: {response_data['perf']['vision']} | AI: {response_data['perf']['ai']} | Total: {response_data['perf']['total']}")
    except Exception as e:
        print(f"⚠️ Display Error: {e}")
        import traceback
        traceback.print_exc()

    # 💾 データ自動収集
    if 'COLLECT_DATA' in globals() and COLLECT_DATA:
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"jantama_{timestamp}"
            save_img_path = os.path.join(COLLECT_DIR, "images", f"{filename}.jpg")
            cv2.imwrite(save_img_path, img_std)
            save_lbl_path = os.path.join(COLLECT_DIR, "labels", f"{filename}.txt")
            with open(save_lbl_path, "w") as f:
                h, w = img_std.shape[:2]
                for box in res[0].boxes:
                    cls = int(box.cls[0]); cx, cy, bw, bh = box.xywh[0].tolist()
                    nx, ny, nw, nh = cx/w, cy/h, bw/w, bh/h
                    f.write(f"{cls} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n")
        except Exception as e:
            print(f"⚠️ Save Error: {e}")

    return jsonify(response_data)


if __name__ == '__main__':
    print("🚀 Full Stack Mahjong Server Started (NO Compromise Version).")
    app.run(host='0.0.0.0', port=5000)