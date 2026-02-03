import os
import time
import datetime 
import numpy as np
import cv2
import math
import collections
import glob
import torch
import torch.nn as nn
import re
import pygetwindow as gw
from mss import mss
from PIL import Image
from ultralytics import YOLO
import easyocr
import random
import threading
import sys

# ★高速化ライブラリ
from ppadb.client import Client as AdbClient

# ==========================================
# ⚙️ 設定
# ==========================================
# ADBサーバーへの接続設定 (LDPlayer標準ポート)
ADB_HOST = "127.0.0.1"
ADB_PORT = 5037 

# エミュレータの内部解像度 (設定と完全に一致させること)
LD_WIDTH = 1920
LD_HEIGHT = 1080

weights_path = r"E:\AI_Project_Hub\Mahjong_Maker\runs\jantama_absolute_limit_1280px\weights\best.pt"
if not os.path.exists(weights_path):
    weights_path = r"E:\AI_Project_Hub\Mahjong_Maker\runs\jantama_absolute_limit_1280px\best.pt"

BRAIN_MODEL_PATH = "./checkpoints_superhuman_blackwell/mahjong_god_ultimate.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

base_dir = r"E:\AI_Project_Hub\Mahjong_Maker"
template_dir = os.path.join(base_dir, "templates")
TARGET_WINDOW_TITLE = "LDPlayer" 

# デバッグ画像保存
debug_dir = os.path.join(base_dir, "debug_clicks")
os.makedirs(debug_dir, exist_ok=True)

# --- 操作設定 (スムーズ化) ---
CLICK_CONFIDENCE_THRESHOLD = 0.55 # 閾値を少し下げて反応を良くする
MIN_COOLDOWN = 2.5                # 最低待機時間を短縮
MAX_COOLDOWN = 4.0                # 最大待機時間も短縮

# AI排他制御ロック
ai_lock = threading.Lock()

# ==========================================
# 🔌 ADB Connection Manager (高速化の肝)
# ==========================================
class ADBManager:
    def __init__(self):
        self.client = None
        self.devices = {}
        try:
            self.client = AdbClient(host=ADB_HOST, port=ADB_PORT)
            # サーバーが起動していない場合は起動を試みる
            if not self.client.version():
                print("⚠️ ADB Server not running. Starting...")
                os.system("adb start-server")
                time.sleep(2)
                self.client = AdbClient(host=ADB_HOST, port=ADB_PORT)
            print("✅ ADB Server Connected.")
        except Exception as e:
            print(f"❌ ADB Connection Failed: {e}")
            print("👉 Try running 'adb start-server' in your terminal.")

    def get_device(self, title):
        """
        ウィンドウタイトルから対応するADBデバイスを取得
        LDPlayerのポートルール: 5555 + (index * 2)
        """
        index = 0
        if title != "LDPlayer":
            match = re.search(r"LDPlayer-(\d+)", title)
            if match:
                index = int(match.group(1))
        
        target_port = 5555 + (index * 2)
        serial = f"127.0.0.1:{target_port}"
        
        # すでに接続済みならそれを返す
        if serial in self.devices:
            return self.devices[serial]
        
        # 新規取得
        try:
            device = self.client.device(serial)
            if device:
                self.devices[serial] = device
                return device
        except:
            pass
        return None

adb_manager = ADBManager()

# ==========================================
# 🧠 Brain & Network (変更なし)
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

TILE_MAP = {
    "1m":0, "2m":1, "3m":2, "4m":3, "5m":4, "6m":5, "7m":6, "8m":7, "9m":8,
    "1p":9, "2p":10, "3p":11, "4p":12, "5p":13, "6p":14, "7p":15, "8p":16, "9p":17,
    "1s":18, "2s":19, "3s":20, "4s":21, "5s":22, "6s":23, "7s":24, "8s":25, "9s":26,
    "1z":27, "2z":28, "3z":29, "4z":30, "5z":31, "6z":32, "7z":33
}
INV_TILE_MAP = {v: k for k, v in TILE_MAP.items()}
TILE_MAP["0m"] = 4; TILE_MAP["0p"] = 13; TILE_MAP["0s"] = 22

def get_suji(tile):
    suji = []
    if tile >= 27: return suji
    num = tile % 9
    if num >= 3: suji.append(tile - 3)
    if num <= 5: suji.append(tile + 3)
    return suji

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
# 🛠️ 初期化
# ==========================================
print("⏳ Loading Models for Jantama Turbo...")
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
        print("✅ Brain Ready (Fast Mode).")
    except Exception as e:
        print(f"❌ Brain Load Error: {e}")

class TemplateMatcher:
    def __init__(self, folder):
        self.templates = {}
        paths = glob.glob(os.path.join(folder, "*.*"))
        for p in paths:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is not None:
                label = os.path.splitext(os.path.basename(p))[0]
                self.templates[label] = img
    def match(self, target_img, threshold=0.8):
        if not self.templates: return None, 0.0
        best_score = -1.0; best_label = None
        h, w = target_img.shape[:2]
        for label, temp_img in self.templates.items():
            try:
                res = cv2.matchTemplate(target_img, cv2.resize(temp_img, (w, h)), cv2.TM_CCOEFF_NORMED)
                score = np.max(res)
                if score > best_score: best_score = score; best_label = label
            except: pass
        return (best_label, best_score) if best_score > threshold else (None, best_score)

matcher = TemplateMatcher(template_dir)

class GameStateManager:
    def __init__(self):
        self.memories = {}
    def update(self, opps, reach_status):
        now = time.time()
        expired = [s for s, m in self.memories.items() if now - m['timestamp'] > 600]
        for s in expired: del self.memories[s]
        for seat in range(4):
            curr_river = [t['label'] for t in opps[seat]['river_data']]
            is_matched = False
            if seat in self.memories:
                mem = self.memories[seat]
                snap = mem['snapshot']
                if len(curr_river) >= len(snap) and curr_river[:len(snap)] == snap:
                    is_matched = True
                    idx = mem['decl_idx']
                    if 0 <= idx < len(opps[seat]['river_data']):
                        opps[seat]['river_data'][idx]['is_declaration'] = True
                    self.memories[seat]['timestamp'] = now
            if reach_status[seat] and not is_matched:
                if len(curr_river) > 0:
                    self.memories[seat] = {'snapshot': list(curr_river), 'decl_idx': len(curr_river) - 1, 'timestamp': now}
                    opps[seat]['river_data'][-1]['is_declaration'] = True

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
    def scan_hud_info(img_std, reader, matcher):
        if reader is None: return {}, []
        areas = {"Round_Wind": (922, 414, 78, 30), "Info_TilesLeft": (948, 449, 50, 26), "Score_Top": (922, 380, 78, 30), "Score_Left": (877, 415, 30, 60), "Score_Right": (1019, 415, 30, 60), "Score_Self": (920, 470, 78, 50), "Info_Kyoutaku": (212, 230, 30, 35), "Info_Honba": (324, 230, 30, 35), "Self_Wind_Tile": (825, 500, 50, 45)}
        results = {}; debug_boxes = []
        for key, (ax, ay, aw, ah) in areas.items():
            x1, y1 = int(ax), int(ay); x2, y2 = int(ax+aw), int(ay+ah)
            crop_img = img_std[y1:y2, x1:x2]
            if crop_img.size == 0: continue
            if key == "Score_Top": crop_img = cv2.rotate(crop_img, cv2.ROTATE_180)
            elif key == "Score_Left": crop_img = cv2.rotate(crop_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif key == "Score_Right": crop_img = cv2.rotate(crop_img, cv2.ROTATE_90_CLOCKWISE)
            val = "?"
            if key in ["Round_Wind", "Self_Wind_Tile"]:
                label, _ = matcher.match(crop_img, threshold=0.8)
                val = label if label else "Unk"
            else:
                try: 
                    text_list = reader.readtext(crop_img, detail=0, allowlist='0123456789x,')
                    text = "".join(text_list).replace(',', ''); val = text if text else "0"
                except: pass
            results[key] = val; debug_boxes.append((x1, y1, x2, y2, key, val))
        return results, debug_boxes
    @staticmethod
    def is_back_side_tile(img_bgr):
        if img_bgr is None or img_bgr.size == 0: return False
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        crop_h, crop_w = int(h*0.4), int(w*0.4)
        cy, cx = h//2, w//2
        center_region = hsv[cy-crop_h//2 : cy+crop_h//2, cx-crop_w//2 : cx+crop_w//2]
        if center_region.size == 0: return False
        s_channel = center_region[:, :, 1]; v_channel = center_region[:, :, 2]
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
            curr_t = tiles[i]; next_t = tiles[i+1]
            dist = next_t['cx'] - curr_t['cx']
            if curr_t['nx'] > 0.45 and dist > avg_w * 1.5:
                split_idx = i + 1; break 
        if split_idx != -1: return tiles[:split_idx], tiles[split_idx:]
        else: return tiles, []
    @staticmethod
    def get_hand_by_count(bottom_tiles, img_original):
        if not bottom_tiles: return []
        bottom_tiles.sort(key=lambda t: t['cx'])
        tiles_reversed = list(reversed(bottom_tiles))
        final_list = []; total_count = 0; i = 0
        while i < len(tiles_reversed):
            if total_count >= 14: break
            current_tile = tiles_reversed[i]; is_kan = False
            if i + 3 < len(tiles_reversed):
                group_of_4 = tiles_reversed[i : i+4]
                avg_w = np.mean([t['w'] for t in group_of_4])
                width_span = group_of_4[0]['cx'] - group_of_4[3]['cx'] 
                if width_span < avg_w * 4.5:
                    back_tile_count = 0
                    for t in group_of_4:
                        iy1, iy2 = int(t['bbox'][1]), int(t['bbox'][3])
                        ix1, ix2 = int(t['bbox'][0]), int(t['bbox'][2])
                        if iy1<iy2 and ix1<ix2:
                            t_img = img_original[iy1:iy2, ix1:ix2]
                            if RobustGrid.is_back_side_tile(t_img): back_tile_count += 1
                    if back_tile_count > 0: is_kan = True
                    else:
                        labels = [t['label'] for t in group_of_4]
                        if len(set(labels)) == 1: is_kan = True
            if is_kan:
                kan_group = tiles_reversed[i : i+4]; final_list.extend(kan_group)
                total_count += 3; i += 4
            else:
                final_list.append(current_tile); total_count += 1; i += 1
        final_list.sort(key=lambda t: t['cx'])
        return final_list
    @staticmethod
    def check_reach_sticks_by_color(img_std):
        centers = {1: (944, 367), 2: (848, 460), 0: (1078, 460), 3: (930, 533)}
        MARGIN = 5; REF_BGR = np.array([87, 75, 78], dtype=np.float32); DISTANCE_THRESHOLD = 50.0
        reach_status = {0:False, 1:False, 2:False, 3:False}; debug_areas = []
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
        raw_detections = []
        base_center_x = img_w * 0.5; base_center_y = img_h * 0.42; R = img_h * 0.35
        bottom_anchor_y = base_center_y + R - (img_h * 0.04)
        anchors = {1: (base_center_x, base_center_y - R), 2: (base_center_x - R, base_center_y), 0: (base_center_x + R, base_center_y), 3: (base_center_x, bottom_anchor_y)}
        for box in boxes:
            cls_id = int(box.cls[0]); label = names[cls_id]
            cx, cy, w, h = box.xywh[0].tolist()
            x1, y1 = max(0, int(cx - w/2)), max(0, int(cy - h/2)); x2, y2 = min(img_w, int(cx + w/2)), min(img_h, int(cy + h/2))
            conf = float(box.conf[0]); nx, ny = cx/img_w, cy/img_h
            if cy < anchors[3][1]: 
                iy1, iy2 = int(y1), int(y2); ix1, ix2 = int(x1), int(x2)
                if iy1 < iy2 and ix1 < ix2 and w > 5 and h > 5:
                    tile_img = original_img[iy1:iy2, ix1:ix2]
                    if RobustGrid.is_back_side_tile(tile_img): continue
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
            opps[seat]['river_data'] = sorted_data; opps[seat]['river'] = [t['label'] for t in sorted_data]
        reach_status, debug_areas = RobustGrid.check_reach_sticks_by_color(original_img)
        for seat in range(4):
            opps[seat]['reach'] = reach_status[seat]
            if reach_status[seat] and opps[seat]['river_data']: opps[seat]['river_data'][-1]['is_reach_tile'] = True
        return my_hand_objs, my_melds_objs, doras_objs, opps, table_center, anchors, debug_areas

# ==========================================
# 🧵 スレッド管理クラス
# ==========================================
class InstanceThread(threading.Thread):
    def __init__(self, hwnd, title):
        super().__init__()
        self.hwnd = hwnd
        self.title = title
        self.running = True
        self.state_manager = GameStateManager()
        self.last_action_time = 0
        self.adb_device = adb_manager.get_device(title)
        
        status = "✅ Connected" if self.adb_device else "❌ Disconnected"
        print(f"🧵 Thread Started: {title} -> {status}")

    def run(self):
        sct = mss()
        # 起動タイミングを少しずらしてCPUスパイクを回避
        time.sleep(random.uniform(0, 2))
        
        while self.running:
            try:
                try: win = gw.getWindowsWithTitle(self.title)[0]
                except: self.running = False; break
                if win.isMinimized: time.sleep(1); continue
                
                # アクション待ち中なら解析スキップ
                if time.time() - self.last_action_time < MIN_COOLDOWN:
                    time.sleep(0.5); continue

                monitor = {"top": win.top, "left": win.left, "width": win.width, "height": win.height}
                sct_img = sct.grab(monitor)
                img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                
                self._analyze_and_act(frame_bgr)
                time.sleep(0.5) # ループ自体を少しゆっくりに
            except Exception as e:
                print(f"⚠️ Error in {self.title}: {e}"); time.sleep(1)

    def _analyze_and_act(self, img_bgr):
        img_std = RobustGrid.letterbox_image(img_bgr, target_size=(1920, 1080))
        hud_info, debug_boxes = RobustGrid.scan_hud_info(img_std, reader, matcher)
        
        with ai_lock: 
            res = model_main.predict(source=img_std, imgsz=1280, conf=0.25, iou=0.45, augment=False, verbose=False)
        
        hand_objs, melds_objs, doras_objs, opps, table_center, anchors, r_areas = RobustGrid.parse_frame_dual(img_std, res[0].boxes, model_main.names, 1920, 1080)
        reach_status = {i: opps[i]['reach'] for i in range(4)}
        self.state_manager.update(opps, reach_status)

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
        try: honba_val = int(hud_info.get("Info_Honba", "0"))
        except: honba_val = 0
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
                nid = TILE_MAP[d]; nx = 0
                if nid < 27: nx = nid - 8 if nid % 9 == 8 else nid + 1
                elif nid < 31: nx = 27 if nid == 30 else nid + 1
                else: nx = 31 if nid == 33 else nid + 1
                doras_ids.append(nx)

        best_move = "Wait"; confidence = 0.0; riichi_advice = "🤫"
        if brain_model and len(hand) > 0:
            try:
                tsumogiri_dummy = [[False]*len(d) for d in discards_hist]
                tensor_now = encode_state_god(hands_arr, discards_hist, tsumogiri_dummy, melds_arr, scores, doras_ids, round_val, honba_val, riichi_sticks, [False]*4, 3, full_hands_ids_dummy)
                t_now_batch = torch.from_numpy(tensor_now).unsqueeze(0).to(DEVICE).half()
                with ai_lock: logits, naki_now = brain_model(t_now_batch)
                mask = torch.full((35,), -float('inf'), device=DEVICE)
                for i in range(34):
                    if hands_arr[3][i] > 0: mask[i] = 0
                mask[34] = 0 
                probs = torch.softmax(logits[0] + mask, 0)
                topk_vals, topk_indices = torch.topk(probs, 5)
                candidates = []
                for i in range(5):
                    idx = topk_indices[i].item(); val = topk_vals[i].item()
                    if val < 0.01: continue 
                    label = "ツモ切り" if idx == 34 else INV_TILE_MAP[idx]
                    candidates.append({"tile": label, "idx": idx, "base_conf": val})
                if candidates:
                    best_move = candidates[0]["tile"]; confidence = candidates[0]["base_conf"]
                    if best_move == "ツモ切り" and sum(hands_arr[3]) < 14 and len(candidates) > 1:
                         best_move = candidates[1]["tile"]; confidence = candidates[1]["base_conf"]
                riichi_prob = torch.softmax(naki_now[0], 0)[4].item()
                if riichi_prob > 0.5: riichi_advice = f"⚠️リーチ推奨"
                elif riichi_prob > 0.20: riichi_advice = f"🤔リーチ考慮"
            except Exception as e: print(f"[{self.title}] Infer Err: {e}")
        
        print(f"[{self.title}] 打: {best_move} ({confidence:.1%}) {riichi_advice}")
        
        # === 高速フリック実行 (ADB Socket) ===
        if (best_move != "Wait") and (confidence > CLICK_CONFIDENCE_THRESHOLD):
            target_obj = None
            if best_move == "ツモ切り":
                if hand_objs: hand_objs.sort(key=lambda x: x['cx']); target_obj = hand_objs[-1]
            else:
                candidates_objs = [obj for obj in hand_objs if obj['label'] == best_move]
                if candidates_objs: candidates_objs.sort(key=lambda x: x['cx']); target_obj = candidates_objs[-1] 
            
            if target_obj and self.adb_device:
                # ログ保存
                ts = datetime.datetime.now().strftime("%H%M%S_%f")
                safe_title = self.title.replace(" ", "_").replace("-", "_")
                save_path = os.path.join(debug_dir, f"{safe_title}_{ts}_ACTION_{target_obj['label']}.jpg")
                try: cv2.imwrite(save_path, img_bgr)
                except: pass

                # 座標計算
                x = int(target_obj['nx'] * LD_WIDTH)
                y = int(target_obj['ny'] * LD_HEIGHT)
                end_x = x + random.randint(-5, 5)
                end_y = max(10, y - 300)
                duration = random.randint(150, 250)

                print(f"   🚀 Fast Flick! -> {target_obj['label']} on {self.title}")
                
                # ★高速通信: ここが以前のsubprocessとは違い一瞬で終わる
                try:
                    self.adb_device.shell(f"input swipe {x} {y} {end_x} {end_y} {duration}")
                    self.last_action_time = time.time()
                except Exception as e:
                    print(f"   ❌ ADB Send Failed: {e}")
                    # 接続切れなら再取得を試みる
                    self.adb_device = adb_manager.get_device(self.title)

    def stop(self): self.running = False

# ==========================================
# 🔄 メインループ
# ==========================================
def main_loop():
    print(f"🔍 Searching for multiple windows with title containing '{TARGET_WINDOW_TITLE}'...")
    threads = {} 
    while True:
        try:
            all_windows = gw.getWindowsWithTitle(TARGET_WINDOW_TITLE)
            active_hwnds = [w._hWnd for w in all_windows]
            for hwnd in list(threads.keys()):
                if hwnd not in active_hwnds:
                    print(f"🛑 Stopping thread for closed window ID: {hwnd}")
                    threads[hwnd].stop(); del threads[hwnd]
            for w in all_windows:
                if w.width < 100 or w.height < 100: continue
                hwnd = w._hWnd
                if hwnd not in threads:
                    t = InstanceThread(hwnd, w.title); t.start(); threads[hwnd] = t
            time.sleep(1) 
        except KeyboardInterrupt:
            print("🛑 全停止"); [t.stop() for t in threads.values()]; break
        except Exception as e: print(f"⚠️ Main Loop Error: {e}"); time.sleep(1)

if __name__ == '__main__':
    print("🚀 TURBO Auto Multi-Instance Mahjong AI Started.")
    main_loop()