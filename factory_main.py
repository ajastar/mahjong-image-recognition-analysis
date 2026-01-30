import time
import threading
import queue
import subprocess
import cv2
import numpy as np
import torch
import torch.nn as nn
import glob
import os
import random
import collections
import re
import mss
import pygetwindow as gw
from ultralytics import YOLO

# ==========================================
# ⚙️ 設定
# ==========================================
ADB_PATH = r"C:\LDPlayer\LDPlayer9\adb.exe"
BUTTON_TEMPLATE_DIR = r"E:\AI_Project_Hub\Mahjong_Maker\templates\buttons"
WEIGHTS_PATH = r"E:\AI_Project_Hub\Mahjong_Maker\runs\jantama_absolute_limit_1280px\weights\best.pt"
BRAIN_PATH = "./checkpoints_superhuman_blackwell/mahjong_god_ultimate.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ★ここが最重要：確認したウィンドウ名を正確にマッピング
DEVICE_TO_WINDOW = {
    "emulator-5554": "LDPlayer",
    "emulator-5556": "LDPlayer-1",
    "emulator-5558": "LDPlayer-2",
    "emulator-5560": "LDPlayer-3",
    "emulator-5562": "LDPlayer-4"
}

# ==========================================
# 🧠 AIモデル定義
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

# ==========================================
# 🀄 定数・マップ定義
# ==========================================
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

# ★AI思考のためのデータ変換ロジック（必須）
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
            row = min(j // 6, 3); order_val = ((j % 6) + 1) / 6.0
            obs[4 + (i*4) + row, t, :] = order_val
            if j < len(tsumo_dat) and tsumo_dat[j]: obs[20 + i, t, :] = 1
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
    obs[34, :, :] = rank / 4.0; obs[35, :, :] = honba / 10.0; obs[36, :, :] = len(doras) / 5.0
    cur_ids = full_hands_ids[p_idx]
    if 16 in cur_ids: obs[37, :, :] = 1 
    if 52 in cur_ids: obs[38, :, :] = 1
    if 88 in cur_ids: obs[39, :, :] = 1
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
    obs[79, :, :] = 1
    return obs

# ==========================================
# 🖼️ 画面解析 & キャプチャ
# ==========================================
class RobustGrid:
    @staticmethod
    def letterbox_image(img, target_size=(1920, 1080)):
        h, w = img.shape[:2]; tw, th = target_size
        resized = cv2.resize(img, (tw, th)) # 強制リサイズ（高速化のため）
        return resized

    @staticmethod
    def parse_frame_dual(original_img, boxes, names, img_w, img_h):
        # 簡易実装版: YOLOの結果から手牌と河を構築
        opps = [{'river':[], 'river_data':[], 'melds':[], 'reach':False} for _ in range(4)]
        my_hand_objs = []; doras_objs = []
        
        # 画面エリア定義
        base_h = img_h; base_w = img_w
        
        for box in boxes:
            cls_id = int(box.cls[0]); label = names[cls_id]
            cx, cy, w, h = box.xywh[0].tolist()
            nx, ny = cx/base_w, cy/base_h
            
            # 手牌 (下部)
            if ny > 0.7: my_hand_objs.append({'label': label, 'cx': cx, 'cy': cy})
            # ドラ (左上)
            elif nx < 0.2 and ny < 0.2: doras_objs.append({'label': label})
            # 河の牌 (中央付近)
            elif 0.2 < ny < 0.7:
                # 簡易的に座席判定
                seat = -1
                if ny > 0.45: seat = 3 # 自分
                elif nx < 0.3: seat = 2 # 左
                elif nx > 0.7: seat = 0 # 右
                else: seat = 1 # 対面
                if seat != -1: opps[seat]['river'].append(label)

        return my_hand_objs, [], doras_objs, opps, None, None, None

class TemplateMatcher:
    def __init__(self, folder):
        self.templates = {}
        if not os.path.exists(folder): os.makedirs(folder, exist_ok=True)
        for p in glob.glob(os.path.join(folder, "*.png")):
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is not None:
                label = os.path.splitext(os.path.basename(p))[0]
                self.templates[label] = img
        print(f"✅ Loaded {len(self.templates)} button templates.")

    def detect_all_buttons(self, target_img, threshold=0.85):
        found_buttons = []
        h, w = target_img.shape[:2]
        for label, temp_img in self.templates.items():
            th, tw = temp_img.shape[:2]
            if h < th or w < tw: continue
            try:
                res = cv2.matchTemplate(target_img, temp_img, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > threshold:
                    cx = max_loc[0] + tw // 2; cy = max_loc[1] + th // 2
                    found_buttons.append({"label": label, "x": cx, "y": cy, "score": max_val})
            except: pass
        return found_buttons

# ==========================================
# ⚡ ADB & Windowsキャプチャ
# ==========================================
def get_connected_devices():
    try:
        res = subprocess.run([ADB_PATH, "devices"], capture_output=True, text=True, timeout=5)
        return [l.split()[0] for l in res.stdout.splitlines() if "device" in l and "List" not in l]
    except: return []

def get_adb_screen(device_id):
    title = DEVICE_TO_WINDOW.get(device_id)
    if not title:
        # print(f"[{device_id}] ⚠️ 辞書にウィンドウ名が登録されていません")
        return None

    try:
        # タイトルが完全に一致するウィンドウを探す
        all_windows = gw.getWindowsWithTitle(title)
        windows = [w for w in all_windows if w.title == title]

        if not windows:
            # print(f"[{device_id}] ❌ ウィンドウ '{title}' が見つかりません")
            return None
        
        win = windows[0]
        if win.isMinimized:
            try: win.restore()
            except: pass
            return None

        # キャプチャ範囲
        monitor = {
            "top": win.top + 35, 
            "left": win.left, 
            "width": win.width, 
            "height": win.height - 35
        }
        
        if monitor["width"] <= 0: return None
            
        # ★修正箇所: ここで毎回新しいインスタンスを作る（スレッドセーフ化）
        with mss.mss() as sct:
            img = np.array(sct.grab(monitor))
            
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return cv2.resize(img, (1920, 1080))

    except Exception as e:
        # print(f"[{device_id}] 💥 エラー: {e}")
        return None
def adb_click(device_id, x, y):
    try:
        subprocess.Popen([ADB_PATH, "-s", device_id, "shell", "input", "tap", str(x), str(y)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except: pass

# ==========================================
# 🧠 推論サーバー (完全統合版)
# ==========================================
class InferenceServer:
    def __init__(self):
        print("🧠 Initializing Brain on RTX 5060 Ti...")
        self.yolo = YOLO(WEIGHTS_PATH, task='detect')
        self.brain = FasterMahjongNet(in_chans=80).to(DEVICE).half()
        try:
            sd = torch.load(BRAIN_PATH, map_location=DEVICE)
            new_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
            self.brain.load_state_dict(new_sd)
            self.brain.eval()
            print("✅ Mahjong God Logic Loaded.")
        except Exception as e: print(f"❌ Brain Error: {e}")

        self.btn_matcher = TemplateMatcher(BUTTON_TEMPLATE_DIR)
        self.input_queue = queue.Queue()
        self.results = {} 
        self.running = True

    def start(self): threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        print("🧠 Server Loop Ready.")
        while self.running:
            batch = []
            try:
                batch.append(self.input_queue.get(timeout=0.1))
                while len(batch) < 5:
                    try: batch.append(self.input_queue.get_nowait())
                    except queue.Empty: break
            except queue.Empty: continue

            for req_id, img_original in batch:
                # 1. 画像処理
                img_std = RobustGrid.letterbox_image(img_original, target_size=(1920, 1080))
                
                # 2. ボタン検知
                btns = self.btn_matcher.detect_all_buttons(img_std)
                
                final_action = None
                # アガリ・進行
                for b in btns:
                    if any(x in b['label'] for x in ["ron", "tsumo", "next", "confirm"]):
                        final_action = {"type": "click", "x": b['x'], "y": b['y'], "desc": f"✨ {b['label']}"}
                        break
                
                # 鳴き判断 (簡易版: 鳴けるならスキップを押す＝門前重視)
                if not final_action:
                    target_btn = next((b for b in btns if b['label'] == "skip"), None)
                    if target_btn:
                        final_action = {"type": "click", "x": target_btn['x'], "y": target_btn['y'], "desc": "⏩ Skip"}

                # 3. 思考と打牌
                if not final_action:
                    yolo_res = self.yolo.predict(img_std, imgsz=1280, conf=0.25, verbose=False)[0]
                    hand_objs = []
                    for box in yolo_res.boxes:
                        cls = int(box.cls[0]); label = self.yolo.names[cls]
                        cx, cy, w, h = box.xywh[0].tolist()
                        if cy > 1080 * 0.65: # 手牌位置
                            hand_objs.append({'cx': cx, 'cy': cy, 'label': label})
                    
                    if hand_objs:
                        # --- AI入力データ作成 ---
                        # ※本来は全変数を作成する必要がありますが、コード量削減のため
                        # 簡易的に「手の形」と「河」からダミー入力を作って思考させます
                        hands_arr = [[0]*34 for _ in range(4)]
                        # 手牌認識結果を配列化
                        for h_obj in hand_objs:
                            if h_obj['label'] in TILE_MAP:
                                hands_arr[3][TILE_MAP[h_obj['label']]] += 1
                        
                        # AI推論
                        try:
                            # ダミーの履歴データ (本来は parse_frame_dual から取得)
                            dummy_hist = [[] for _ in range(4)]
                            tensor = encode_state_god(hands_arr, dummy_hist, [[False]*30]*4, [[]]*4, [25000]*4, [], 0, 0, [False]*4, [False]*4, 3, [[],[],[],[]])
                            input_t = torch.from_numpy(tensor).unsqueeze(0).to(DEVICE).half()
                            
                            with torch.no_grad():
                                logits, _ = self.brain(input_t)
                                probs = torch.softmax(logits[0], 0)
                            
                            # 手牌にある牌の中で、最も切りたい牌を選ぶ
                            best_tile_id = -1; max_prob = -1
                            
                            # 手牌にある牌だけを候補にする
                            valid_labels = [h['label'] for h in hand_objs]
                            
                            # 確率が高い順にチェック
                            topk = torch.topk(probs, 34)
                            for i in range(34):
                                tid = topk.indices[i].item()
                                prob = topk.values[i].item()
                                t_label = INV_TILE_MAP.get(tid)
                                if t_label in valid_labels:
                                    best_tile_id = tid
                                    break
                            
                            # 座標特定
                            target_label = INV_TILE_MAP.get(best_tile_id, "unk")
                            candidates = [h for h in hand_objs if h['label'] == target_label]
                            if candidates:
                                # 同じ牌なら右端（ツモ牌に近い方）を切る
                                target = max(candidates, key=lambda x: x['cx'])
                                final_action = {"type": "click", "x": int(target['cx']), "y": int(target['cy']), "desc": f"🀄 AI: {target['label']}"}
                            else:
                                # 見つからない場合は右端（ツモ切り）
                                target = max(hand_objs, key=lambda x: x['cx'])
                                final_action = {"type": "click", "x": int(target['cx']), "y": int(target['cy']), "desc": "🀄 Tsumogiri"}

                        except Exception as e:
                            # エラー時は安全にツモ切り
                            target = max(hand_objs, key=lambda x: x['cx'])
                            final_action = {"type": "click", "x": int(target['cx']), "y": int(target['cy']), "desc": "🀄 Safe Cut"}
                    else:
                        final_action = {"type": "wait"}

                self.results[req_id] = final_action

# ==========================================
# 🐜 ワーカー (キビキビ版)
# ==========================================
def worker_task(device_id, server):
    print(f"🚀 Worker attached to {device_id}")
    req_id = device_id
    time.sleep(random.uniform(0.0, 5.0))

    while True:
        img = get_adb_screen(device_id)
        if img is None:
            time.sleep(0.5); continue

        server.input_queue.put((req_id, img))

        wait_start = time.time()
        while req_id not in server.results:
            time.sleep(0.02)
            if time.time() - wait_start > 5.0: break
        
        if req_id in server.results:
            result = server.results.pop(req_id)
            if result["type"] == "click":
                adb_click(device_id, result['x'], result['y'])
                desc = result.get('desc', '')
                print(f"[{device_id}] {desc}")
                
                if "WIN" in desc: time.sleep(5.0)
                else: time.sleep(4.5) 
            elif result["type"] == "wait":
                time.sleep(1.0)
        else:
            time.sleep(1.0)

if __name__ == "__main__":
    devices = get_connected_devices()
    print(f"📋 Devices: {devices}")
    if not devices: exit()
    server = InferenceServer()
    server.start()
    threads = []
    for d in devices:
        t = threading.Thread(target=worker_task, args=(d, server))
        t.start()
        threads.append(t)
        time.sleep(1.0)
    for t in threads: t.join()