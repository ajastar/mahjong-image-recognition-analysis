import os
import io
import time
import json
import subprocess
import numpy as np
import cv2
import math
import collections
from flask import Flask, request, jsonify
from ultralytics import YOLO

# Rotation AI (必要な場合のみ有効化、なければダミー動作)
import torch
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# ==================================================================================
#  ⚙️ 設定
# ==================================================================================
base_dir = r"E:\AI_Project_Hub\Mahjong_Maker"
weights_dir = os.path.join(base_dir, r"runs\jantama_absolute_limit_1280px\weights")
path_engine = os.path.join(weights_dir, "best.engine")
debug_save_path = os.path.join(base_dir, "debug_result.jpg")

# ★ Akochan設定
AKOCHAN_EXE = os.path.join(base_dir, r"akochan\akochan.exe")
AKOCHAN_CONF = os.path.join(base_dir, r"akochan\tactics.json")

STD_W, STD_H = 1920, 1080 

# ==================================================================================
#  🤖 AI Loaders
# ==================================================================================
if os.path.exists(path_engine):
    print("🚀 AI #1: Loading TensorRT Engine...")
    model_main = YOLO(path_engine, task='detect')
else:
    print("⚠️ AI #1: Loading PyTorch Model...")
    model_main = YOLO(os.path.join(weights_dir, "best.pt"), task='detect')

# Rotation AI (簡易実装)
class RotationAI:
    def __init__(self, path):
        self.loaded = False
        # 必要であればここにロード処理を記述
    def predict(self, img): return 1 # Default Vertical

model_rotation = RotationAI("rotation_net.pth")

# ==================================================================================
#  🧠 Akochan Bridge (The Logic Core)
# ==================================================================================
class AkochanBridge:
    def __init__(self, exe, conf):
        self.exe = exe
        self.conf = conf

    def _make_mjai(self, hand, melds, indicators, opps):
        # Akochanはコンテキスト(文脈)がないとエラーになる場合があるため
        # 最小限の「開局→配牌→現在」の偽装ログを作成する
        events = [{"type": "start_game"}, {"type": "start_kyoku", "bakaze":"E", "kyoku":1, "honba":0, "kyotaku":0, "oya":0, "dora_marker": indicators[0] if indicators else "5z", "tehais":[["?"]*13, ["?"]*13, ["?"]*13, hand]}]
        
        # 河の再現
        max_len = max([len(o['river']) for o in opps])
        for t in range(max_len):
            for s in range(4):
                if t < len(opps[s]['river']):
                    tile = opps[s]['river'][t]
                    # ツモ
                    events.append({"type": "tsumo", "actor": s, "pai": tile if s==3 else "?"})
                    # リーチ
                    if opps[s]['reach'] and t == len(opps[s]['river'])-1:
                         events.append({"type": "reach", "actor": s})
                         events.append({"type": "reach_accepted", "actor": s})
                    # 打牌
                    events.append({"type": "dahai", "actor": s, "pai": tile, "tsumogiri": s!=3})

        # 現在のツモ (手牌が3n+2枚の時)
        if len(hand) % 3 == 2:
            events.append({"type": "tsumo", "actor": 3, "pai": hand[-1]})

        return events

    def think(self, hand, melds, indicators, opps):
        mjai = self._make_mjai(hand, melds, indicators, opps)
        inp_str = json.dumps(mjai)
        
        # Akochan実行: "akochan.exe tactics.json"
        # ※パイプ経由でMJAIログを流し込む
        cmd = [self.exe, self.conf]
        
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
            out, err = proc.communicate(input=inp_str)

            if proc.returncode != 0:
                return f"Error: {err}", {}

            # 結果解析 (最後のdahaiを探す)
            best_move = "Unknown"
            lines = out.strip().split('\n')
            for line in reversed(lines):
                try:
                    d = json.loads(line)
                    if d['type'] == 'dahai' and d['actor'] == 3:
                        best_move = d['pai']
                        break
                except: continue
            
            return best_move, {"full_log": out[-100:]}

        except Exception as e:
            return f"Exception: {e}", {}

akochan_ai = AkochanBridge(AKOCHAN_EXE, AKOCHAN_CONF)

# ==================================================================================
#  👁️ Vision & Server (Minimal)
# ==================================================================================
class RobustGrid:
    # ... (前回のVisionコードと同じ。長くなるため省略しますが、必須です) ...
    # 既存の RobustGrid クラスをそのままここに維持してください
    @staticmethod
    def letterbox_image(img, target_size=(1920, 1080)):
        h, w = img.shape[:2]
        tw, th = target_size
        scale = min(tw/w, th/h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        dx = (tw - nw) // 2
        dy = (th - nh) // 2
        canvas[dy:dy+nh, dx:dx+nw] = resized
        return canvas

    @staticmethod
    def parse_frame_dual(original_img, boxes, names, img_w, img_h):
        # 以前のコードの RobustGrid.parse_frame_dual をそのまま使用
        # (省略なしで貼り付けてください)
        detections = []
        for box in boxes:
            cls_id = int(box.cls[0])
            label = names[cls_id]
            cx, cy, w, h = box.xywh[0].tolist()
            x1, y1 = max(0, int(cx - w/2)), max(0, int(cy - h/2))
            x2, y2 = min(img_w, int(cx + w/2)), min(img_h, int(cy + h/2))
            detections.append({'label': label, 'cx': cx, 'cy': cy, 'w': w, 'h': h, 'nx': cx/img_w, 'ny': cy/img_h, 'bbox': (x1, y1, x2, y2)})

        my_hand_objs, my_melds_objs, doras_objs = [], [], []
        opps = [{'river':[], 'river_data':[], 'melds':[], 'reach':False} for _ in range(4)]
        base_center_x = img_w * 0.5
        base_center_y = img_h * 0.42
        table_center = (int(base_center_x), int(base_center_y))
        R = img_h * 0.35
        bottom_anchor_y = base_center_y + R - (img_h * 0.06) 
        anchors = {1: (base_center_x, base_center_y - R), 2: (base_center_x - R, base_center_y), 0: (base_center_x + R, base_center_y), 3: (base_center_x, bottom_anchor_y)}

        if not detections: return my_hand_objs, my_melds_objs, doras_objs, opps, table_center, anchors

        river_candidates = []
        for t in detections:
            if t['nx'] < 0.20 and t['ny'] < 0.25: doras_objs.append(t); continue
            if t['cy'] > anchors[3][1]: my_hand_objs.append(t); continue
            if (0.18 < t['nx'] < 0.82) and (0.18 < t['ny'] < 0.82): river_candidates.append(t)
            else:
                if t['nx'] < 0.18: opps[2]['melds'].append(t['label'])
                elif t['nx'] > 0.82: opps[0]['melds'].append(t['label'])
                elif t['ny'] < 0.18: opps[1]['melds'].append(t['label'])
                else: river_candidates.append(t)

        if my_hand_objs:
            my_hand_objs.sort(key=lambda x: x['nx'])
            if len(my_hand_objs) > 14: my_hand_objs = my_hand_objs[-14:]
            my_hand_objs = my_hand_objs

        for t in river_candidates:
            best_seat = -1
            min_dist = float('inf')
            for seat, (ax, ay) in anchors.items():
                dist = math.sqrt((t['cx'] - ax)**2 + (t['cy'] - ay)**2)
                if dist < min_dist: min_dist = dist; best_seat = seat
            if best_seat == 3 and t['cy'] > anchors[3][1]: continue
            if best_seat != -1:
                opps[best_seat]['river'].append(t['label'])
                opps[best_seat]['river_data'].append(t)

        for seat in range(4):
            river = opps[seat]['river_data']
            if not river: continue
            ratios = [(t['w'] / t['h']) for t in river if t['h'] > 0]
            med_ratio = np.median(ratios) if ratios else 1.0
            for t in river:
                w, h = t['w'], t['h']
                current_ratio = w / h if h > 0 else 1.0
                is_reach = False
                if seat == 1 and current_ratio > 1.25: is_reach = True
                elif (seat == 0 or seat == 2) and current_ratio < 0.9: is_reach = True
                elif seat == 3 and current_ratio > 1.15: is_reach = True
                if is_reach: opps[seat]['reach'] = True; break

        return my_hand_objs, my_melds_objs, doras_objs, opps, table_center, anchors

class ResultFormatter:
    @staticmethod
    def clean(hand_objs, melds_objs, doras_objs):
        hand = [t['label'] for t in hand_objs]
        melds = [t['label'] for t in melds_objs]
        doras = [t['label'] for t in doras_objs]
        if len(hand) > 14: hand = hand[:14]
        def fix(l): return [('5'+t[1] if t.startswith('0') else t) for t in l]
        return fix(hand), fix(melds), fix(doras)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files: return "No", 400
    file = request.files['file']
    in_mem = io.BytesIO()
    file.save(in_mem)
    data = np.frombuffer(in_mem.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    
    img_std = RobustGrid.letterbox_image(img, target_size=(STD_W, STD_H))
    results = model_main.predict(source=img_std, imgsz=1280, conf=0.1, iou=0.45, augment=True, verbose=False)
    
    hand_objs, melds_objs, doras_objs, opps, table_center, anchors = RobustGrid.parse_frame_dual(
        img_std, results[0].boxes, model_main.names, STD_W, STD_H
    )
    
    cv2.imwrite(debug_save_path, img_std) # Debug保存
    hand, melds, indicators = ResultFormatter.clean(hand_objs, melds_objs, doras_objs)
    
    # 🧠 Akochan思考
    result_txt = "待機中..."
    if len(hand) % 3 == 2:
        print("🤔 Akochan Thinking...")
        tile, info = akochan_ai.think(hand, melds, indicators, opps)
        result_txt = f"打: {tile}"
        if "Error" in str(tile): result_txt = tile
    
    response = {
        "result": result_txt,
        "my_hand": hand,
        "opponents": [{"seat": i, "reach": opps[i]['reach']} for i in range(4)]
    }
    
    print("\n" + "="*50)
    print(f"🀄 手牌: {hand}")
    print(f"🤖 Akochan: {result_txt}")
    print("="*50)
    
    return jsonify(response)

if __name__ == '__main__':
    print("🚀 Akochan AI Server Started.")
    if not os.path.exists(AKOCHAN_EXE): print(f"❌ Error: {AKOCHAN_EXE} not found")
    app.run(host='0.0.0.0', port=5000)