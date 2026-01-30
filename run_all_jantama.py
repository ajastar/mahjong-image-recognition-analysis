import os
import shutil
import random
import math
import glob
from PIL import Image
import yaml
import sys
from ultralytics import YOLO
import torch

# ==========================================
# 設定エリア
# ==========================================
BASE_DIR = r"E:\AI_Project_Hub\Mahjong_Maker"
ASSETS_DIR = os.path.join(BASE_DIR, "jantama_assets")
DATASET_DIR = os.path.join(BASE_DIR, "dataset_jantama_synthetic")

# ==========================================
# 1. 強力クリーンアップ機能
# ==========================================
def cleanup():
    print("-" * 50)
    print("【工程1】データとキャッシュの完全消去")
    
    # データセットフォルダ削除
    if os.path.exists(DATASET_DIR):
        try:
            shutil.rmtree(DATASET_DIR)
            print(f"[削除] データフォルダを削除: {DATASET_DIR}")
        except Exception as e:
            print(f"[Error] フォルダ削除エラー: {e}")
            sys.exit()

    # ★最重要: 悪さをする .cache ファイルをプロジェクト全体から探して消す
    print("[掃除] キャッシュファイルを検索して削除中...")
    cache_files = glob.glob(os.path.join(BASE_DIR, "**", "*.cache"), recursive=True)
    for cf in cache_files:
        try:
            os.remove(cf)
            print(f"   -> 削除: {cf}")
        except:
            pass

# ==========================================
# 2. データ生成機能 (ポリゴン形式に変更)
# ==========================================
def get_box_corners(cx, cy, w, h, angle_deg):
    """
    中心座標とサイズ、角度から、4つの角の座標(x1,y1...x4,y4)を計算する関数
    これによりYOLOが確実にデータを読み込めるようになります。
    """
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    
    # 4つの角のオフセット（中心からの距離）
    # 左上, 右上, 右下, 左下
    offsets = [
        (-w/2, -h/2),
        (w/2, -h/2),
        (w/2, h/2),
        (-w/2, h/2)
    ]
    
    corners = []
    for dx, dy in offsets:
        # 回転行列で計算
        rot_x = dx * cos_a - dy * sin_a
        rot_y = dx * sin_a + dy * cos_a
        # 中心座標を足して、実際の座標にする
        corners.append(cx + rot_x)
        corners.append(cy + rot_y)
        
    return corners # [x1, y1, x2, y2, x3, y3, x4, y4]

def generate_data():
    print("-" * 50)
    print("【工程2】学習データの生成 (ポリゴン形式)")
    
    num_images = 5000
    img_size = 1024
    min_tiles = 10
    max_tiles = 30
    
    class_names = [
        "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
        "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
        "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
        "east", "south", "west", "north",
        "haku", "hatsu", "chun",
        "5mr", "5pr", "5sr",
        "ura"
    ]

    images_dir = os.path.join(DATASET_DIR, "train", "images")
    labels_dir = os.path.join(DATASET_DIR, "train", "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # data.yaml 生成
    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    yaml_data = {
        'path': DATASET_DIR,
        'train': 'train/images',
        'val': 'train/images',
        'names': {i: name for i, name in enumerate(class_names)}
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)

    # 素材読み込み
    tile_images = {}
    print(f"[読込] 素材フォルダ: {ASSETS_DIR}")
    
    for cls_name in class_names:
        path_png = os.path.join(ASSETS_DIR, f"{cls_name}.png")
        path_jpg = os.path.join(ASSETS_DIR, f"{cls_name}.jpg")
        target = path_png if os.path.exists(path_png) else (path_jpg if os.path.exists(path_jpg) else None)
        
        if target:
            try:
                img = Image.open(target).convert("RGBA")
                tile_images[cls_name] = img
            except:
                pass
    
    if not tile_images:
        print("[Fatal] 素材が読み込めません。")
        sys.exit()

    # 生成ループ
    print("[生成] 画像生成中...")
    for i in range(num_images):
        bg_color = (random.randint(0, 50), random.randint(60, 150), random.randint(60, 120))
        bg_img = Image.new("RGB", (img_size, img_size), bg_color)
        labels = []
        
        for _ in range(random.randint(min_tiles, max_tiles)):
            cls_name = random.choice(list(tile_images.keys()))
            tile_img = tile_images[cls_name].copy()
            
            angle = random.uniform(0, 360)
            rotated_img = tile_img.rotate(angle, expand=True, resample=Image.BICUBIC)
            
            w, h = rotated_img.size
            x = random.randint(-w//4, img_size - w + w//4)
            y = random.randint(-h//4, img_size - h + h//4)
            
            bg_img.paste(rotated_img, (x, y), rotated_img)
            
            # ★ここを変更: 5つの数字ではなく、8つの座標(ポリゴン)で出力する
            # これによりYOLOが「フォーマット不正」で弾くのを防ぎます
            cx_px = x + w/2
            cy_px = y + h/2
            # 回転前の元の幅と高さ（推測値だが合成なので近似でOK）
            # rotate(expand=True)しているので、w,hは大きくなっている。
            # 牌の元画像のサイズを使うのがベストだが、ここでは簡易的に縮小率を逆算
            # しかし計算が複雑になるため、バウンディングボックスの座標をそのまま正規化して渡す
            
            # シンプルかつ確実な方法: 
            # YOLO OBB Polygon形式: class x1 y1 x2 y2 x3 y3 x4 y4 (全て0-1正規化)
            corners = get_box_corners(cx_px, cy_px, tile_img.width, tile_img.height, angle)
            
            # 座標を0-1に正規化
            norm_corners = [c / img_size for c in corners]
            
            # 座標を文字列に変換 (class x1 y1 x2 y2 x3 y3 x4 y4)
            coords_str = " ".join([f"{c:.6f}" for c in norm_corners])
            labels.append(f"{class_names.index(cls_name)} {coords_str}")

        base_name = f"jantama_syn_{i:05d}"
        bg_img.save(os.path.join(images_dir, f"{base_name}.jpg"), quality=85)
        with open(os.path.join(labels_dir, f"{base_name}.txt"), "w") as f:
            f.write("\n".join(labels))
            
        if (i+1) % 1000 == 0:
            print(f"   -> {i+1} / {num_images} 完了")

# ==========================================
# 3. 学習機能 (実行)
# ==========================================
def train_model():
    print("-" * 50)
    print("【工程3】AI学習の開始")
    
    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    
    # GPUチェック
    if torch.cuda.is_available():
        print(f"[GPU] 使用デバイス: {torch.cuda.get_device_name(0)}")
    
    model = YOLO("yolo11l-obb.pt")
    
    # 念のためAMPはFalseにして安定動作を優先（速度は十分速い）
    model.train(
        data=yaml_path,
        epochs=30,
        imgsz=1024,
        device=0,
        batch=8,
        name="jantama_master",
        project=os.path.join(BASE_DIR, "runs"),
        exist_ok=True,
        
        # 高速化オプション
        cache=True, 
        workers=8,
        amp=True, # エラーが出たらFalseにする
        
        # デジタル画像用設定
        degrees=180.0,
        scale=0.5,
        mosaic=1.0,
        hsv_h=0.015, hsv_s=0.2, hsv_v=0.2,
        nbs=64
    )

if __name__ == '__main__':
    print("【開始】最終修正版スクリプトを起動します...")
    cleanup()
    generate_data()
    train_model()
    print("【完了】すべての工程が終了しました！")