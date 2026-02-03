import os
import cv2
import numpy as np
import glob
import random

def main():
    # ---------------------------------------------------------
    # 工場設定
    # ---------------------------------------------------------
    base_dir = r"E:\AI_Project_Hub\Mahjong_Maker"
    assets_dir = os.path.join(base_dir, r"assets_tiles")
    
    # 生成データを保存する場所
    output_img_dir = os.path.join(base_dir, r"dataset_synthetic\train\images")
    output_lbl_dir = os.path.join(base_dir, r"dataset_synthetic\train\labels")
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_lbl_dir, exist_ok=True)

    # 背景として使う画像のフォルダ (raw_imagesから1枚拝借するか、適当な画像を使う)
    # ここでは便宜上、黒い背景を自動生成します
    
    GENERATE_COUNT = 5000  # 何枚生成するか (ここで数を増やせます)
    MAX_TILES_PER_IMG = 14 # 1枚の画像に置く牌の最大数

    # クラスリスト
    class_names = [
        "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
        "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
        "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
        "east", "south", "west", "north",
        "haku", "hatsu", "chun",
        "5mr", "5pr", "5sr", "ura"
    ]
    
    # 素材のロード
    tile_assets = {}
    for i, name in enumerate(class_names):
        p = os.path.join(assets_dir, name, "*.png")
        files = glob.glob(p)
        if files:
            tile_assets[i] = [cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in files]
    
    print(f"[Info] 合成画像の生成を開始します。目標: {GENERATE_COUNT}枚")

    for img_idx in range(GENERATE_COUNT):
        # 1. 背景の作成 (1024x1024 の麻雀マットっぽい色)
        bg_h, bg_w = 1024, 1024
        # マット色 (青緑〜濃い緑のランダム)
        bg_color = (random.randint(50, 100), random.randint(50, 100), random.randint(30, 60)) 
        background = np.full((bg_h, bg_w, 3), bg_color, dtype=np.uint8)
        
        labels = []
        
        # 2. 牌を配置
        num_tiles = random.randint(5, MAX_TILES_PER_IMG)
        
        # 利用可能なクラスIDからランダムに選択
        available_classes = [k for k, v in tile_assets.items() if len(v) > 0]
        if not available_classes:
            print("[Error] 素材がありません。harvest_tiles.py を先に実行してください。")
            return

        for _ in range(num_tiles):
            cls_id = random.choice(available_classes)
            tile_img = random.choice(tile_assets[cls_id])
            
            if tile_img is None: continue

            # ランダム回転 (0〜360度)
            angle = random.uniform(0, 360)
            scale = random.uniform(0.8, 1.2) # 少し大きさを変える
            
            # 画像の回転処理
            (h, w) = tile_img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            
            rotated_tile = cv2.warpAffine(tile_img, M, (new_w, new_h))
            
            # 配置位置 (はみ出さないように)
            if bg_w - new_w <= 0 or bg_h - new_h <= 0: continue
            
            pos_x = random.randint(0, bg_w - new_w)
            pos_y = random.randint(0, bg_h - new_h)
            
            # 貼り付け (単純な上書き)
            # ※ 本来はマスク処理をして背景となじませるべきですが、YOLO学習用ならベタ張りでも効果あり
            roi = background[pos_y:pos_y+new_h, pos_x:pos_x+new_w]
            
            # アルファチャンネルがないのでそのまま上書き
            # (端っこの黒い部分を簡易マスクとして抜く処理)
            gray_tile = cv2.cvtColor(rotated_tile, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_tile, 1, 255, cv2.THRESH_BINARY)
            
            # マスクを使って合成
            mask_inv = cv2.bitwise_not(mask)
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            img_fg = cv2.bitwise_and(rotated_tile, rotated_tile, mask=mask)
            dst = cv2.add(img_bg, img_fg)
            background[pos_y:pos_y+new_h, pos_x:pos_x+new_w] = dst
            
            # ラベル生成 (OBB形式: class x1 y1 x2 y2 x3 y3 x4 y4)
            # 回転した矩形の4隅の座標を計算
            rect_points = cv2.boxPoints(((pos_x + new_w/2, pos_y + new_h/2), (w*scale, h*scale), angle))
            
            # 正規化
            norm_points = []
            for pt in rect_points:
                norm_points.append(pt[0] / bg_w)
                norm_points.append(pt[1] / bg_h)
            
            label_str = f"{cls_id} " + " ".join([f"{x:.6f}" for x in norm_points])
            labels.append(label_str)

        # 保存
        file_name = f"synthetic_{img_idx:05d}"
        cv2.imwrite(os.path.join(output_img_dir, file_name + ".jpg"), background)
        with open(os.path.join(output_lbl_dir, file_name + ".txt"), "w") as f:
            f.write("\n".join(labels))
            
        if img_idx % 50 == 0:
            print(f"[Generatimg] {img_idx}/{GENERATE_COUNT} 枚生成完了...")

    print(f"[Success] {GENERATE_COUNT} 枚の合成学習データを生成しました！")
    print(f"場所: {os.path.join(base_dir, 'dataset_synthetic')}")

if __name__ == "__main__":
    main()