import os
import cv2
from ultralytics import YOLO

# --- 設定 ---
# 【重要】ここに「牌の画像」が入っているフォルダの絶対パスを貼ってください
# 例: r"E:\AI_Project_Hub\Mahjong_Maker\my_dataset\train\images"
IMAGE_DIR = r"E:\AI_Project_Hub\Mahjong_Maker\dataset_roboflow_public\train\images" 

# 切り抜いた画像を保存するフォルダ（自動で作られます）
OUTPUT_DIR = "crop_dataset"

# あなたの作成したYOLOモデルのパス
MODEL_PATH = r"runs\jantama_absolute_limit_1280px\weights\best.pt"

# --------------------------

# モデル読み込み
if not os.path.exists(MODEL_PATH):
    print(f"エラー: モデルファイルが見つかりません -> {MODEL_PATH}")
    exit()

model = YOLO(MODEL_PATH)
os.makedirs(f"{OUTPUT_DIR}/all", exist_ok=True)

# 画像フォルダの確認
if not os.path.exists(IMAGE_DIR):
    print(f"エラー: 画像フォルダが見つかりません -> {IMAGE_DIR}")
    print("スクリプト内の 'IMAGE_DIR' を、正しい画像フォルダのパスに書き換えてください。")
    exit()

count = 0
print(f"処理開始: {IMAGE_DIR} から画像を読み込みます...")

for filename in os.listdir(IMAGE_DIR):
    if not filename.lower().endswith(('.jpg', '.png', '.jpeg')): continue
    
    img_path = os.path.join(IMAGE_DIR, filename)
    img = cv2.imread(img_path)
    if img is None: continue
    
    # 推論実行
    results = model(img, verbose=False)
    
    for box in results[0].boxes:
        # 座標取得
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        # 画像範囲内に収める
        h, w, _ = img.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # 切り抜き
        crop = img[y1:y2, x1:x2]
        
        if crop.size == 0: continue
        
        # 保存
        cv2.imwrite(f"{OUTPUT_DIR}/all/tile_{count}.jpg", crop)
        count += 1
        
print(f"完了: {count}枚の牌画像を切り抜きました。")
print(f"保存先: {os.path.abspath(OUTPUT_DIR)}/all")