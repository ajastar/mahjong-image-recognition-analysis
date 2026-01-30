from ultralytics import YOLO

# 学習したモデルをロード
model = YOLO(r"E:\AI_Project_Hub\Mahjong_Maker\runs\detect\jantama_aug_final\weights\best.pt")

# テストしたい画像の場所
source = r"E:\AI_Project_Hub\Mahjong_Maker\dataset_learning\val\images"

# 予測実行
results = model.predict(
    source=source,
    save=True,           # 結果を保存する
    imgsz=1280,          # 高画質で処理
    conf=0.5,            # 確信度50%以上のものだけ表示
    
    # --- ここで見た目を調整します ---
    line_width=1,        # 枠の線の太さ（1が最小。デフォルトは3程度）
    show_labels=True,    # ラベル名（1mなど）を表示する
    show_conf=False      # 0.99 などの「確率」を非表示にしてスッキリさせる
)

print(f"✅ 完了！ 'runs/detect/predict' フォルダの中身を確認してください。")