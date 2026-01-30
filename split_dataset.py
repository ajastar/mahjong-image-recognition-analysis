import os
import shutil
import random

# ================= 設定 =================
# 1. データの収集元 (アノテーション済みの場所)
SOURCE_DIR = "dataset_collector"

# 2. データの出力先 (学習用に整理する場所)
# ※実行するたびに中身を作り直します
DEST_DIR = "dataset_learning"

# 3. テストデータの割合 (0.2 = 20%をテストに回す)
VAL_RATIO = 0.2
# ========================================

def split_data():
    # パスの定義
    src_images_dir = os.path.join(SOURCE_DIR, "images")
    src_labels_dir = os.path.join(SOURCE_DIR, "labels")

    # もし学習用フォルダが既にあったら、一度消して作り直す (古いデータの混入防止)
    if os.path.exists(DEST_DIR):
        print(f"🧹 古いフォルダ '{DEST_DIR}' を掃除しています...")
        shutil.rmtree(DEST_DIR)

    # 新しいフォルダ構造を作成
    for split in ["train", "val"]:
        os.makedirs(os.path.join(DEST_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(DEST_DIR, split, "labels"), exist_ok=True)

    # 画像リストを取得
    if not os.path.exists(src_images_dir):
        print(f"❌ エラー: '{src_images_dir}' が見つかりません。")
        return

    all_images = [f for f in os.listdir(src_images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if not all_images:
        print("❌ 画像ファイルが見つかりません。")
        return

    # シャッフルして分割
    random.shuffle(all_images)
    split_idx = int(len(all_images) * (1 - VAL_RATIO))
    train_files = all_images[:split_idx]
    val_files = all_images[split_idx:]

    print(f"🚀 全 {len(all_images)} 枚のデータを振り分けます...")
    print(f"   - 学習用 (train): {len(train_files)} 枚")
    print(f"   - テスト用 (val)  : {len(val_files)} 枚")

    # コピー実行関数
    def copy_files(file_list, split_type):
        for img_file in file_list:
            base_name = os.path.splitext(img_file)[0]
            txt_file = base_name + ".txt"

            # 1. 画像コピー
            src_img = os.path.join(src_images_dir, img_file)
            dst_img = os.path.join(DEST_DIR, split_type, "images", img_file)
            shutil.copy2(src_img, dst_img)

            # 2. ラベルコピー (存在する場合のみ)
            src_lbl = os.path.join(src_labels_dir, txt_file)
            dst_lbl = os.path.join(DEST_DIR, split_type, "labels", txt_file)
            
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, dst_lbl)

    copy_files(train_files, "train")
    copy_files(val_files, "val")

    print(f"\n✅ 完了！ '{DEST_DIR}' にデータの準備ができました。")
    print("👉 次は 'train_yolo.py' を実行して学習を開始してください！")

if __name__ == "__main__":
    split_data()