import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import shutil
import cv2
import numpy as np

# --- 設定 ---
DATA_DIR = "crop_dataset"        # 画像データの場所
MODEL_PATH = "rotation_net.pth"  # 学習済みモデル
OUTPUT_DIR = "debug_errors"      # 間違えた画像を保存する場所

# --- 設定終わり ---

def diagnose():
    # 1. 準備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 診断開始 (Device: {device})")

    # 保存先リセット
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    # 2. データ読み込み (学習時と同じ前処理)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    except:
        print(f"❌ エラー: '{DATA_DIR}' が見つかりません。")
        return

    # クラス名の確認
    class_names = dataset.classes
    print(f"ℹ️ クラス定義: {dataset.class_to_idx}")
    # 例: {'horizontal': 0, 'vertical': 1}

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 3. モデルロード (ResNet50)
    try:
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(class_names))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"❌ モデル読み込みエラー: {e}")
        print("※ ResNet34で学習した場合は、コード内の resnet50 を resnet34 に書き換えてください。")
        return

    # 4. 全画像テスト
    print("🚀 テスト実行中...")
    total = 0
    correct = 0
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 推論
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1) # 確率計算
            conf, predicted = torch.max(probs, 1)
            
            label_idx = labels.item()
            pred_idx = predicted.item()
            
            total += 1
            
            # 正解・不正解チェック
            if label_idx == pred_idx:
                correct += 1
            else:
                # ❌ 間違えた場合: 画像を保存して確認できるようにする
                # 元画像のパスを取得
                img_path = dataset.samples[i][0]
                img_name = os.path.basename(img_path)
                
                true_label = class_names[label_idx]
                pred_label = class_names[pred_idx]
                confidence = conf.item() * 100
                
                # ファイル名に詳細を含める
                # 例: 誤_pred_horizontal(99%)_true_vertical_image01.jpg
                new_name = f"Err_Pred[{pred_label}-{confidence:.0f}%]_True[{true_label}]_{img_name}"
                dst_path = os.path.join(OUTPUT_DIR, new_name)
                
                shutil.copy(img_path, dst_path)
                print(f"  ⚠️ ミス: {img_name} -> 予測: {pred_label} ({confidence:.1f}%) / 正解: {true_label}")

    # 5. 結果発表
    accuracy = 100 * correct / total
    print("-" * 50)
    print(f"📊 総合精度: {accuracy:.2f}% ({correct}/{total})")
    print(f"📂 間違えた画像は '{OUTPUT_DIR}' フォルダに保存しました。")
    
    if accuracy > 95:
        print("✅ モデルの性能は良好です。")
        print("   もし実戦で間違えるなら、学習データと実戦画像の「見た目（明るさ・画質）」が違いすぎます。")
    elif accuracy < 60:
        print("❌ 精度が低すぎます。ラベル定義が逆転しているか、学習に失敗しています。")
    else:
        print("⚠️ 過学習の疑いがあります。ResNet50は大きすぎるかもしれません。")

if __name__ == '__main__':
    diagnose()