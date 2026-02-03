# 🀄 Mahjong Maker (Jantama AI)

**Advanced AI for Mahjong Soul (Jantama) / 雀魂（じゃんたま）用 高性能AI**

[English](#english) | [日本語](#japanese)

---

<a name="english"></a>
## 🇬🇧 English Documentation

This repository contains an advanced AI for **Mahjong Soul (Jantama)**, featuring computer vision (YOLO) and a deep reinforcement learning brain (GodBrain).

### 📂 Project Structure

- **`single_bot.py`** / **`start_all.bat`**: The core automation bot. It captures the game window, analyzes tiles, and plays automatically.
- **`server.py`** / **`start_ai.bat`**: A backend server (Flask) for visualization, debugging, or separating the brain calculation from the client.
- **`run_all_jantama.py`**: The master training script. Used for generating synthetic training data and fine-tuning the YOLO model.
- **`mahjong_logic.py`**: Contains the rules needed for shanten calculation, score estimation, and yaku verification.
- **`mahjong_factory.py`**: A utility for generating unlimited synthetic training images of mahjong tiles.

### 🚀 Getting Started

#### 1. Prerequisites
- **OS**: Windows (Required for `single_bot.py` screen capture)
- **GPU**: NVIDIA GPU with CUDA installed (Recommended for performance)
- **Python**: 3.10 or newer

#### 2. Installation
Run the following command to install the required dependencies:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics numpy opencv-python flask mss pygetwindow easyocr pyyaml
```
*(Note: Adjust the PyTorch installation command based on your CUDA version from [pytorch.org](https://pytorch.org/).)*

### 🎮 How to Use

#### A. Run the Auto-Bot (`single_bot.py`)
To start the automated player:
1. Open **Mahjong Soul** (e.g., in LDPlayer or Chrome).
2. Run **`start_all.bat`**.
   - This script generates multiple bot instances.
   - You can edit `start_all.bat` or `single_bot.py` to target specific window titles.

#### B. Run the AI Server (`server.py`)
To run the AI logic as a server (for debugging or separating load):
1. Run **`start_ai.bat`**.
2. The server will start on `localhost:5000`.

### 🧠 Training (Retraining the Vision)
If you want to improve the AI's ability to recognize tiles:
1. **Place Source Images**: Ensure `jantama_assets` contains your clean tile images.
2. **Run Training**:
   ```bash
   python run_all_jantama.py
   ```
   - This script cleans up datasets, generates synthetic data, and trains the YOLO model.

---

<a name="japanese"></a>
## 🇯🇵 日本語ドキュメント

**Mahjong Maker** は、コンピュータビジョン (YOLO) と深層強化学習モデル (GodBrain) を搭載した、**雀魂**用の高度な自動AIです。

### 📂 プロジェクト構成

- **`single_bot.py`** / **`start_all.bat`**: 全自動ボットの本体です。画面をキャプチャし、牌を認識して自動で操作を行います。
- **`server.py`** / **`start_ai.bat`**: AIサーバー（バックエンド）です。思考エンジンの可視化や、計算負荷の分散に使用します。
- **`run_all_jantama.py`**: 学習用スクリプトです。学習データの自動生成からYOLOモデルの学習までを一括で行います。
- **`mahjong_logic.py`**: 局進行、シャンテン数計算、点数計算などの麻雀ルールロジックです。
- **`mahjong_factory.py`**: 学習用の麻雀牌画像を無限に生成するツールです。

### 🚀 はじめに

#### 1. 動作環境
- **OS**: Windows（画面キャプチャ及び操作のため必須）
- **GPU**: NVIDIA GPU / CUDA推奨（AIの高速推論のため）
- **Python**: 3.10 以上

#### 2. インストール
以下のコマンドで必要なライブラリをインストールしてください：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics numpy opencv-python flask mss pygetwindow easyocr pyyaml
```
*(注意: PyTorchのコマンドはお使いのCUDAバージョンに合わせて [pytorch.org](https://pytorch.org/) で確認してください)*

### 🎮 使い方

#### A. 自動ボットの起動 (`single_bot.py`)
自動打ちを開始するには：
1. **雀魂**を開きます（LDPlayer や Chromeブラウザなど）。
2. **`start_all.bat`** を実行します。
   - 複数のボットインスタンスが起動します。
   - ウィンドウタイトル名などの設定は `single_bot.py` 内の「⚙️ 設定」で変更可能です。

#### B. AIサーバーの起動 (`server.py`)
AIの思考ロジックのみをサーバーとして起動する場合：
1. **`start_ai.bat`** を実行します。
2. `localhost:5000` でサーバーが待機状態になります。

### 🧠 再学習（画像認識の精度向上）
牌のスキンを変更したり、認識精度を上げたい場合：
1. **素材の準備**: `jantama_assets` フォルダに牌の画像（png/jpg）を配置します。
2. **学習の実行**:
   ```bash
   python run_all_jantama.py
   ```
   - このスクリプトを実行すると、自動的にデータの掃除、合成データの生成、そしてYOLOモデルの学習が行われます。

## 📝 License
[MIT License](LICENSE)
