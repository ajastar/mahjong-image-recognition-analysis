from ultralytics import YOLO
import multiprocessing

# ==========================================
# 🔥 データ拡張（Augmentation）ハイパーパラメータ設定 🔥
# ここで「どれくらい画像をいじめるか」を決めます
# ==========================================
HYPS = {
    # --- 幾何学的変換（回転・ズレ） ---
    'degrees': 15.0,      # 回転 (±15度): カメラの傾きに対応
    'translate': 0.1,     # 平行移動 (±10%): フレーム内の位置ズレに対応
    'scale': 0.5,         # 拡大縮小 (±50%): 牌の大きさの違いに対応
    'shear': 2.0,         # せん断 (斜めに歪ませる): 遠近感に対応
    'perspective': 0.0,   # 透視投影: 麻雀ではあまり極端なのは不要なので0
    'flipud': 0.5,        # 上下反転 (確率50%): 捨て牌は逆さになることがあるので有効
    'fliplr': 0.5,        # 左右反転 (確率50%): 鏡像。これも有効

    # --- 色・ノイズ変換 ---
    'hsv_h': 0.015,       # 色相の変化: 卓の色被りや照明の違いに対応
    'hsv_s': 0.4,         # 彩度の変化: 鮮やかさの違いに対応
    'hsv_v': 0.4,         # 明度の変化: 暗い部屋/明るい部屋に対応
    
    # --- 特殊効果（YOLO強力機能） ---
    'mosaic': 1.0,        # モザイク (確率100%): 4枚を合成。絶対にON！
    'mixup': 0.1,         # MixUp (確率10%): 画像を半透明に重ねる。ごちゃついた場に強くなる
    'copy_paste': 0.1,    # コピー＆ペースト (確率10%): 牌を切り抜いて別の場所に貼る
}
# ==========================================


if __name__ == '__main__':
    multiprocessing.freeze_support()

    # 1. 先生役のモデル (前回の学習結果のベストモデルを指定)
    # ※パスが正しいか必ず確認してください！
    model_path = r"E:\AI_Project_Hub\Mahjong_Maker\runs\jantama_absolute_limit_1280px\weights\best.pt"
    print(f"🧠 モデルをロードします: {model_path}")
    model = YOLO(model_path)

    # 2. 学習開始 (ハイパーパラメータを適用)
    print("🚀 データ拡張をフル活用して学習を開始します... (imgsz=1280)")
    
    # train関数に **HYPS を渡すことで設定を適用します
    results = model.train(
        data="mahjong.yaml",  # 設定ファイル
        epochs=100,           # 【変更】データが増えたようなものなので、回数を増やします(50->100)
        imgsz=1280,           # 高画質で学習
        batch=2,              # メモリ不足なら2に下げる
        name="jantama_aug_final", # 今回の名前(Augmentation Final)
        device=0,             # GPUを使用
        exist_ok=True,        # 上書き許可
        
        **HYPS                # 🔥 ここで上の設定を全部流し込みます 🔥
    )
    print("🎉 学習完了！")