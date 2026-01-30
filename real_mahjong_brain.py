import jax
import jax.numpy as jnp
import equinox as eqx
import pickle
import numpy as np

# ==============================================================================
# 🧠 AIモデル定義 (学習コードと同じものを定義する必要があります)
# ==============================================================================
ACTION_SIZE = 181 

class MahjongNet(eqx.Module):
    conv1: eqx.nn.Conv2d
    norm1: eqx.nn.GroupNorm
    conv2: eqx.nn.Conv2d
    norm2: eqx.nn.GroupNorm
    conv3: eqx.nn.Conv2d
    norm3: eqx.nn.GroupNorm
    flatten: eqx.nn.Linear
    actor_head: eqx.nn.Linear
    critic_head: eqx.nn.Linear

    def __init__(self, key):
        k1, k2, k3, kf, ka, kc = jax.random.split(key, 6)
        C_in = 62 
        self.conv1 = eqx.nn.Conv2d(C_in, 64, kernel_size=(3, 1), padding="SAME", key=k1)
        self.norm1 = eqx.nn.GroupNorm(groups=8, channels=64)
        self.conv2 = eqx.nn.Conv2d(64, 128, kernel_size=(3, 1), padding="SAME", key=k2)
        self.norm2 = eqx.nn.GroupNorm(groups=8, channels=128)
        self.conv3 = eqx.nn.Conv2d(128, 128, kernel_size=(3, 1), padding="SAME", key=k3)
        self.norm3 = eqx.nn.GroupNorm(groups=8, channels=128)
        self.flatten = eqx.nn.Linear(128 * 34 * 4, 512, key=kf)
        self.actor_head = eqx.nn.Linear(512, ACTION_SIZE, key=ka)
        self.critic_head = eqx.nn.Linear(512, 1, key=kc)

    def __call__(self, x):
        x = jnp.transpose(x, (2, 0, 1)) # (H, W, C) -> (C, H, W)
        x = jax.nn.relu(self.norm1(self.conv1(x)))
        x = jax.nn.relu(self.norm2(self.conv2(x)))
        x = jax.nn.relu(self.norm3(self.conv3(x)))
        x = x.reshape(-1)
        x = jax.nn.relu(self.flatten(x))
        return self.actor_head(x), self.critic_head(x)[0]

# ==============================================================================
# 🌉 Brain Bridge: JSON to Tensor 変換器
# ==============================================================================

# タイルID変換マップ (MJx / Suphx準拠)
TILE_MAP = {
    "1m":0, "2m":1, "3m":2, "4m":3, "5m":4, "6m":5, "7m":6, "8m":7, "9m":8,
    "1p":9, "2p":10, "3p":11, "4p":12, "5p":13, "6p":14, "7p":15, "8p":16, "9p":17,
    "1s":18, "2s":19, "3s":20, "4s":21, "5s":22, "6s":23, "7s":24, "8s":25, "9s":26,
    "1z":27, "2z":28, "3z":29, "4z":30, "5z":31, "6z":32, "7z":33,
    # 赤ドラは通常牌として扱う (5m, 5p, 5s)
    "0m":4, "0p":13, "0s":22, "5mr":4, "5pr":13, "5sr":22
}
INV_TILE_MAP = {v: k for k, v in TILE_MAP.items()}
# 赤ドラの表示用修正
INV_TILE_MAP[4] = "5m"
INV_TILE_MAP[13] = "5p"
INV_TILE_MAP[22] = "5s"

class RealMahjongBrain:
    def __init__(self, model_path="mahjong_riichi_model.pkl"):
        print("🧠 Loading Mahjong Brain...")
        # モデル初期化
        self.model = MahjongNet(key=jax.random.PRNGKey(0))
        # 学習済み重みのロード
        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            self.inference_fn = eqx.filter_jit(self.model)
            print("✅ Brain Loaded Successfully!")
        except FileNotFoundError:
            print("⚠️ Model file not found. Using random weights (DEBUG MODE).")
            self.inference_fn = eqx.filter_jit(self.model)

    def _json_to_tensor(self, data):
        """
        Vision AIのJSON出力を、MJx(Suphx互換)の (34, 4, 62) 特徴量テンソルに変換する
        """
        # 34種, 4枚, 62チャンネル
        features = np.zeros((34, 4, 62), dtype=np.float32)

        # -----------------------------------------------------
        # 1. 自分の手牌 (Channels 0-3)
        # -----------------------------------------------------
        hand_tiles = data.get("my_hand", [])
        counts = np.zeros(34, dtype=int)
        for t in hand_tiles:
            if t in TILE_MAP:
                counts[TILE_MAP[t]] += 1
        
        for t_idx in range(34):
            c = counts[t_idx]
            for i in range(4):
                if c > i:
                    features[t_idx, :, i] = 1.0

        # -----------------------------------------------------
        # 2. ドラ表示牌 (Suphx仕様では Channel 50周辺だが簡易実装)
        # -----------------------------------------------------
        # ※本来のMJxの特徴量生成はC++内部で行われるため、
        # 完全再現にはMJxのStateを復元する必要がありますが、
        # ここでは「手牌」と「安全度」を重視した簡易マッピングを行います。
        # チャンネル4にドラ情報を注入（簡易）
        dora_indicators = data.get("dora_indicators", [])
        for d in dora_indicators:
            if d in TILE_MAP:
                # 表示牌の次がドラだが、ここでは簡易的に表示牌の位置にフラグを立てる
                idx = TILE_MAP[d]
                features[idx, :, 4] = 1.0

        # -----------------------------------------------------
        # 3. 他家のリーチ状況 (Channels ? -> 簡易的に全体にバイアス)
        # -----------------------------------------------------
        # リーチに対して危険牌を切らないようにするロジックは
        # 本来モデルが学習していますが、入力テンソルにリーチ情報がないと判断できません。
        # ここでは「他家の河」情報を使って再現を試みます。
        
        opponents = data.get("opponents", [])
        # 座席変換: VisionAI [0:下家, 1:対面, 2:上家, 3:自分]
        # MJxの特徴量は相対座標 (下家=1, 対面=2, 上家=3)
        
        seat_map_vision_to_mjx = {0: 1, 1: 2, 2: 3} # 自分(3)は除外

        for opp in opponents:
            seat_name = opp["seat"]
            is_reach = opp["reach"]
            river = opp["river"]
            
            # 座席特定
            target_rel_idx = -1
            if "Shimoch" in seat_name: target_rel_idx = 1
            elif "Toimen" in seat_name: target_rel_idx = 2
            elif "Kamicha" in seat_name: target_rel_idx = 3
            
            if target_rel_idx != -1:
                # リーチしている場合、特定のチャンネルを埋める（MJx仕様に合わせるのが理想だが...）
                # 今回は、モデルが「リーチ者の現物」を学習していることを期待し、
                # 河の情報をセットします。
                
                # 河の登録 (Channels 10-20あたりが捨て牌履歴)
                # 簡易的に Channel 10 + target_rel_idx に書き込む
                ch_offset = 10 + target_rel_idx * 2 
                for tile in river:
                    if tile in TILE_MAP:
                        features[TILE_MAP[tile], :, ch_offset] = 1.0
                
                if is_reach:
                    # リーチフラグ (Channel 55あたりと仮定、または全捨て牌を強調)
                    # ここは学習時の正確なfeature_type="suphx"の実装依存ですが、
                    # 手牌進行（攻撃）に関しては手牌情報だけで9割決まるため、
                    # リーチに対するベタオリ判断は別途ロジック補助しても良いです。
                    pass

        return jnp.array(features)

    def think(self, json_data):
        """
        JSONを受け取り、最適なアクション（切る牌）と勝率を返す
        """
        # 1. テンソル作成
        obs_tensor = self._json_to_tensor(json_data)

        # 2. 推論実行
        logits, value = self.inference_fn(obs_tensor)
        
        # 3. マスク処理（持っていない牌は切れない）
        # アクションID 0~33 が打牌。
        valid_mask = -1e9 * jnp.ones(ACTION_SIZE)
        
        # 手牌にある牌だけマスクを外す (0.0にする)
        hand_tiles = json_data.get("my_hand", [])
        has_valid_move = False
        
        for t in hand_tiles:
            if t in TILE_MAP:
                idx = TILE_MAP[t]
                valid_mask = valid_mask.at[idx].set(0.0) # 打牌アクション
                # ツモ切り(idx+34)等は今回省略し、手出しのみ考慮
                has_valid_move = True
        
        if not has_valid_move:
            return "None", 0.0 # エラーハンドリング

        # 4. ロジットにマスク適用
        masked_logits = logits + valid_mask
        
        # 5. 最善手選択
        best_action_idx = jnp.argmax(masked_logits)
        win_rate = jnp.tanh(value)

        # 6. アクションIDを文字列に戻す
        idx = int(best_action_idx)
        if 0 <= idx <= 33:
            tile_name = INV_TILE_MAP.get(idx, "?")
            action_str = f"打 {tile_name}"
        else:
            action_str = f"Action_{idx}" # リーチや鳴きなど

        return action_str, float(win_rate)

if __name__ == "__main__":
    # テスト用
    brain = RealMahjongBrain()
    dummy_data = {
        "my_hand": ["1m", "2m", "3m", "5p", "6p", "7p", "1s", "1s", "8s", "9s", "1z", "1z", "5z"],
        "dora_indicators": ["2z"],
        "opponents": []
    }
    act, rate = brain.think(dummy_data)
    print(f"Decision: {act}, WinRate: {rate:.2f}")