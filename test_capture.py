import mss
import pygetwindow as gw
import cv2
import numpy as np
import time

# 動作確認したいウィンドウ名（あなたの環境で確認されたもの）
TARGET_TITLE = "LDPlayer" 

def test_capture():
    print(f"🔍 '{TARGET_TITLE}' を探しています...")
    
    # 1. ウィンドウ検索
    windows = gw.getWindowsWithTitle(TARGET_TITLE)
    if not windows:
        print(f"❌ 失敗: ウィンドウが見つかりません。")
        print("   -> LDPlayerが起動しているか、名前が正しいか確認してください。")
        return

    win = windows[0]
    print(f"✅ 発見: {win.title} (位置: {win.left},{win.top} サイズ: {win.width}x{win.height})")

    if win.isMinimized:
        print("⚠️ 警告: ウィンドウが最小化されています。復元を試みます...")
        win.restore()
        time.sleep(1)

    # 2. MSSでキャプチャ
    with mss.mss() as sct:
        monitor = {
            "top": win.top + 35, # タイトルバー分調整
            "left": win.left, 
            "width": win.width, 
            "height": win.height - 35
        }
        
        print("📸 キャプチャを実行します...")
        try:
            img = np.array(sct.grab(monitor))
            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # 保存して確認
            filename = "test_capture_result.jpg"
            cv2.imwrite(filename, img_bgr)
            print(f"🎉 成功: '{filename}' を保存しました。")
            print("   -> この画像を開いて、正しく映っているか確認してください。")
            
        except Exception as e:
            print(f"❌ キャプチャエラー: {e}")
            print("   -> ウィンドウが画面外にあるか、モニター設定の問題の可能性があります。")

if __name__ == "__main__":
    test_capture()