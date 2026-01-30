import pygetwindow as gw

print("📋 現在開いているウィンドウ一覧:")
all_windows = gw.getAllTitles()
ld_windows = [title for title in all_windows if "LD" in title or "Player" in title]

if not ld_windows:
    print("❌ 'LD' または 'Player' を含むウィンドウが見つかりません。")
else:
    for w in ld_windows:
        print(f"✅ 発見: '{w}'")

print("\n--------------------------------")
print("もし上記に 'LDPlayer' などが表示されていない場合、")
print("エミュレーターを起動しているか、最小化していないか確認してください。")