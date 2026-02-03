@echo off
set "SOURCE_DIR=%~dp0"
set "TARGET_DIR=E:\AI_Project_Hub\Mahjong_Maker"


echo [0/3] 最新のコードを適用中...

:: 既存のプロセスをクリーンアップ
taskkill /F /IM ngrok.exe >nul 2>&1
taskkill /F /IM python.exe >nul 2>&1

copy /Y "%SOURCE_DIR%server.py" "%TARGET_DIR%\"
copy /Y "%SOURCE_DIR%mahjong_logic.py" "%TARGET_DIR%\"

E:
cd /d "%TARGET_DIR%"

:: Pythonライブラリのインストール（すでにあれば一瞬で終わります）
echo [1/3] ライブラリを確認中...
pip install numpy opencv-python flask ultralytics numba

:: ngrokの認証（secrets.batがあれば読み込む）
echo [2/3] ngrokを認証中...
if exist secrets.bat (
    call secrets.bat
) else (
    echo [Skip] secrets.bat not found. Skipping auth...
)

:: サーバーとngrokを別ウィンドウで起動
echo [3/3] AIサーバーとトンネルを起動します...
start "Jantama AI Server" cmd /k "python server.py"
timeout /t 3 >nul
start "Ngrok Tunnel" cmd /k "ngrok http 5000"

echo すべてのプロセスを開始しました。
pause