@echo off
REM ■■■ 魔法の1行: 管理者で実行しても、このファイルの場所を作業フォルダにする ■■■
cd /d %~dp0

echo Starting 5 Bots...

REM ■ 5台起動 (pythonのパスが通っている前提)
start "Bot-1" python single_bot.py --device emulator-5554
start "Bot-2" python single_bot.py --device emulator-5556
start "Bot-3" python single_bot.py --device emulator-5558
start "Bot-4" python single_bot.py --device emulator-5560
start "Bot-5" python single_bot.py --device emulator-5562

echo All bots started!
pause