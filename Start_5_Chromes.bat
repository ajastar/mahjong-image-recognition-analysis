@echo off
set "CHROME_PATH=C:\Program Files\Google\Chrome\Application\chrome.exe"
set "BASE_PROFILE=E:\AI_Project_Hub\Mahjong_Maker\browser_profiles"

mkdir "%BASE_PROFILE%" 2>nul

echo 🚀 Launching 5 Chrome Instances for AI Control...

:: Browser 1 (Port 9222)
start "" "%CHROME_PATH%" --remote-debugging-port=9222 --user-data-dir="%BASE_PROFILE%\Profile_1" "https://game.mahjongsoul.com/"

:: Browser 2 (Port 9223)
start "" "%CHROME_PATH%" --remote-debugging-port=9223 --user-data-dir="%BASE_PROFILE%\Profile_2" "https://game.mahjongsoul.com/"

:: Browser 3 (Port 9224)
start "" "%CHROME_PATH%" --remote-debugging-port=9224 --user-data-dir="%BASE_PROFILE%\Profile_3" "https://game.mahjongsoul.com/"

:: Browser 4 (Port 9225)
start "" "%CHROME_PATH%" --remote-debugging-port=9225 --user-data-dir="%BASE_PROFILE%\Profile_4" "https://game.mahjongsoul.com/"

:: Browser 5 (Port 9226)
start "" "%CHROME_PATH%" --remote-debugging-port=9226 --user-data-dir="%BASE_PROFILE%\Profile_5" "https://game.mahjongsoul.com/"

echo ✅ All browsers started! Please login manually.
pause