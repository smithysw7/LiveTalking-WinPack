@echo off
chcp 65001 >nul
title 停止 LiveTalking 服务
echo 正在停止 LiveTalking 相关进程...
taskkill /IM llama-server.exe /F >nul 2>&1
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { ($_.CommandLine -like '*app.py*') -or ($_.CommandLine -like '*asr_worker.py*') -or ($_.CommandLine -like '*launcher.py*') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >nul 2>&1
echo 完成。
pause
