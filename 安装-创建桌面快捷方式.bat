@echo off
chcp 65001 >nul
title LiveTalking 安装程序
echo.
echo  ┌─────────────────────────────────────────────┐
echo  │   LiveTalking AI 数字人整合包 - 安装程序      │
echo  └─────────────────────────────────────────────┘
echo.
echo  本整合包为绿色免安装版，无需此步骤也可直接双击
echo  「启动数字人.bat」使用。
echo.
echo  安装程序将执行以下操作：
echo   1. 检测运行环境（显卡/系统）
echo   2. 在桌面创建「LiveTalking数字人」快捷方式
echo.
set /p CONFIRM=是否继续？(Y/N):
if /i not "%CONFIRM%"=="Y" exit /b

echo.
echo [1/2] 检测运行环境...
wmic path win32_VideoController get name | findstr /i "NVIDIA" >nul
if %errorlevel%==0 (
    echo   √ 检测到 NVIDIA 显卡
) else (
    echo   × 未检测到 NVIDIA 显卡！数字人推理需要 NVIDIA 显卡才能实时运行
)

if not exist "%~dp0runtime\python\python.exe" (
    echo   × 缺少 runtime\python 运行环境，整合包不完整！
    pause & exit /b 1
) else (
    echo   √ 内置 Python 运行环境完整
)

echo   √ 本地大模型检查...
dir /b "%~dp0models\llm\*.gguf" >nul 2>&1 && (echo   √ Qwen3.5-9B 大模型已就绪) || (echo   × models\llm 下缺少 .gguf 大模型文件)

echo.
echo [2/2] 创建桌面快捷方式...
powershell -NoProfile -Command ^
  "$ws = New-Object -ComObject WScript.Shell;" ^
  "$s = $ws.CreateShortcut([Environment]::GetFolderPath('Desktop') + '\LiveTalking数字人.lnk');" ^
  "$s.TargetPath = '%~dp0启动数字人.bat';" ^
  "$s.WorkingDirectory = '%~dp0';" ^
  "$s.Description = 'LiveTalking AI 数字人（本地大模型版）';" ^
  "$s.Save()"
echo   √ 桌面快捷方式已创建
echo.
echo  ═══════════════════════════════════════════
echo   安装完成！双击桌面「LiveTalking数字人」即可使用
echo  ═══════════════════════════════════════════
pause
