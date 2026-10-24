@echo off
chcp 65001 >nul
title LiveTalking 数字人
cd /d "%~dp0"

rem ── 首次运行自举：下载内嵌 Python（约 11MB），后续启动跳过 ──
if exist "runtime\python\Scripts\pip.exe" goto run

echo [首次运行] 正在准备运行环境（下载内嵌 Python 约 11MB）...
if not exist runtime mkdir runtime
if not exist runtime\python mkdir runtime\python
curl -sL -o runtime\python-embed.zip https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip
tar -xf runtime\python-embed.zip -C runtime\python
del runtime\python-embed.zip
(echo python312.zip & echo, & echo Lib/site-packages & echo, & echo import site) > runtime\python\python312._pth
curl -sL -o runtime\get-pip.py https://bootstrap.pypa.io/get-pip.py
runtime\python\python.exe runtime\get-pip.py -q --no-warn-script-location
del runtime\get-pip.py
echo [首次运行] 内嵌 Python 就绪。

:run
set PYTHONIOENCODING=utf-8
runtime\python\python.exe launcher.py
pause
