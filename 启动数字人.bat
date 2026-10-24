@echo off
chcp 65001 >nul
title LiveTalking 数字人整合包
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
runtime\python\python.exe launcher.py
pause
