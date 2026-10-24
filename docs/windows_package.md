# Windows 一键整合包

本项目的 Windows 整合包在开源版 LiveTalking 基础上做了以下二次开发，实现
**解压即用、无需安装 Python/CUDA、默认全部本地推理**。

## 整合内容

| 组件 | 模型/引擎 | 说明 |
|------|-----------|------|
| 数字人口型 | wav2lip256 | 官方预处理头像 wav2lip256_avatar1，25fps 实时 |
| 大语言模型 | **Qwen3.5-9B（Q4_K_M）** | 阿里通义千问最新开源模型，llama.cpp CUDA 引擎本地推理 |
| 语音合成 TTS | edge-tts（默认） / **Qwen3-TTS-1.7B**（本地离线） | Qwen3-TTS 为阿里 2026 开源 TTS，支持 9 音色/音色克隆 |
| 语音识别 ASR | **SenseVoiceSmall + fsmn-vad**（本地） | 浏览器语音对话，无需云端 ASR |

## 二次开发点

1. `llm.py` — LLM 提供商扩展：
   - `localllm`：内置 llama.cpp llama-server（`bin/llama/`），启动器自动拉起，OpenAI 兼容接口
   - `deepseek` / `zhipu` / `custom`（任意 OpenAI 兼容端点）+ 原有 dashscope/orcarouter
   - 系统提示词可配置（`--llm_system_prompt` 或环境变量 `LLM_SYSTEM_PROMPT`）
2. `tts/qwen3tts.py` — 新增本地 TTS 插件（固定音色 + 音色克隆两种模式），模型走本地目录
3. `server/asr_server.py` — 优先加载 `models/asr/` 内置 SenseVoice 模型，离线可用
4. `launcher.py` + `启动数字人.bat` — 一键启动：自动拉起本地大模型 → 数字人服务 → 打开浏览器
5. `安装-创建桌面快捷方式.bat` — 环境自检 + 创建桌面快捷方式

## 整合包目录结构

```
LiveTalking数字人整合包/
├── 启动数字人.bat            # 双击启动
├── 停止服务.bat
├── 安装-创建桌面快捷方式.bat
├── 使用说明.txt
├── app.py / config.yaml / launcher.py ...
├── runtime/python/           # 内嵌 Python 3.12 + torch 2.9.1 cu128 + 全部依赖
├── bin/llama/                # llama.cpp (CUDA 13.4, 含 sm_120 Blackwell 内核)
├── models/
│   ├── wav2lip.pth           # 口型模型
│   ├── llm/Qwen3.5-9B-Q4_K_M.gguf
│   ├── qwen3-tts/1.7B-CustomVoice/   # Qwen3-TTS（本地合成）
│   └── asr/{SenseVoiceSmall, fsmn-vad}
└── data/avatars/wav2lip256_avatar1/  # 预处理头像
```

## 构建方法（复现）

1. 克隆本仓库，切换到 `win-pack` 分支
2. 下载 [python-3.12.10-embed-amd64.zip](https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip) 解压到 `runtime/python`，
   修改 `python312._pth` 启用 `import site` 与 `Lib/site-packages`，get-pip 后安装：
   `pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128`
   再安装 `requirements.txt` + `funasr modelscope qwen-tts`
3. 添加 `runtime/python/Lib/site-packages/sitecustomize.py`（把包根加入 sys.path，见构建脚本）
4. [llama.cpp b11849](https://github.com/ggml-org/llama.cpp/releases) 下载
   `llama-b11849-bin-win-cuda-13.4-x64.zip` 与 `cudart-llama-bin-win-cuda-13.4-x64.zip` 解压到 `bin/llama`
5. 模型下载：
   - wav2lip256.pth / wav2lip256_avatar1.tar.gz：官方 [Google Drive](https://drive.google.com/drive/folders/1FOC_MD6wdogyyX_7V1d4NDIO7P9NlSAJ)
   - `Qwen/Qwen3.5-9B-GGUF`（unsloth）→ `models/llm/`
   - `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` → `models/qwen3-tts/1.7B-CustomVoice`
   - `iic/SenseVoiceSmall`、`iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` → `models/asr/`
6. 7-Zip 打包：`7z a -t7z -mx=1 LiveTalking数字人一键整合包-v1.0.7z LiveTalking数字人整合包`

## 已知限制

- Qwen3-TTS 本地合成为逐 token 循环实现，在 RTX 5060 Ti 上 RTF≈2（音频时长 2 倍），
  追求流畅体验请使用默认 edge-tts（需联网）
- 本地大模型需 ≥12GB 显存（Q4 量化 5.7GB + 数字人推理约 2GB）
- musetalk/ultralight 数字人模型未打包，可按官方文档自行添加
