# Windows 一键整合包

本项目的 Windows 整合包在开源版 LiveTalking 基础上做了以下二次开发，实现
**解压即用、无需安装 Python/CUDA、默认全部本地推理**。

## 整合内容

| 组件 | 模型/引擎 | 说明 |
|------|-----------|------|
| 数字人口型 | wav2lip256 | 官方预处理头像 wav2lip256_avatar1，25fps 实时 |
| 大语言模型 | **Qwen3.5-9B（Q4_K_M）** | 阿里通义千问最新开源模型，llama.cpp CUDA 引擎本地推理 |
| 语音合成 TTS | **Ming-3.0-tts-0.5B**（默认，蚂蚁集团，MIT） | 本地合成 44.1kHz，支持零样本身音克隆（参考 wav → 声纹缓存） |
| 语音识别 ASR | **Ming-3.0-asr-tiny**（默认，蚂蚁集团，MIT）+ SenseVoice 兜底 | Ming 独立进程运行；显存不足自动回退 SenseVoice |

## 二次开发点

1. `llm.py` — LLM 提供商扩展：
   - `localllm`：内置 llama.cpp llama-server（`bin/llama/`），启动器自动拉起，OpenAI 兼容接口
   - `deepseek` / `zhipu` / `custom`（任意 OpenAI 兼容端点）+ 原有 dashscope/orcarouter
   - 系统提示词可配置（`--llm_system_prompt` 或环境变量 `LLM_SYSTEM_PROMPT`）
2. `tts/mingtts.py` — 新增 Ming-3.0-TTS 本地插件（默认音色 / 内置音色 / **零样本声音克隆**：REF_FILE 填参考 wav，自动提取声纹缓存 .spkemb.pt）
3. `server/asr_server.py` — ASR 后端重构：默认 Ming-3.0-asr-tiny（`server/asr_worker.py` 在 `bin/asr-python` 独立解释器中常驻，stdin/stdout JSON 协议），失败自动回退 SenseVoice；浏览器端协议不变
4. `server/ming_asr_runtime.py` — vendor 自 Ming-3.0-asr（MIT）的请求构造/解码辅助
5. `launcher.py` + `启动数字人.bat` — 一键启动：自动拉起本地大模型 → 数字人服务 → 打开浏览器
6. `安装-创建桌面快捷方式.bat` — 环境自检 + 创建桌面快捷方式

## 依赖版本要点（重要）

- **Ming-3.0-TTS 只能跑在 transformers 4.46.x**（4.50+ 会破坏其 KV-cache 生成循环，输出乱音），requirements.txt 已钉 `transformers==4.46.3`
- **Ming-3.0-ASR 只能跑在 transformers 5.12.x**（`LinearAttentionLayer`）→ 因此 ASR 必须独立进程（bin/asr-python，CUDA torch + triton-windows + fla-core）
- triton 需要 CPython 头文件编译驱动绑定：内嵌 Python 需补 Include/ 与 python312.lib（见构建步骤）
- qwen3tts 插件保留在代码中，但需 transformers 4.57 环境，整合包默认不装

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
   - `inclusionAI/Ming-3.0-tts-0.5B` → `models/ming-tts/0.5B`
   - `inclusionAI/Ming-3.0-asr-tiny` → `models/ming-asr/asr-tiny`（fp32 约 16.6GB）
   - `iic/SenseVoiceSmall`、`iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` → `models/asr/`（兜底）
6. 7-Zip 打包：`7z a -t7z -mx=1 LiveTalking数字人一键整合包-v1.0.7z LiveTalking数字人整合包`

## 已知限制

- Ming-3.0-TTS 本地合成 RTF≈1.2-1.3（5060 Ti 实测），每句有 1-3 秒生成延迟；
  追求绝对流畅可在 config.yaml 切 edgetts（需联网）
- Ming-3.0-ASR 显存需求较高（权重 8.3GB bf16 + KDA 内核工作区），与 LLM/TTS/数字人
  同时驻留 16GB 显存会溢出 → 自动回退 SenseVoice；语音识别延迟约 5-10 秒/句
- 本地大模型需 ≥12GB 显存（Q4 量化 5.7GB + 数字人推理约 2GB）
- musetalk/ultralight 数字人模型未打包，可按官方文档自行添加
