###############################################################################
#  ASR WebSocket Server — Local SenseVoice/FunASR Integration
#
#  Resolves: https://github.com/lipku/LiveTalking/issues/604
#
#  This module provides a WebSocket endpoint (/api/asr) that speaks the same
#  protocol as the external FunASR server (wss://www.funasr.com:10096/).
#  The browser client (web/asr/main.js) can connect here instead, keeping
#  all ASR processing local and cutting ~600ms of network + Whisper latency.
#
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  Licensed under the Apache License, Version 2.0
###############################################################################

import json
import time
import io
import os
import sys
import subprocess
import asyncio
import threading
import numpy as np
from aiohttp import web

from utils.logger import logger


# ─── Lazy Model Loader ────────────────────────────────────────────────────

_asr_load_lock = threading.Lock()
_asr_inference_lock = threading.Lock()

DEFAULT_MING_ASR_DIR = 'models/ming-asr/asr-tiny'
MING_ASR_PYTHON = 'bin/asr-python/python.exe'


class _MingASRWorker:
    """
    Ming-3.0-ASR 常驻子进程客户端。

    Ming-3.0-TTS 需要 transformers 4.46.x，Ming-3.0-ASR 需要 5.12.x，
    二者无法共存于同一进程，因此 ASR 由独立解释器（bin/asr-python）
    以子进程方式运行，见 server/asr_worker.py。
    """
    _proc = None
    _disabled = False   # 显存不足等原因失败后，本次会话不再重试

    @classmethod
    def _start(cls):
        # 懒加载：首次语音对话时自动下载 ASR 组件（模型 + 独立环境，约 22GB）
        if (not os.path.isfile(MING_ASR_PYTHON)) or (
                not os.path.isdir(DEFAULT_MING_ASR_DIR)):
            logger.info("[ASR] 首次使用语音识别：开始下载 Ming-3.0-ASR 与独立环境"
                        "（约 22GB，视网速需数十分钟，期间本次请求会等待）...")
            r = subprocess.run(
                [sys.executable, 'setup_components.py', 'download', 'asr'],
                cwd=os.getcwd())
            if r.returncode != 0:
                raise RuntimeError('ASR 组件自动下载失败，请检查网络或手动运行 '
                                   'python setup_components.py download asr')
        if not os.path.isfile(MING_ASR_PYTHON):
            raise FileNotFoundError(f"{MING_ASR_PYTHON} 不存在")
        cls._proc = subprocess.Popen(
            [MING_ASR_PYTHON, 'server/asr_worker.py', DEFAULT_MING_ASR_DIR, '.'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding='utf-8', bufsize=1,
        )
        # 等待模型加载完成（stdout 第一行 "ready"），日志走 stderr 直通控制台
        deadline = time.time() + 600
        while time.time() < deadline:
            line = cls._proc.stdout.readline()
            if not line:
                raise RuntimeError("ASR worker exited during startup")
            data = json.loads(line)
            if data.get('text') == 'ready':
                logger.info("[ASR] Ming-3.0-ASR worker ready")
                return
        raise TimeoutError("ASR worker startup timeout")

    @classmethod
    def transcribe(cls, audio_float32: np.ndarray, sample_rate: int) -> str:
        import base64
        if cls._disabled:
            raise RuntimeError("Ming-ASR worker disabled for this session")
        if cls._proc is None or cls._proc.poll() is not None:
            with _asr_load_lock:
                if cls._proc is None or cls._proc.poll() is not None:
                    cls._start()
        pcm = audio_float32.astype('<f4').tobytes()
        req = json.dumps({
            "audio_b64": base64.b64encode(pcm).decode('ascii'),
            "sr": sample_rate,
        })
        cls._proc.stdin.write(req + '\n')
        cls._proc.stdin.flush()
        line = cls._proc.stdout.readline()
        if not line:
            raise RuntimeError("ASR worker died")
        data = json.loads(line)
        if 'error' in data:
            raise RuntimeError(f"ASR worker error: {data['error']}")
        logger.info(f"[ASR] worker inference {data.get('ms')}ms")
        return data.get('text', '')


def _run_ming_inference(audio_float32: np.ndarray, sample_rate: int):
    """Transcribe a float32 mono waveform with Ming-3.0-ASR (worker process)."""
    t0 = time.perf_counter()
    with _asr_inference_lock:
        text = _MingASRWorker.transcribe(audio_float32, sample_rate)
    duration = len(audio_float32) / sample_rate
    return text, (time.perf_counter() - t0) * 1000, duration


_sensevoice_model = None


def _load_sensevoice():
    """
    Fallback backend: Load the SenseVoice model on first call (lazy singleton).
    Concurrent first requests must share the same model initialization.
    """
    global _sensevoice_model
    if _sensevoice_model is not None:
        return _sensevoice_model

    with _asr_load_lock:
        if _sensevoice_model is not None:
            return _sensevoice_model

        import torch
        from funasr import AutoModel

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(
            f"[ASR] Loading SenseVoiceSmall on device='{device}' "
            f"(first run will download ~500MB from ModelScope)..."
        )

        t0 = time.perf_counter()
        # Prefer bundled local copies (models/asr/) so the package works offline;
        # fall back to ModelScope ids which download on first use.
        sensevoice_ref = "models/asr/SenseVoiceSmall" if os.path.isdir("models/asr/SenseVoiceSmall") else "iic/SenseVoiceSmall"
        vad_ref = "models/asr/fsmn-vad" if os.path.isdir("models/asr/fsmn-vad") else "fsmn-vad"
        _sensevoice_model = AutoModel(
            model=sensevoice_ref,
            vad_model=vad_ref,
            vad_kwargs={"max_single_segment_time": 30000},
            device=device,
            trust_remote_code=True,
        )
        elapsed = time.perf_counter() - t0
        logger.info(
            f"[ASR] ✅ SenseVoiceSmall ready — loaded in {elapsed:.1f}s on {device}"
        )
    return _sensevoice_model


def _run_inference(audio_float32: np.ndarray, sample_rate: int, use_itn: bool):
    """
    Run ASR on a float32 audio array.

    Backend selection (env ASR_BACKEND, default "ming"):
      - "ming":       Ming-3.0-asr-tiny (Ant Group, MIT) in a separate worker
                      process — better accuracy, but needs ~5GB VRAM and its
                      kernels are tuned for datacenter GPUs; on consumer GPUs
                      it may run slowly or OOM when other models are loaded.
      - "sensevoice": SenseVoiceSmall via funasr — fast and lightweight.
    Either way the other backend serves as an automatic fallback.

    This is a **blocking** call — always invoke from ``run_in_executor``.

    Returns
    -------
    tuple[str, float, float]
        (transcribed_text, inference_ms, audio_duration_s)
    """
    backend = os.environ.get('ASR_BACKEND', 'ming').lower()
    primary, secondary = (
        (_run_ming_inference, _run_sensevoice_inference)
        if backend != 'sensevoice' else
        (_run_sensevoice_inference, _run_ming_inference)
    )
    t0 = time.perf_counter()
    try:
        if primary is _run_sensevoice_inference:
            text, inference_ms, audio_duration_s = primary(audio_float32, sample_rate, use_itn)
        else:
            text, inference_ms, audio_duration_s = primary(audio_float32, sample_rate)
        logger.info(f"[ASR] backend={'Ming-3.0-asr-tiny' if primary is _run_ming_inference else 'SenseVoice'}")
        return text, inference_ms, audio_duration_s
    except Exception as e:
        logger.warning(f"[ASR] primary backend failed ({e}), falling back")
        if primary is _run_ming_inference:
            _MingASRWorker._disabled = True

    if secondary is _run_sensevoice_inference:
        text, inference_ms, audio_duration_s = secondary(audio_float32, sample_rate, use_itn)
    else:
        text, inference_ms, audio_duration_s = secondary(audio_float32, sample_rate)
    return text, inference_ms, audio_duration_s


def _run_sensevoice_inference(audio_float32: np.ndarray, sample_rate: int, use_itn: bool = False):
    import soundfile as sf
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    model = _load_sensevoice()

    # Write to in-memory WAV so funasr can read the sample rate from the header
    wav_buf = io.BytesIO()
    sf.write(wav_buf, audio_float32, sample_rate, format="WAV")
    wav_buf.seek(0)

    t0 = time.perf_counter()
    with _asr_inference_lock:
        res = model.generate(
            input=wav_buf,
            cache={},
            language="auto",
            use_itn=use_itn,
            batch_size_s=60,
        )
    inference_ms = (time.perf_counter() - t0) * 1000

    text = ""
    if res and len(res) > 0 and res[0].get("text"):
        text = rich_transcription_postprocess(res[0]["text"])

    audio_duration_s = len(audio_float32) / sample_rate

    logger.info(
        f"[ASR] ✅ SenseVoice inference complete\n"
        f"       ├─ Latency     : {inference_ms:>8.0f} ms\n"
        f"       ├─ Audio length: {audio_duration_s:>8.1f} s\n"
        f"       ├─ RTF         : {inference_ms / 1000 / max(audio_duration_s, 0.001):>8.3f}\n"
        f"       └─ Text        : \"{text[:100]}{'…' if len(text) > 100 else ''}\""
    )

    return text, inference_ms, audio_duration_s


# ─── WebSocket Handler ─────────────────────────────────────────────────────

SAMPLE_RATE = 16000  # The browser client records at 16 kHz mono PCM16


async def asr_websocket_handler(request):
    """
    WebSocket handler implementing the FunASR client protocol.

    Protocol flow
    -------------
    1. Client opens connection
    2. Client sends JSON config::

           {"chunk_size":[5,10,5], "wav_name":"h5",
            "is_speaking":true, "mode":"2pass", "itn":false, ...}

    3. Client streams binary PCM16 audio chunks (960 bytes = 60 ms @ 16 kHz)
    4. Client sends stop signal::

           {"is_speaking":false, ...}

    5. Server responds with transcription::

           {"text":"hello world", "mode":"2pass-offline",
            "is_final":true, "timestamp":null}
    """
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    client_ip = request.remote
    logger.info(f"[ASR] 🔌 WebSocket connected from {client_ip}")

    audio_buffer = bytearray()
    config: dict = {}
    session_start = time.perf_counter()
    chunks_received = 0

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    logger.warning("[ASR] Received invalid JSON, ignoring")
                    continue

                if data.get("is_speaking") is True:
                    # ── Session start ──────────────────────────────────
                    config = data
                    audio_buffer = bytearray()
                    chunks_received = 0
                    session_start = time.perf_counter()
                    logger.info(
                        f"[ASR] 🎙️  Recording started | "
                        f"mode={config.get('mode', 'offline')} | "
                        f"itn={config.get('itn', False)} | "
                        f"hotwords={bool(config.get('hotwords'))}"
                    )

                elif data.get("is_speaking") is False:
                    # ── End of speech → run inference ──────────────────
                    buf_bytes = len(audio_buffer)
                    audio_seconds = buf_bytes / (SAMPLE_RATE * 2)  # 2 bytes per int16
                    session_elapsed = time.perf_counter() - session_start

                    logger.info(
                        f"[ASR] 🛑 Recording stopped | "
                        f"{chunks_received} chunks | "
                        f"{buf_bytes:,} bytes | "
                        f"{audio_seconds:.1f}s audio | "
                        f"session wall time {session_elapsed:.1f}s"
                    )

                    if buf_bytes < 640:  # < 20 ms of audio — skip
                        logger.warning("[ASR] Audio too short (< 20ms), returning empty")
                        await ws.send_str(json.dumps({
                            "text": "",
                            "mode": config.get("mode", "offline"),
                            "is_final": True,
                            "timestamp": None,
                        }))
                        continue

                    # Ensure even number of bytes for int16 conversion
                    if buf_bytes % 2 != 0:
                        logger.warning(f"[ASR] Odd number of bytes received ({buf_bytes}), dropping incomplete sample")
                        audio_buffer = audio_buffer[:-1]
                        buf_bytes -= 1

                    # Convert PCM16 → float32 in [-1, 1]
                    audio_int16 = np.frombuffer(bytes(audio_buffer), dtype=np.int16)
                    audio_float32 = audio_int16.astype(np.float32) / 32768.0
                    use_itn = config.get("itn", False)

                    # Offload blocking inference to a thread
                    loop = asyncio.get_event_loop()
                    try:
                        text, inference_ms, audio_dur = await loop.run_in_executor(
                            None,
                            _run_inference,
                            audio_float32,
                            SAMPLE_RATE,
                            use_itn,
                        )
                    except Exception as e:
                        logger.exception(f"[ASR] ❌ Inference failed: {e}")
                        text = ""

                    # Map the client mode to the response mode the frontend expects
                    mode = config.get("mode", "offline")
                    if mode == "2pass":
                        response_mode = "2pass-offline"
                    else:
                        response_mode = mode  # "online" or "offline"

                    await ws.send_str(json.dumps({
                        "text": text,
                        "mode": response_mode,
                        "is_final": True,
                        "timestamp": None,
                    }))
                    logger.info(f"[ASR] 📤 Result sent to client (mode={response_mode})")

            elif msg.type == web.WSMsgType.BINARY:
                audio_buffer.extend(msg.data)
                chunks_received += 1

            elif msg.type in (web.WSMsgType.ERROR, web.WSMsgType.CLOSE):
                break

    except asyncio.CancelledError:
        logger.info("[ASR] WebSocket handler cancelled")
    except Exception as e:
        logger.exception(f"[ASR] ❌ WebSocket handler error: {e}")

    logger.info(f"[ASR] 🔌 WebSocket disconnected ({client_ip})")
    return ws


# ─── Availability Check ───────────────────────────────────────────────────

def is_funasr_available() -> bool:
    """Return True if any local ASR backend (Ming or funasr) is importable."""
    try:
        import transformers  # noqa: F401
        return True  # Ming-3.0-ASR shares the transformers stack
    except ImportError:
        pass
    try:
        import funasr  # noqa: F401
        return True
    except ImportError:
        return False
