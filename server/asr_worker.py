#!/usr/bin/env python3
"""
Ming-3.0-ASR 常驻 worker（在独立 Python 环境中运行，见 bin/asr-python/）。

Ming-3.0-TTS 需要 transformers 4.46.x，而 Ming-3.0-ASR 需要 5.12.x，
两者无法共存于同一进程，因此 ASR 在本独立进程中运行（CPU 推理，
不占用显存）。与主程序通过 stdin/stdout 的 JSON 行协议通信：

  请求: {"audio_b64": "<base64 float32 little-endian PCM>", "sr": 16000}
  响应: {"text": "...", "ms": 1234.5}
  心跳: {"cmd": "ping"}          -> {"text": "pong"}
  结束: {"cmd": "quit"}          -> 进程退出

模型在首次启动时加载（日志走 stderr，不污染 stdout 协议通道）。
"""

import base64
import io
import json
import sys
import time

import numpy as np


def log(msg):
    print(f"[asr-worker] {msg}", file=sys.stderr, flush=True)


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    model_ref = sys.argv[1] if len(sys.argv) > 1 else 'models/ming-asr/asr-tiny'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    log(f"loading {model_ref} ({device}, bfloat16) ...")
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(model_ref, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_ref, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map=device, attn_implementation='sdpa',
    ).eval()
    log(f"model loaded in {time.time()-t0:.1f}s")
    return model, proc


def transcribe(model, proc, pcm_bytes: bytes, sr: int) -> str:
    import torch
    from server.ming_asr_runtime import (
        prepare_asr_request, context_chunk_frames, decode_transcript,
    )

    audio = np.frombuffer(pcm_bytes, dtype='<f4').astype(np.float32)
    duration = len(audio) / sr
    if duration > 60:
        audio = audio[-int(60 * sr):]
        duration = 60.0
    chunk_frames = context_chunk_frames(duration)

    inputs = prepare_asr_request(
        proc, (audio, sr), audio_chunk_frames=chunk_frames,
    ).to(model.device)
    eos = getattr(model.config, "text_config", model.config).eos_token_id
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=512, do_sample=False,
            eos_token_id=eos, pad_token_id=eos, no_repeat_ngram_size=10,
        )
    return decode_transcript(proc.tokenizer, out[0, inputs.input_ids.shape[1]:].tolist())


def main():
    # 让 worker 能找到 server/ming_asr_runtime.py（包根在 argv[2] 或 cwd）
    root = sys.argv[2] if len(sys.argv) > 2 else '.'
    if root not in sys.path:
        sys.path.insert(0, root)

    model, proc = load_model()
    print(json.dumps({"text": "ready"}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("cmd") == "quit":
                break
            if req.get("cmd") == "ping":
                print(json.dumps({"text": "pong"}), flush=True)
                continue
            pcm = base64.b64decode(req["audio_b64"])
            sr = int(req.get("sr", 16000))
            t0 = time.time()
            text = transcribe(model, proc, pcm, sr)
            print(json.dumps({"text": text, "ms": round((time.time()-t0)*1000)}), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == '__main__':
    main()
