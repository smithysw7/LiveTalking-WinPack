"""
Qwen3-TTS local TTS plugin (https://github.com/QwenLM/Qwen3-TTS).

Runs the open-source Qwen3-TTS model in-process, so the digital human works
fully offline (no cloud TTS needed). Supports:
  - CustomVoice models  (--tts_model pointing to Qwen3-TTS-12Hz-1.7B-CustomVoice):
    9 premium speakers selected via --REF_FILE (default Vivian), optional
    style instruction via --REF_TEXT.
  - Base models (Qwen3-TTS-12Hz-*-Base): zero-shot voice cloning — set
    --REF_FILE to a reference wav (16kHz mono recommended) and --REF_TEXT to
    its transcript.

Usage:
    python app.py --tts qwen3tts --tts_model models/qwen3-tts/1.7B-CustomVoice \
                  --REF_FILE Vivian
"""

import os
import time
import threading

import numpy as np
import resampy

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register

DEFAULT_CUSTOMVOICE_DIR = 'models/qwen3-tts/1.7B-CustomVoice'


@register("tts", "qwen3tts")
class Qwen3TTS(BaseTTS):
    # Model instances are shared across avatar sessions (class-level cache).
    _shared_models: dict = {}
    _gen_lock = threading.Lock()

    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        self.model_ref = getattr(opt, 'tts_model', '') or DEFAULT_CUSTOMVOICE_DIR
        if not os.path.isdir(self.model_ref):
            logger.warning(
                f"[qwen3tts] 本地模型目录不存在: {self.model_ref}，"
                "将尝试从 HuggingFace 自动下载 Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
            )
            self.model_ref = 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'
        self.speaker = opt.REF_FILE or 'Vivian'
        self.is_clone_model = 'base' in os.path.basename(
            self.model_ref.rstrip('/\\')).lower()
        self._ref_audio = None
        self._ref_text = None
        self._clone_prompt = None

    # ── lazy model loading ─────────────────────────────────────────────
    def _get_model(self):
        model = Qwen3TTS._shared_models.get(self.model_ref)
        if model is not None:
            return model
        with Qwen3TTS._gen_lock:
            model = Qwen3TTS._shared_models.get(self.model_ref)
            if model is not None:
                return model
            import torch
            from qwen_tts import Qwen3TTSModel
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            dtype = torch.bfloat16 if device.startswith('cuda') else torch.float32
            logger.info(f"[qwen3tts] loading {self.model_ref} on {device} ({dtype})...")
            t = time.time()
            model = Qwen3TTSModel.from_pretrained(
                self.model_ref,
                device_map=device,
                dtype=dtype,
                attn_implementation='sdpa',
            )
            Qwen3TTS._shared_models[self.model_ref] = model
            logger.info(f"[qwen3tts] model loaded in {time.time()-t:.1f}s")
            return model

    # ── generation ─────────────────────────────────────────────────────
    def _generate(self, text: str):
        model = self._get_model()
        with Qwen3TTS._gen_lock:
            if self.is_clone_model:
                ref = self._resolve_ref_audio()
                if ref is None:
                    return None, None
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language='Auto',
                    ref_audio=self._ref_audio,
                    ref_text=self._ref_text,
                )
            else:
                instruct = None
                speaker = self.speaker
                # REF_TEXT doubles as style instruction on CustomVoice models
                # (only when it is not a reference transcript).
                if self.opt.REF_TEXT and not self.opt.REF_FILE.endswith('.wav'):
                    instruct = self.opt.REF_TEXT
                wavs, sr = model.generate_custom_voice(
                    text=text,
                    speaker=speaker,
                    language='Auto',
                    **({'instruct': instruct} if instruct else {}),
                )
            wav = wavs[0]
        return wav, sr

    def _resolve_ref_audio(self):
        import soundfile as sf
        if not self.opt.REF_FILE or not os.path.exists(self.opt.REF_FILE):
            logger.error(
                f"[qwen3tts] 克隆模式需要 --REF_FILE 指向有效的参考音频，当前: {self.opt.REF_FILE}"
            )
            return None
        if self._ref_audio is None:
            data, sr = sf.read(self.opt.REF_FILE, dtype='float32')
            if data.ndim > 1:
                data = data[:, 0]
            self._ref_audio = (data, sr)
            self._ref_text = self.opt.REF_TEXT or ''
        return self._ref_audio

    # ── BaseTTS interface ──────────────────────────────────────────────
    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        t = time.time()
        try:
            wav, sr = self._generate(text)
        except Exception as e:
            logger.exception(f'[qwen3tts] generate failed: {e}')
            return
        if wav is None or len(wav) == 0:
            logger.error('[qwen3tts] empty audio output')
            return
        logger.info(f'-------qwen3tts time:{time.time()-t:.4f}s sr={sr}')

        stream = np.asarray(wav, dtype=np.float32)
        if stream.ndim > 1:
            stream = stream[:, 0]
        if sr != self.sample_rate and stream.shape[0] > 0:
            stream = resampy.resample(x=stream, sr_orig=sr, sr_new=self.sample_rate)

        streamlen = stream.shape[0]
        idx = 0
        while streamlen >= self.chunk and self.state == State.RUNNING:
            eventpoint = {}
            streamlen -= self.chunk
            if idx == 0:
                eventpoint = {'status': 'start', 'text': text}
            elif streamlen < self.chunk:
                eventpoint = {'status': 'end', 'text': text}
            eventpoint.update(**textevent)
            self.parent.put_audio_frame(stream[idx:idx + self.chunk], eventpoint)
            idx += self.chunk
