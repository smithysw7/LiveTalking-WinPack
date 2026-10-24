"""
Ming-3.0-TTS 本地语音合成插件（蚂蚁集团 inclusionAI，MIT 许可证）
https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B

功能：
  - 默认音色：直接文本转语音（44.1kHz）
  - 语音克隆：--REF_FILE 指向参考音频 wav，首次使用时自动提取说话人
    嵌入并缓存（<wav>.spkemb.pt），数字人用该声音说话
  - 内置音色：--REF_FILE 填官方内置音色名（非 wav 文件时按 IP 音色处理）

依赖：ming3tts-lib/（推理代码，MIT）+ transformers==4.46.3
注意：必须配合 transformers 4.46.x（4.50+ 会破坏其 KV-cache 生成循环）。
"""

import os
import sys
import time
import threading

import numpy as np
import resampy

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register

DEFAULT_MING_TTS_DIR = 'models/ming-tts/0.5B'
MING_LIB_DIR = 'ming3tts-lib'


@register("tts", "mingtts")
class MingTTSPlugin(BaseTTS):
    _shared_models: dict = {}
    _gen_lock = threading.Lock()

    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        self.model_ref = getattr(opt, 'tts_model', '') or DEFAULT_MING_TTS_DIR
        lib_dir = os.path.join(os.getcwd(), MING_LIB_DIR)
        if os.path.isdir(lib_dir) and lib_dir not in sys.path:
            sys.path.insert(0, lib_dir)
        self.ref_file = opt.REF_FILE or ''
        self.is_clone = bool(self.ref_file) and os.path.isfile(self.ref_file)
        self.is_ip_voice = bool(self.ref_file) and not self.is_clone
        self._spk_emb_path = None

    # ── lazy model loading ─────────────────────────────────────────────
    def _get_model(self):
        model = MingTTSPlugin._shared_models.get(self.model_ref)
        if model is not None:
            return model
        with MingTTSPlugin._gen_lock:
            model = MingTTSPlugin._shared_models.get(self.model_ref)
            if model is not None:
                return model
            from ming_tts import MingTTS
            import torch
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            logger.info(f"[mingtts] loading {self.model_ref} on {device} ...")
            t = time.time()
            model = MingTTS(
                self.model_ref,
                device=device,
                text_normalization=False,  # pynini 无 Windows wheel；数字由 cn2an 预处理
            )
            MingTTSPlugin._shared_models[self.model_ref] = model
            logger.info(f"[mingtts] model loaded in {time.time()-t:.1f}s")
            return model

    # ── 语音克隆：参考音频 → 说话人嵌入缓存 ───────────────────────────
    def _get_spk_emb(self):
        if self._spk_emb_path:
            return self._spk_emb_path
        cache_path = self.ref_file + '.spkemb.pt'
        import torch
        if os.path.exists(cache_path):
            self._spk_emb_path = cache_path
            logger.info(f"[mingtts] using cached speaker embedding: {cache_path}")
            return cache_path
        model = self._get_model()
        import torchaudio
        data, sr = __import__('soundfile').read(self.ref_file, dtype='float32', always_2d=True)
        wf = torch.from_numpy(data.T)
        if wf.shape[0] > 1:
            wf = wf.mean(dim=0, keepdim=True)
        target_sr = model.sample_rate
        if sr != target_sr:
            wf = torchaudio.transforms.Resample(sr, target_sr)(wf)
        with torch.inference_mode():
            emb = model.spkemb_extractor(
                torchaudio.transforms.Resample(target_sr, 16000)(wf))
        torch.save(emb, cache_path)
        self._spk_emb_path = cache_path
        logger.info(f"[mingtts] speaker embedding extracted -> {cache_path}")
        return cache_path

    # ── generation ─────────────────────────────────────────────────────
    @staticmethod
    def _normalize_digits(text: str) -> str:
        """阿拉伯数字→中文（官方文本正则化依赖 pynini，无 Windows wheel）。

        年份类 4 位数字按"二零二六"直读，其余按数值读，百分比转为"百分之X"。
        """
        try:
            import cn2an
        except ImportError:
            return text
        import re

        def rep(s: str) -> str:
            try:
                if re.fullmatch(r'\d{4}', s):
                    return cn2an.an2cn(s, 'direct')
                return cn2an.an2cn(s)
            except Exception:
                return s

        text = re.sub(r'(\d+(?:\.\d+)?)%', lambda m: '百分之' + rep(m.group(1)), text)
        text = re.sub(r'\d+(?:\.\d+)?', lambda m: rep(m.group(0)), text)
        return text

    def _generate(self, text: str):
        model = self._get_model()
        text = self._normalize_digits(text)
        with MingTTSPlugin._gen_lock:
            if self.is_clone:
                wav = model.synthesize(
                    task='TTS', text=text,
                    spk_emb_path=self._get_spk_emb(), use_spk_emb=True)
            elif self.is_ip_voice:
                wav = model.synthesize(
                    task='IP', text=text, controls={'ip': self.ref_file})
            else:
                wav = model.synthesize(task='TTS', text=text)
        return wav, model.sample_rate

    # ── BaseTTS interface ──────────────────────────────────────────────
    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        t = time.time()
        try:
            wav, sr = self._generate(text)
        except Exception as e:
            logger.exception(f'[mingtts] generate failed: {e}')
            return
        if wav is None or len(wav) == 0:
            logger.error('[mingtts] empty audio output')
            return
        logger.info(f'-------mingtts time:{time.time()-t:.4f}s sr={sr}')

        stream = np.asarray(wav, dtype=np.float32).squeeze()
        if stream.ndim > 1:
            stream = stream[0]
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
