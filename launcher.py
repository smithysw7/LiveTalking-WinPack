#!/usr/bin/env python3
"""
LiveTalking 一键启动器
  1. 读取 config.yaml
  2. 若 llm_provider=localllm，自动拉起内置 llama.cpp 大模型服务
  3. 启动 LiveTalking 数字人服务 (app.py)
  4. 自动打开浏览器进入控制台

由「启动数字人.bat」调用，一般不需要手动运行。
"""

import os
import sys
import time
import subprocess
import urllib.request
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)


def load_cfg() -> dict:
    try:
        import yaml
        p = os.path.join(ROOT, 'config.yaml')
        if os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
    except Exception as e:
        print(f'[launcher] 读取 config.yaml 失败: {e}', flush=True)
    return {}


def http_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def find_gguf() -> str:
    d = os.path.join(ROOT, 'models', 'llm')
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith('.gguf'):
                return os.path.join(d, f)
    return ''


def start_local_llm(cfg: dict):
    """启动 llama.cpp 本地大模型服务，返回进程或 None。"""
    exe = os.path.join(ROOT, 'bin', 'llama', 'llama-server.exe')
    gguf = os.environ.get('LLM_GGUF') or find_gguf()
    port = int(cfg.get('llm_port', 8765) or 8765)
    ctx = int(cfg.get('llm_ctx_size', 8192) or 8192)
    base = os.environ.get('LLM_BASE_URL') or f'http://127.0.0.1:{port}/v1'

    if not os.path.exists(exe) or not gguf:
        print('[launcher] 未找到本地大模型（bin/llama/llama-server.exe 或 models/llm/*.gguf）', flush=True)
        print('           请在 .env 中改用云端 provider（dashscope/deepseek/zhipu/custom）', flush=True)
        return None

    if http_ok(base.replace('/v1', '/health')):
        print(f'[launcher] 本地大模型服务已在运行: {base}', flush=True)
        return None

    print(f'[launcher] 正在启动本地大模型: {os.path.basename(gguf)} (port {port})', flush=True)
    log = open(os.path.join(ROOT, 'data', 'llama-server.log'), 'a', encoding='utf-8')
    cmd = [
        exe,
        '-m', gguf,
        '--host', '127.0.0.1',
        '--port', str(port),
        '-ngl', '99',
        '--ctx-size', str(ctx),
        '--reasoning-budget', '0',
        '--no-webui',
    ]
    proc = subprocess.Popen(cmd, cwd=os.path.dirname(exe), stdout=log, stderr=subprocess.STDOUT)

    t0 = time.time()
    while time.time() - t0 < 180:
        if http_ok(base.replace('/v1', '/health')):
            print(f'[launcher] 本地大模型就绪: {base} ({time.time()-t0:.0f}s)', flush=True)
            return proc
        if proc.poll() is not None:
            print('[launcher] llama-server 启动失败，详见 data/llama-server.log', flush=True)
            return None
        time.sleep(1)
    print('[launcher] 等待本地大模型超时（服务可能仍在加载，稍后仍可对话）', flush=True)
    return proc


def main():
    cfg = load_cfg()
    listenport = int(cfg.get('listenport', 8010) or 8010)
    provider = cfg.get('llm_provider', 'localllm')

    if http_ok(f'http://127.0.0.1:{listenport}/'):
        print(f'[launcher] 数字人服务已在运行，直接打开页面', flush=True)
        webbrowser.open(f'http://127.0.0.1:{listenport}/dashboard.html')
        return

    llama_proc = None
    if provider == 'localllm':
        llama_proc = start_local_llm(cfg)

    print(f'[launcher] 启动数字人服务 (http://127.0.0.1:{listenport}) ...', flush=True)
    app = subprocess.Popen([sys.executable, 'app.py'], cwd=ROOT)

    t0 = time.time()
    while time.time() - t0 < 300:
        if http_ok(f'http://127.0.0.1:{listenport}/'):
            break
        if app.poll() is not None:
            print('[launcher] 数字人服务启动失败，请查看上方日志', flush=True)
            if llama_proc:
                llama_proc.terminate()
            sys.exit(1)
        time.sleep(1)

    url = f'http://127.0.0.1:{listenport}/dashboard.html'
    print(f'[launcher] ✅ 启动完成，正在打开浏览器: {url}', flush=True)
    webbrowser.open(url)

    try:
        app.wait()
    except KeyboardInterrupt:
        pass
    finally:
        print('[launcher] 正在停止服务...', flush=True)
        for p in (app, llama_proc):
            if p and p.poll() is None:
                p.terminate()
        for p in (app, llama_proc):
            if p and p.poll() is None:
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
        print('[launcher] 已退出', flush=True)


if __name__ == '__main__':
    main()
