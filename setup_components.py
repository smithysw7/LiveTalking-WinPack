#!/usr/bin/env python3
"""
LiveTalking 按需组件下载器。

整合包/GitHub 仓库本身不含模型与环境；首次使用时按需下载：

  env      内嵌 Python 环境（torch cu128 + 依赖）      必需，首次启动下载
  llm      Qwen3.5-9B 本地大模型 (5.7GB)               必需（或改用云端 provider）
  tts      Ming-3.0-tts-0.5B 语音合成 (3.8GB)          必需（或改用 edgetts）
  avatar   wav2lip 模型 + 数字人形象 (约 550MB)         必需
  asr      Ming-3.0-asr-tiny 语音识别 (16.6GB)         懒加载：首次语音对话才下载
  asr-env  ASR 独立运行环境 (约 5GB)                   懒加载：随 asr 一起

用法:
  python setup_components.py status            # 查看各组件状态
  python setup_components.py download env llm tts avatar
  python setup_components.py download asr      # 语音对话前手动预下载
"""

import os
import sys
import subprocess
import zipfile
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

PY_EMBED_URL = 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip'
GET_PIP_URL = 'https://bootstrap.pypa.io/get-pip.py'
GGUF_URLS = [
    'https://hf-mirror.com/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf',
    'https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf',
]
GGUF_NAME = 'Qwen3.5-9B-Q4_K_M.gguf'
NUGET_PY_URL = 'https://api.nuget.org/v3-flatcontainer/python/3.12.10/python.3.12.10.nupkg'

AVATAR_URLS = [
    # 官方 LiveTalking 模型（Google Drive）
    ('https://drive.google.com/uc?id=1wu6XujFL9rF-0P2l44G6kpeapeY0cME7', 'wav2lip256.pth'),
    ('https://drive.google.com/uc?id=1aU-9SMEAZWN00hbvAGlRHG2iB17r9dEW', 'wav2lip256_avatar1.tar.gz'),
]

MS_TTS = 'inclusionAI/Ming-3.0-tts-0.5B'
MS_ASR = 'inclusionAI/Ming-3.0-asr-tiny'

COMPONENT_INFO = {
    'env':      ('内嵌 Python 环境与依赖 (torch cu128 等)', '约 15GB 下载 / 22GB 占用'),
    'llm':      ('Qwen3.5-9B 本地大模型', '约 5.7GB'),
    'tts':      ('Ming-3.0-TTS 语音合成', '约 3.8GB'),
    'avatar':   ('wav2lip 口型模型 + 数字人形象', '约 550MB'),
    'asr':      ('Ming-3.0-ASR 语音识别 + 独立运行环境', '约 22GB（语音对话才需要）'),
}


def log(msg):
    print(msg, flush=True)


def is_installed(name: str) -> bool:
    if name == 'env':
        return os.path.isfile('runtime/.env_ready')
    if name == 'llm':
        return os.path.isfile(os.path.join('models', 'llm', GGUF_NAME))
    if name == 'tts':
        return os.path.isfile(os.path.join('models', 'ming-tts', '0.5B', 'model.safetensors'))
    if name == 'avatar':
        return os.path.isfile('models/wav2lip.pth') and os.path.isdir('data/avatars/wav2lip256_avatar1')
    if name == 'asr':
        return (os.path.isfile(os.path.join('models', 'ming-asr', 'asr-tiny', 'model-00002-of-00002.safetensors'))
                and os.path.isdir('bin/asr-python'))
    return False


# ────────────────────────── env：内嵌 Python + 依赖 ──────────────────────────

def _ensure_base_python():
    """确保 runtime/python 存在（embeddable + pip）。"""
    if os.path.isfile('runtime/python/python.exe') and os.path.isfile(
            'runtime/python/Scripts/pip.exe'):
        return
    log('[setup] 下载内嵌 Python 3.12 ...')
    os.makedirs('runtime', exist_ok=True)
    zip_path = 'runtime/python-embed.zip'
    urllib.request.urlretrieve(PY_EMBED_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall('runtime/python')
    os.remove(zip_path)
    with open('runtime/python/python312._pth', 'w', encoding='utf-8') as f:
        f.write('python312.zip\n.\nLib/site-packages\n\nimport site\n')
    urllib.request.urlretrieve(GET_PIP_URL, 'runtime/get-pip.py')
    subprocess.check_call(['runtime/python/python.exe', 'runtime/get-pip.py',
                           '--no-warn-script-location', '-q'])
    log('[setup] 内嵌 Python 就绪')


def install_env():
    """安装主运行环境：torch cu128 + requirements + sitecustomize + triton 头文件。"""
    _ensure_base_python()
    py = 'runtime/python/python.exe'
    marker = 'runtime/.env_ready'
    if os.path.isfile(marker):
        return
    log('[setup] 安装 PyTorch (CUDA 12.8，约 3GB 下载) ...')
    subprocess.check_call([py, '-m', 'pip', 'install', '-q',
                           'torch==2.9.1', 'torchvision==0.24.1', 'torchaudio==2.9.1',
                           '--index-url', 'https://download.pytorch.org/whl/cu128',
                           '--no-warn-script-location'])
    log('[setup] 安装项目依赖 ...')
    subprocess.check_call([py, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt',
                           'gdown', '--no-warn-script-location'])
    # sitecustomize：把包根加入 sys.path
    sc = 'runtime/python/Lib/site-packages/sitecustomize.py'
    if not os.path.exists(sc):
        with open(sc, 'w', encoding='utf-8') as f:
            f.write(
                "# LiveTalking 整合包：把包根目录加入 sys.path\n"
                "import os\nimport sys\n\n"
                "_root = os.path.abspath(os.path.join(sys.prefix, '..', '..'))\n"
                "if os.path.isfile(os.path.join(_root, 'app.py')) and _root not in sys.path:\n"
                "    sys.path.insert(0, _root)\n")
    # triton 头文件（CPython Include + python312.lib）
    _install_triton_headers(py)
    with open(marker, 'w') as f:
        f.write('ok')
    log('[setup] ✅ 环境安装完成')


def _install_triton_headers(py):
    """triton 编译驱动绑定需要 CPython 头文件，内嵌版不带，从 NuGet 包补齐。"""
    import shutil
    site = 'runtime/python/Lib/site-packages'
    if not os.path.isdir(os.path.join(site, 'triton')):
        return
    if os.path.isfile(os.path.join(site, 'triton', 'backends', 'nvidia', 'lib', 'python312.lib')):
        return
    log('[setup] 补齐 triton 所需的 CPython 头文件 ...')
    nupkg = 'runtime/python.nupkg'
    urllib.request.urlretrieve(NUGET_PY_URL, nupkg)
    with zipfile.ZipFile(nupkg) as z:
        members = [n for n in z.namelist()
                   if n.startswith('tools/include/') or n.startswith('tools/libs/python312.lib')
                   or n.startswith('tools/libs/libpython')]
        z.extractall('runtime/nupkg', members=members)
    os.remove(nupkg)
    inc_src = 'runtime/nupkg/tools/include'
    if os.path.isdir(inc_src):
        shutil.copytree(inc_src, 'runtime/python/Include', dirs_exist_ok=True)
        lib_dst = os.path.join(site, 'triton', 'backends', 'nvidia', 'lib')
        os.makedirs(lib_dst, exist_ok=True)
        shutil.copy('runtime/nupkg/tools/libs/python312.lib', lib_dst)
        os.makedirs('runtime/python/libs', exist_ok=True)
        shutil.copy('runtime/nupkg/tools/libs/python312.lib', 'runtime/python/libs')
    shutil.rmtree('runtime/nupkg', ignore_errors=True)


# ────────────────────────── 模型组件 ──────────────────────────

def install_llm():
    if is_installed('llm'):
        return
    os.makedirs('models/llm', exist_ok=True)
    dest = os.path.join('models', 'llm', GGUF_NAME)
    for url in GGUF_URLS:
        log(f'[setup] 下载本地大模型 ({GGUF_NAME} 5.7GB) ...')
        try:
            urllib.request.urlretrieve(url, dest)
            log('[setup] ✅ 大模型下载完成')
            return
        except Exception as e:
            log(f'[setup] 源 {url.split("/")[2]} 失败: {e}，尝试下一个源')
    raise RuntimeError('大模型所有下载源均失败，请检查网络或手动下载放置到 models/llm/')


def install_tts():
    if is_installed('tts'):
        return
    log(f'[setup] 下载 Ming-3.0-TTS ({MS_TTS} 3.8GB) ...')
    from modelscope import snapshot_download
    snapshot_download(MS_TTS, local_dir='models/ming-tts/0.5B')
    log('[setup] ✅ 语音合成模型下载完成')


def install_avatar():
    if is_installed('avatar'):
        return
    log('[setup] 下载 wav2lip 模型与数字人形象 (约 550MB) ...')
    try:
        import gdown
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'])
        import gdown
    os.makedirs('models', exist_ok=True)
    for url, name in AVATAR_URLS:
        target = os.path.join('downloads', name)
        os.makedirs('downloads', exist_ok=True)
        if not os.path.isfile(target):
            gdown.download(url, target, quiet=False)
    if not os.path.isfile('models/wav2lip.pth'):
        import shutil
        shutil.copy('downloads/wav2lip256.pth', 'models/wav2lip.pth')
    if not os.path.isdir('data/avatars/wav2lip256_avatar1'):
        import tarfile
        os.makedirs('data/avatars', exist_ok=True)
        with tarfile.open('downloads/wav2lip256_avatar1.tar.gz') as t:
            try:
                t.extractall('data/avatars', filter='data')
            except TypeError:
                t.extractall('data/avatars')
    log('[setup] ✅ 数字人形象就绪')


def install_asr_env():
    """构建 ASR 独立解释器 bin/asr-python（CUDA torch + transformers 5.12 + fla/triton）。"""
    if os.path.isdir('bin/asr-python') and os.path.isfile('bin/asr-python/.ready'):
        return
    log('[setup] 构建 ASR 独立运行环境（约 5GB，首次语音对话需要）...')
    os.makedirs('bin', exist_ok=True)
    asr_py = 'bin/asr-python/python.exe'
    if not os.path.isfile(asr_py):
        zip_path = 'bin/asr-python.zip'
        urllib.request.urlretrieve(PY_EMBED_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall('bin/asr-python')
        os.remove(zip_path)
        with open('bin/asr-python/python312._pth', 'w', encoding='utf-8') as f:
            f.write('python312.zip\n.\nLib/site-packages\n\nimport site\n')
        urllib.request.urlretrieve(GET_PIP_URL, 'bin/get-pip.py')
        subprocess.check_call([asr_py, 'bin/get-pip.py', '-q', '--no-warn-script-location'])
        os.remove('bin/get-pip.py')
    pkgs = [
        ['torch==2.9.1', 'torchaudio==2.9.1', '--index-url', 'https://download.pytorch.org/whl/cu128'],
        ['transformers==5.12.1', 'accelerate', 'soundfile', 'numpy', 'einops',
         'triton-windows', 'fla-core'],
    ]
    for p in pkgs:
        subprocess.check_call([asr_py, '-m', 'pip', 'install', '-q', *p,
                               '--no-warn-script-location'])
    # triton 头文件
    site = 'bin/asr-python/Lib/site-packages'
    log('[setup] 补齐 ASR 环境 triton 头文件 ...')
    nupkg = 'bin/python.nupkg'
    urllib.request.urlretrieve(NUGET_PY_URL, nupkg)
    import shutil
    with zipfile.ZipFile(nupkg) as z:
        members = [n for n in z.namelist()
                   if n.startswith('tools/include/') or n.startswith('tools/libs/python312.lib')]
        z.extractall('bin/nupkg', members=members)
    os.remove(nupkg)
    shutil.copytree('bin/nupkg/tools/include', 'bin/asr-python/Include', dirs_exist_ok=True)
    lib_dst = os.path.join(site, 'triton', 'backends', 'nvidia', 'lib')
    os.makedirs(lib_dst, exist_ok=True)
    shutil.copy('bin/nupkg/tools/libs/python312.lib', lib_dst)
    os.makedirs('bin/asr-python/libs', exist_ok=True)
    shutil.copy('bin/nupkg/tools/libs/python312.lib', 'bin/asr-python/libs')
    shutil.rmtree('bin/nupkg', ignore_errors=True)
    with open('bin/asr-python/.ready', 'w') as f:
        f.write('ok')
    log('[setup] ✅ ASR 独立环境就绪')


def install_asr():
    log(f'[setup] 下载 Ming-3.0-ASR ({MS_ASR} 16.6GB，请耐心等待) ...')
    from modelscope import snapshot_download
    snapshot_download(MS_ASR, local_dir='models/ming-asr/asr-tiny')
    install_asr_env()
    log('[setup] ✅ 语音识别组件就绪')


INSTALLERS = {
    'env': install_env,
    'llm': install_llm,
    'tts': install_tts,
    'avatar': install_avatar,
    'asr': install_asr,
}


def ensure(components):
    """按顺序确保组件可用，返回失败的组件列表。"""
    failed = []
    for name in components:
        if is_installed(name):
            continue
        info, size = COMPONENT_INFO[name]
        log(f'\n[setup] 组件 [{name}] 缺失：{info}（{size}）')
        try:
            INSTALLERS[name]()
        except Exception as e:
            log(f'[setup] ❌ 组件 [{name}] 安装失败：{e}')
            failed.append(name)
    return failed


def ensure_core(auto=True):
    """启动必需组件：env/llm/tts/avatar。返回失败列表。"""
    return ensure(['env', 'llm', 'tts', 'avatar']) if auto else (
        [c for c in ('env', 'llm', 'tts', 'avatar') if not is_installed(c)])


def status():
    log('组件状态：')
    for name, (info, size) in COMPONENT_INFO.items():
        mark = '✅ 已安装' if is_installed(name) else '⬜ 未安装'
        log(f'  [{name:6s}] {mark}  {info}（{size}）')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'status'
    if cmd == 'status':
        status()
    elif cmd == 'download':
        failed = ensure(sys.argv[2:])
        sys.exit(1 if failed else 0)
    elif cmd == 'core':
        failed = ensure_core()
        sys.exit(1 if failed else 0)
    else:
        log('用法: python setup_components.py [status|download <组件...>|core]')
