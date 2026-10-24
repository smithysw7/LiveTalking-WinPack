import time
import os
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar
from utils.logger import logger

# Built-in LLM providers. Each exposes an OpenAI-compatible chat completions
# endpoint; the active one is selected with --llm_provider (default: localllm).
#
# "localllm"  — bundled llama.cpp llama-server.exe started by the launcher
#               script (configurable via LLM_BASE_URL, default local port).
# "custom"    — any OpenAI-compatible endpoint: LLM_BASE_URL + LLM_API_KEY,
#               model name via --llm_model or LLM_MODEL.
LLM_PROVIDERS = {
    "localllm": {
        "api_key_env": "LOCAL_LLM_API_KEY",
        "base_url": "http://127.0.0.1:8765/v1",
        "base_url_env": "LLM_BASE_URL",
        "default_model": "Qwen3.5-9B",
    },
    "dashscope": {
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus",
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
    },
    "zhipu": {
        "api_key_env": "ZHIPU_API_KEY",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "glm-5.1",
    },
    "custom": {
        "api_key_env": "LLM_API_KEY",
        "base_url": "",
        "base_url_env": "LLM_BASE_URL",
        "default_model": "gpt-4o-mini",
    },
    "orcarouter": {
        "api_key_env": "ORCAROUTER_API_KEY",
        "base_url": "https://api.orcarouter.ai/v1",
        "default_model": "orcarouter/auto",
    },
}

DEFAULT_SYSTEM_PROMPT = '你是一个知识助手，尽量以简短、口语化的方式输出'


def _llm_provider(opt) -> str:
    """Return the configured provider name, defaulting to localllm."""
    return getattr(opt, 'llm_provider', 'localllm') or 'localllm'


def _provider_cfg(opt) -> dict:
    name = _llm_provider(opt)
    cfg = LLM_PROVIDERS.get(name)
    if cfg is None:
        logger.warning(f"Unknown llm_provider '{name}', falling back to localllm")
        cfg = LLM_PROVIDERS['localllm']
    return cfg


def _llm_client(opt):
    """Create the OpenAI-compatible client for the configured provider."""
    from openai import OpenAI
    cfg = _provider_cfg(opt)
    base_url = cfg['base_url']
    base_url_env = cfg.get('base_url_env')
    if base_url_env and os.getenv(base_url_env):
        base_url = os.getenv(base_url_env)
    if not base_url:
        raise ValueError(
            f"llm_provider '{_llm_provider(opt)}' 需要 {base_url_env} 环境变量"
            "（可在 .env 文件中配置）"
        )
    return OpenAI(
        api_key=os.getenv(cfg['api_key_env']) or 'local',
        base_url=base_url,
    )


def _llm_model(opt) -> str:
    """Resolve the model name, falling back to the provider default."""
    cfg = _provider_cfg(opt)
    return getattr(opt, 'llm_model', '') or cfg['default_model']


def _llm_system_prompt(opt) -> str:
    env_prompt = os.getenv('LLM_SYSTEM_PROMPT')
    if env_prompt:
        return env_prompt
    return getattr(opt, 'llm_system_prompt', '') or DEFAULT_SYSTEM_PROMPT


def llm_response(message,avatar_session:'BaseAvatar',datainfo:dict={}):
    try:
        opt = avatar_session.opt
        start = time.perf_counter()
        client = _llm_client(opt)
        model = _llm_model(opt)
        end = time.perf_counter()
        logger.info(f"llm Time init: {end-start}s,{message}")
        completion = client.chat.completions.create(
            model=model,
            messages=[{'role': 'system', 'content': _llm_system_prompt(opt)},
                    {'role': 'user', 'content': message}],
            stream=True,
            # Display token usage in the last line of the streamed response.
            stream_options={"include_usage": True}
        )
        result=""
        first = True
        for chunk in completion:
            if len(chunk.choices)>0:
                #print(chunk.choices[0].delta.content)
                if first:
                    end = time.perf_counter()
                    logger.info(f"llm Time to first chunk: {end-start}s")
                    first = False
                msg = chunk.choices[0].delta.content
                if msg is None:
                    continue
                lastpos=0
                #msglist = re.split('[,.!;:，。！?]',msg)
                for i, char in enumerate(msg):
                    if char in ",.!;:，。！？：；" :
                        result = result+msg[lastpos:i+1]
                        lastpos = i+1
                        if len(result)>10:
                            logger.info(result)
                            avatar_session.put_msg_txt(result,datainfo)
                            result=""
                result = result+msg[lastpos:]
        end = time.perf_counter()
        logger.info(f"llm Time to last chunk: {end-start}s")
        if result:
            avatar_session.put_msg_txt(result,datainfo)

    except Exception as e:
        if _llm_provider(opt) == 'localllm':
            base = os.getenv('LLM_BASE_URL', 'http://127.0.0.1:8765/v1')
            logger.error(
                f"本地大模型服务({base})连接失败：{e}。"
                "请先运行「启动数字人.bat」以自动拉起 llama-server，"
                "或在 .env 中改用云端 provider(dashscope/deepseek/zhipu/custom)。"
            )
        else:
            logger.exception('llm exceptiopn:')
        return
