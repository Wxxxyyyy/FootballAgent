# -*- coding: utf-8 -*-
"""
LLM 统一调度工具
═══════════════════
- 从 .env 中读取所有可用模型（平级配置，不分主/从/轻量）
- 调用方直接传入模型变量名，指定想用哪个模型
- 若远程 API 调用失败，自动降级到本地 Ollama 备用模型
- 提供 get_llm() 和 llm_call() 两个统一接口

使用方式：
    from common.llm_select import llm_call, LLM_MODEL_QWEN_SIMPLE_NAME

    # 直接指定模型名
    response = llm_call("你好", model=LLM_MODEL_QWEN_SIMPLE_NAME)

    # 换个模型也行
    from common.llm_select import LLM_MODEL_GLM_NAME
    response = llm_call("你好", model=LLM_MODEL_GLM_NAME)

    # 强制走本地 Ollama
    response = llm_call("你好", force_fallback=True)
"""

import os
import time
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage

# 加载 .env 配置
load_dotenv(override=True)

# ═══════════════════════════════════════════════════════════════
#  远程 API 公共配置（所有模型共享同一个 API Key 和 Base URL）
# ═══════════════════════════════════════════════════════════════

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://coding.dashscope.aliyuncs.com/v1")

# ═══════════════════════════════════════════════════════════════
#  可用模型列表（平级，从 .env 读取，调用方按需选择）
# ═══════════════════════════════════════════════════════════════

LLM_MODEL_QWEN_NAME = os.getenv("LLM_MODEL_QWEN_NAME", "qwen3.5-plus")
LLM_MODEL_QWEN_SIMPLE_NAME = os.getenv("LLM_MODEL_QWEN_SIMPLE_NAME", "qwen3-coder-next")
LLM_MODEL_GLM_NAME = os.getenv("LLM_MODEL_GLM_NAME", "glm-5")
LLM_MODEL_KIMI_NAME = os.getenv("LLM_MODEL_KIMI_NAME", "kimi-k2.5")
LLM_MODEL_MINIMAX_NAME = os.getenv("LLM_MODEL_MINIMAX_NAME", "MiniMax-M2.5")

# 所有远程模型变量名 → 值的映射（方便遍历 & 测试）
ALL_REMOTE_MODELS: dict[str, str] = {
    "LLM_MODEL_QWEN_NAME": LLM_MODEL_QWEN_NAME,
    "LLM_MODEL_QWEN_SIMPLE_NAME": LLM_MODEL_QWEN_SIMPLE_NAME,
    "LLM_MODEL_GLM_NAME": LLM_MODEL_GLM_NAME,
    "LLM_MODEL_KIMI_NAME": LLM_MODEL_KIMI_NAME,
    "LLM_MODEL_MINIMAX_NAME": LLM_MODEL_MINIMAX_NAME,
}

# ═══════════════════════════════════════════════════════════════
#  本地 Ollama 备用模型
# ═══════════════════════════════════════════════════════════════

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")

# ═══════════════════════════════════════════════════════════════
#  超时与重试配置
# ═══════════════════════════════════════════════════════════════

PRIMARY_TIMEOUT = 30
PRIMARY_MAX_RETRIES = 1


# ═══════════════════════════════════════════════════════════════
#  模型实例缓存（按模型名缓存，避免重复构造）
# ═══════════════════════════════════════════════════════════════

_remote_cache: dict[str, BaseChatModel] = {}
_fallback_instance: BaseChatModel | None = None


def _build_remote_llm(model_name: str) -> BaseChatModel:
    """构造远程 API 模型实例（阿里百炼平台，走 OpenAI 兼容协议）"""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model_name,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        timeout=PRIMARY_TIMEOUT,
        max_retries=PRIMARY_MAX_RETRIES,
        temperature=0.7,
    )


def _get_remote_llm(model_name: str) -> BaseChatModel:
    """获取远程模型单例（按模型名缓存）"""
    if model_name not in _remote_cache:
        _remote_cache[model_name] = _build_remote_llm(model_name)
    return _remote_cache[model_name]


def _build_fallback_llm() -> BaseChatModel:
    """构造备用模型实例（本地 Ollama）"""
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=OLLAMA_CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.7,
    )


def _get_fallback_llm() -> BaseChatModel:
    """获取备用模型单例"""
    global _fallback_instance
    if _fallback_instance is None:
        _fallback_instance = _build_fallback_llm()
    return _fallback_instance


# ═══════════════════════════════════════════════════════════════
#  核心接口：get_llm() — 获取 LLM 实例
# ═══════════════════════════════════════════════════════════════

def get_llm(
    model: str = LLM_MODEL_QWEN_SIMPLE_NAME,
    force_fallback: bool = False,
) -> BaseChatModel:
    """
    获取 LLM 实例（统一入口）。

    Args:
        model:          模型名称（直接传 .env 中的变量值，如 LLM_MODEL_QWEN_SIMPLE_NAME）
        force_fallback: 是否强制使用本地 Ollama 备用模型

    Returns:
        BaseChatModel: LangChain Chat 模型实例
    """
    if force_fallback:
        print(f"[LLM] 强制使用备用模型: Ollama ({OLLAMA_CHAT_MODEL})")
        return _get_fallback_llm()

    print(f"[LLM] 使用远程模型: {model}")
    return _get_remote_llm(model)


# ═══════════════════════════════════════════════════════════════
#  核心接口：llm_call() — 带自动降级的调用
# ═══════════════════════════════════════════════════════════════

def llm_call(
    prompt: str | list,
    model: str = LLM_MODEL_QWEN_SIMPLE_NAME,
    temperature: float | None = None,
    force_fallback: bool = False,
) -> AIMessage:
    """
    调用 LLM 并自动降级。

    流程：
      1. 使用指定的远程模型调用
      2. 若远程调用失败，自动降级到本地 Ollama
      3. 若备用模型也失败，抛出异常

    Args:
        prompt:         用户提示词（字符串或 LangChain Message 列表）
        model:          模型名称（直接传 .env 中读取的变量值）
        temperature:    温度参数（可选，覆盖默认值）
        force_fallback: 是否强制跳过远程模型直接使用 Ollama

    Returns:
        AIMessage: LLM 的回复消息
    """
    # 统一处理输入格式
    if isinstance(prompt, str):
        messages = [HumanMessage(content=prompt)]
    else:
        messages = prompt

    # ── 如果强制使用备用模型 ──
    if force_fallback:
        print(f"[LLM] 强制使用备用模型: Ollama ({OLLAMA_CHAT_MODEL})")
        fallback = _get_fallback_llm()
        return fallback.invoke(messages)

    # ── 尝试远程模型 ──
    active_llm = _get_remote_llm(model)
    try:
        print(f"[LLM] 正在调用远程模型: {model}...")
        start = time.time()
        response = active_llm.invoke(messages)
        elapsed = time.time() - start
        print(f"[LLM] 远程模型 {model} 响应成功 ({elapsed:.2f}s)")
        return response

    except Exception as e:
        print(f"[LLM] ⚠️ 远程模型 {model} 调用失败: {type(e).__name__}: {e}")
        print(f"[LLM] 正在降级到备用模型: Ollama ({OLLAMA_CHAT_MODEL})...")

        # ── 降级到备用模型 ──
        try:
            fallback = _get_fallback_llm()
            start = time.time()
            response = fallback.invoke(messages)
            elapsed = time.time() - start
            print(f"[LLM] 备用模型响应成功 ({elapsed:.2f}s)")
            return response

        except Exception as fallback_err:
            print(f"[LLM] ❌ 备用模型也失败: {type(fallback_err).__name__}: {fallback_err}")
            raise RuntimeError(
                f"远程模型({model})和备用模型(Ollama {OLLAMA_CHAT_MODEL})均不可用。\n"
                f"远程模型错误: {e}\n"
                f"备用模型错误: {fallback_err}"
            ) from fallback_err


# ═══════════════════════════════════════════════════════════════
#  辅助接口：检测模型可用性
# ═══════════════════════════════════════════════════════════════

def check_model_status(model_name: str) -> dict:
    """
    检测单个远程模型的可用性。

    Args:
        model_name: 模型名称

    Returns:
        dict: {"available": bool, "model": str, "latency": float | None, "error": str | None}
    """
    result = {"available": False, "model": model_name, "latency": None, "error": None}
    test_msg = [HumanMessage(content="你好，请回复ok")]

    try:
        llm = _get_remote_llm(model_name)
        start = time.time()
        llm.invoke(test_msg)
        result["available"] = True
        result["latency"] = round(time.time() - start, 2)
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def check_all_status() -> dict:
    """
    检测所有远程模型 + Ollama 的可用性。

    Returns:
        dict: {变量名: {available, model, latency, error}, ..., "ollama": {...}}
    """
    results = {}
    test_msg = [HumanMessage(content="你好，请回复ok")]

    # 所有远程模型
    for var_name, model_name in ALL_REMOTE_MODELS.items():
        results[var_name] = check_model_status(model_name)

    # Ollama
    ollama_result = {"available": False, "model": OLLAMA_CHAT_MODEL, "latency": None, "error": None}
    try:
        fallback = _get_fallback_llm()
        start = time.time()
        fallback.invoke(test_msg)
        ollama_result["available"] = True
        ollama_result["latency"] = round(time.time() - start, 2)
    except Exception as e:
        ollama_result["error"] = f"{type(e).__name__}: {e}"
    results["OLLAMA"] = ollama_result

    return results


# ═══════════════════════════════════════════════════════════════
#  命令行测试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  LLM 统一调度工具 · 测试")
    print("=" * 60)
    print(f"  API Base URL: {LLM_BASE_URL}")
    print(f"  可用远程模型:")
    for var_name, model_name in ALL_REMOTE_MODELS.items():
        print(f"    {var_name} = {model_name}")
    print(f"  备用模型: Ollama {OLLAMA_CHAT_MODEL} ({OLLAMA_BASE_URL})")
    print("=" * 60)

    # 快速连通性测试
    print("\n[测试] 检测所有模型可用性...")
    status = check_all_status()
    for key, info in status.items():
        tag = "✅" if info["available"] else "❌"
        latency_str = f" ({info['latency']}s)" if info["latency"] else ""
        error_str = f" | 错误: {info['error']}" if info["error"] else ""
        print(f"  {tag} {key}: {info['model']}{latency_str}{error_str}")

    print("\n[完成]")
