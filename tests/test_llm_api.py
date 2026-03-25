# -*- coding: utf-8 -*-
"""
LLM API 可用性测试
═══════════════════════
- 支持测试 .env 中配置的任意远程模型
- 想测哪个模型，直接把变量名放到下方 TEST_MODELS 列表即可
- 也可以测试 Ollama 本地备用模型

使用方式:
    python tests/test_llm_api.py
"""

import sys
import os
import time

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_core.messages import HumanMessage
from common.llm_select import (
    # 公共配置
    LLM_API_KEY,
    LLM_BASE_URL,
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    # 所有可用远程模型（平级，按需选用）
    LLM_MODEL_QWEN_NAME,
    LLM_MODEL_QWEN_SIMPLE_NAME,
    LLM_MODEL_GLM_NAME,
    LLM_MODEL_KIMI_NAME,
    LLM_MODEL_MINIMAX_NAME,
    ALL_REMOTE_MODELS,
    # 核心调用
    llm_call,
    _build_remote_llm,
    _build_fallback_llm,
)


# ═══════════════════════════════════════════════════════════════
#  ✏️ 配置区：想测试哪些模型，把变量名放这里就行
# ═══════════════════════════════════════════════════════════════

TEST_MODELS = [
    # LLM_MODEL_QWEN_SIMPLE_NAME,   # qwen3-coder-next
    LLM_MODEL_QWEN_NAME,        # qwen3.5-plus（取消注释即可测试）
    # LLM_MODEL_GLM_NAME,         # glm-5
    # LLM_MODEL_KIMI_NAME,        # kimi-k2.5
    # LLM_MODEL_MINIMAX_NAME,     # MiniMax-M2.5
]

# 是否也测试 Ollama 本地备用模型
TEST_OLLAMA = True


# ═══════════════════════════════════════════════════════════════
#  测试函数
# ═══════════════════════════════════════════════════════════════

def test_env_config():
    """测试: 检查 .env 环境变量是否正确加载"""
    print("=" * 60)
    print("[检查] .env 环境变量配置")
    print("=" * 60)
    print(f"  LLM_API_KEY  : {LLM_API_KEY[:10]}...{LLM_API_KEY[-6:]}" if len(LLM_API_KEY) > 16 else f"  LLM_API_KEY  : {LLM_API_KEY}")
    print(f"  LLM_BASE_URL : {LLM_BASE_URL}")
    print()
    print("  可用远程模型:")
    for var_name, model_name in ALL_REMOTE_MODELS.items():
        print(f"    {var_name} = {model_name}")
    print(f"\n  Ollama 备用: {OLLAMA_CHAT_MODEL} ({OLLAMA_BASE_URL})")

    assert LLM_API_KEY, "❌ LLM_API_KEY 未配置"
    assert LLM_BASE_URL, "❌ LLM_BASE_URL 未配置"
    print("  ✅ 环境变量配置正常\n")


def test_remote_model(model_name: str):
    """测试单个远程模型的连通性 + 对话能力"""
    print("=" * 60)
    print(f"[测试] 远程模型: {model_name}")
    print("=" * 60)

    llm = _build_remote_llm(model_name)
    print(f"  模型:    {llm.model_name}")
    print(f"  BaseURL: {llm.openai_api_base}")

    # 连通性测试
    print("\n  ── 连通性测试 ──")
    test_msg = [HumanMessage(content="你好，请回复ok")]
    try:
        start = time.time()
        response = llm.invoke(test_msg)
        elapsed = time.time() - start
        print(f"  响应耗时: {elapsed:.2f}s")
        print(f"  回复内容: {response.content}")
        print("  ✅ 连通正常")
    except Exception as e:
        print(f"  ❌ 连通失败!")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误详情: {e}")
        print()
        return False

    # 对话能力测试
    print("\n  ── 对话能力测试 ──")
    chat_msg = [HumanMessage(content="用一句话介绍梅西")]
    try:
        start = time.time()
        response = llm.invoke(chat_msg)
        elapsed = time.time() - start
        print(f"  问题: '用一句话介绍梅西'")
        print(f"  回复: {response.content}")
        print(f"  耗时: {elapsed:.2f}s")
        token_usage = response.response_metadata.get('token_usage', '未知')
        print(f"  Token: {token_usage}")
        print("  ✅ 对话能力正常")
    except Exception as e:
        print(f"  ❌ 对话测试失败: {type(e).__name__}: {e}")
        return False

    # llm_call 降级测试
    print("\n  ── llm_call 降级机制测试 ──")
    try:
        start = time.time()
        response = llm_call("请用一个词回答: 1+1等于几？", model=model_name)
        elapsed = time.time() - start
        print(f"  回复: {response.content}")
        print(f"  耗时: {elapsed:.2f}s")
        print("  ✅ llm_call 调用正常")
    except Exception as e:
        print(f"  ❌ llm_call 调用失败: {type(e).__name__}: {e}")
        return False

    print()
    return True


def test_ollama():
    """测试 Ollama 本地备用模型"""
    print("=" * 60)
    print(f"[测试] Ollama 本地模型: {OLLAMA_CHAT_MODEL} ({OLLAMA_BASE_URL})")
    print("=" * 60)

    llm = _build_fallback_llm()

    test_msg = [HumanMessage(content="你好，请回复ok")]
    try:
        start = time.time()
        response = llm.invoke(test_msg)
        elapsed = time.time() - start
        print(f"  响应耗时: {elapsed:.2f}s")
        print(f"  回复内容: {response.content}")
        print("  ✅ Ollama 连通正常\n")
        return True
    except Exception as e:
        print(f"  ❌ Ollama 调用失败!")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误详情: {e}\n")
        return False


# ═══════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + " LLM API 可用性测试 ".center(56, "=") + "\n")

    # 检查环境变量
    test_env_config()

    # 测试指定的远程模型
    results = {}
    for model_name in TEST_MODELS:
        ok = test_remote_model(model_name)
        results[model_name] = ok

    # 测试 Ollama
    if TEST_OLLAMA:
        results[f"Ollama ({OLLAMA_CHAT_MODEL})"] = test_ollama()

    # 汇总
    print("=" * 60)
    print(" 测试汇总 ".center(56, "="))
    print("=" * 60)
    for name, ok in results.items():
        tag = "✅" if ok else "❌"
        print(f"  {tag} {name}")
    print("=" * 60)
    print("测试结束\n")
