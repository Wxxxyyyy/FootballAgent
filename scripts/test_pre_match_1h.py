# -*- coding: utf-8 -*-
"""
赛前1小时预测自动触发脚本

比赛: 厄瓜多尔 vs 德国
match_id: 2906956
开赛时间: 2026-06-26 04:00 (北京时间)

触发时间:
  - 03:00 (赛前1小时) — 测试第二档阵容采集（官方首发）
  - 03:30 (赛前30分钟) — 再次触发，检查首发是否有更新

运行方式:
  nohup python3 scripts/test_pre_match_1h.py > logs/pre_match_1h_test.log 2>&1 &

结果保存: data/predictions/ecuador_germany_1h_{timestamp}.json
"""

import os
import sys
import json
import time
import httpx
import traceback
from datetime import datetime, timedelta

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)

# ═══════════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════════

MATCH_ID = "2906956"
HOME_TEAM = "Ecuador"
AWAY_TEAM = "Germany"
DATE = "2026-06-26"
KICKOFF_TIME = datetime(2026, 6, 26, 4, 0)  # 北京时间 04:00

# 触发时间点（赛前 3.5h / 2.5h / 1.5h / 0.5h 四档）
TRIGGER_TIMES = [
    KICKOFF_TIME - timedelta(hours=3.5),  # 赛前3.5h
    KICKOFF_TIME - timedelta(hours=2.5),  # 赛前2.5h
    KICKOFF_TIME - timedelta(hours=1.5),  # 赛前1.5h
    KICKOFF_TIME - timedelta(hours=0.5),  # 赛前0.5h
]

OPENCLAW_URL = "http://127.0.0.1:9000"
PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, "data", "predictions")

# 确保目录存在
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)


def log(msg: str):
    """带时间戳的日志"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════
#  赔率获取
# ═══════════════════════════════════════════════════════════════

def fetch_odds_from_openclaw() -> dict:
    """从 OpenClaw 获取赔率快照"""
    try:
        resp = httpx.get(f"{OPENCLAW_URL}/odds/{MATCH_ID}", timeout=10)
        if resp.status_code == 200:
            return resp.json()
        else:
            log(f"  OpenClaw 赔率返回 {resp.status_code}")
            return {}
    except Exception as e:
        log(f"  OpenClaw 赔率获取失败: {e}")
        return {}


def build_mock_odds(odds_data: dict) -> dict:
    """将 OpenClaw 赔率快照转换为 advance_predictor 可用的格式"""
    if not odds_data:
        # 兜底：用之前的赔率
        return {
            "B365H": 3.6, "B365D": 4.0, "B365A": 1.91,
            "B365H_open": 2.7, "B365D_open": 3.4, "B365A_open": 2.5,
            "_source": "Bet365",
        }

    # OpenClaw 快照格式: B365H(初盘), B365CH(终盘) 等
    b365h_open = odds_data.get("B365H", 2.7)   # 初盘主胜
    b365d_open = odds_data.get("B365D", 3.4)   # 初盘平
    b365a_open = odds_data.get("B365A", 2.5)   # 初盘客胜
    b365h = odds_data.get("B365CH", b365h_open)  # 终盘主胜
    b365d = odds_data.get("B365CD", b365d_open)  # 终盘平
    b365a = odds_data.get("B365CA", b365a_open)  # 终盘客胜

    log(f"  初盘: H={b365h_open}, D={b365d_open}, A={b365a_open}")
    log(f"  终盘: H={b365h}, D={b365d}, A={b365a}")

    return {
        "B365H": float(b365h), "B365D": float(b365d), "B365A": float(b365a),
        "B365H_open": float(b365h_open),
        "B365D_open": float(b365d_open),
        "B365A_open": float(b365a_open),
        "_source": "Bet365",
    }


# ═══════════════════════════════════════════════════════════════
#  预测执行
# ═══════════════════════════════════════════════════════════════

def run_prediction(odds_data: dict, trigger_label: str) -> dict:
    """执行一次完整预测"""
    from agents.predicted_agent.advance_predictor import PreMatchPredictor
    from agents.predicted_agent.scouters import PreMatchIntel

    predictor = PreMatchPredictor()

    # Mock OpenClaw 请求（直接用赔率快照）
    predictor._request_openclaw = lambda home, away, date: {
        "odds": {"found": True, "league": "World Cup 2026"}
    }

    # Mock 赔率提取（用真实赔率）
    mock_odds = build_mock_odds(odds_data)
    predictor._extract_odds = lambda data, home, away: mock_odds

    # Mock 赛前情报：传入正确的 hours_to_kickoff（触发第二档阵容）
    def mock_gather_intel(home, away, date):
        intel = PreMatchIntel()
        now = datetime.now()
        hours_to_kickoff = max(0, (KICKOFF_TIME - now).total_seconds() / 3600)
        log(f"  hours_to_kickoff = {hours_to_kickoff:.2f}h")
        if hours_to_kickoff <= 1:
            log(f"  阵容档位: 第二档（官方首发）")
        else:
            log(f"  阵容档位: 第一档（缺阵预判）")
        return intel.gather(home, away, date, hours_to_kickoff=hours_to_kickoff)

    predictor._gather_pre_match_intel = mock_gather_intel

    try:
        result = predictor.predict(HOME_TEAM, AWAY_TEAM, DATE)
        result["trigger_label"] = trigger_label
        result["trigger_time"] = datetime.now().isoformat()
        return result
    finally:
        predictor.close()


def save_result(result: dict, trigger_label: str):
    """保存预测结果到 JSON"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ecuador_germany_{trigger_label}_{ts}.json"
    filepath = os.path.join(PREDICTIONS_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    log(f"  结果已保存: {filepath}")
    return filepath


def print_summary(result: dict):
    """打印预测摘要"""
    ml = result.get("ml_prediction", {})
    llm = result.get("llm_analysis", {})
    intel = result.get("pre_match_intel", {})

    log("--- 预测摘要 ---")
    if "error" in ml:
        log(f"  ML: {ml['error']}")
    else:
        log(f"  ML: 主胜={ml.get('home_win_prob',0):.1%} 平={ml.get('draw_prob',0):.1%} 客胜={ml.get('away_win_prob',0):.1%} 倾向={ml.get('wdl_prediction','?')}")

    # 赛前情报摘要
    summary = intel.get("summary", "")
    if summary:
        # 只打印阵容部分
        for line in summary.split("\n"):
            if "阵容" in line or "首发" in line or "缺阵" in line or "阵型" in line:
                log(f"  情报: {line.strip()}")

    log(f"  LLM综合: {str(llm.get('overall_analysis', 'N/A'))[:200]}")
    log(f"  耗时: {result.get('elapsed_seconds',0)}s")


# ═══════════════════════════════════════════════════════════════
#  主循环
# ═══════════════════════════════════════════════════════════════

def main():
    log("=" * 60)
    log("赛前1小时预测自动触发脚本")
    log(f"比赛: {HOME_TEAM} vs {AWAY_TEAM}")
    log(f"开赛时间: {KICKOFF_TIME.strftime('%Y-%m-%d %H:%M')}")
    log(f"触发时间: {[t.strftime('%H:%M') for t in TRIGGER_TIMES]}")
    log("=" * 60)

    # 检查 OpenClaw 是否在线
    try:
        resp = httpx.get(f"{OPENCLAW_URL}/health", timeout=5)
        log(f"OpenClaw 状态: {resp.json().get('status', 'unknown')}")
    except Exception as e:
        log(f"⚠️ OpenClaw 未响应: {e}")

    for i, target_time in enumerate(TRIGGER_TIMES):
        label = "1h" if i == 0 else "30m"

        # 等待到目标时间
        now = datetime.now()
        if now < target_time:
            wait_seconds = (target_time - now).total_seconds()
            log(f"等待到 {target_time.strftime('%H:%M')}（还需 {wait_seconds/3600:.1f}h）...")
            time.sleep(wait_seconds)

        log(f"\n{'='*60}")
        log(f"触发预测: 赛前{label} ({datetime.now().strftime('%H:%M:%S')})")
        log(f"{'='*60}")

        # 获取赔率
        log("获取赔率快照...")
        odds_data = fetch_odds_from_openclaw()
        if odds_data:
            log(f"  赔率: {json.dumps(odds_data, ensure_ascii=False)}")
        else:
            log("  ⚠️ 未获取到赔率，使用兜底数据")

        # 跑预测
        try:
            result = run_prediction(odds_data, label)
            save_result(result, label)
            print_summary(result)
        except Exception as e:
            log(f"  ❌ 预测失败: {e}")
            traceback.print_exc()

    log(f"\n{'='*60}")
    log("所有触发完成，脚本结束")
    log(f"结果目录: {PREDICTIONS_DIR}")
    log("=" * 60)


if __name__ == "__main__":
    main()
