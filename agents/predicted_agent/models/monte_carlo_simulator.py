# -*- coding: utf-8 -*-
"""
蒙特卡洛足球比赛模拟器

原理:
  不训练模型，而是用泊松分布模拟比赛进球数，重复上万次，统计胜平负概率。

  λ_home = 主队预期进球数
  λ_away = 客队预期进球数

  每次模拟:
    home_goals ~ Poisson(λ_home)
    away_goals ~ Poisson(λ_away)
    统计: 主胜 / 平局 / 客胜次数 → 胜平负概率

λ 值来源（无需训练数据）:
  从 Bet365 赔率反推隐含概率，再用 Dixon-Coles 方法分解为两队进攻强度。

  方法:
    1. 从 1X2 赔率得到 (p_h, p_d, p_a)
    2. 设主队预期进球 x, 客队预期进球 y
    3. 通过泊松分布计算 P(主胜|x,y), P(平|x,y), P(客胜|x,y)
    4. 搜索 (x, y) 使三个概率最接近 (p_h, p_d, p_a)
    5. 用最优 (x, y) = (λ_home, λ_away) 做蒙特卡洛模拟

使用方式:
  from agents.predicted_agent.models.monte_carlo_simulator import MonteCarloSimulator
  sim = MonteCarloSimulator()
  result = sim.simulate(b365h=1.5, b365d=4.2, b365a=6.0)
  # result = {"home_win_prob", "draw_prob", "away_win_prob",
  #           "score_distribution", "expected_goals_home", "expected_goals_away"}
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson


class MonteCarloSimulator:
    """蒙特卡洛足球比赛模拟器（基于泊松分布）"""

    def __init__(self, n_simulations: int = 10000, max_goals: int = 10):
        """
        Args:
            n_simulations: 蒙特卡洛模拟次数（默认 10000）
            max_goals: 单队最大进球数（用于概率截断，默认 10）
        """
        self.n_simulations = n_simulations
        self.max_goals = max_goals
        self.rng = np.random.default_rng(seed=42)

    # ═══════════════════════════════════════════════════════════
    #  赔率 → 隐含概率
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def odds_to_implied_probs(h: float, d: float, a: float) -> tuple:
        """
        赔率 → 去除 overround 的隐含概率

        Returns: (p_home, p_draw, p_away)
        """
        inv_h = 1.0 / h
        inv_d = 1.0 / d
        inv_a = 1.0 / a
        total = inv_h + inv_d + inv_a
        return inv_h / total, inv_d / total, inv_a / total

    # ═══════════════════════════════════════════════════════════
    #  从隐含概率反推 λ（预期进球数）
    # ═══════════════════════════════════════════════════════════

    def _poisson_outcome_probs(self, lam_h: float, lam_a: float) -> tuple:
        """
        给定 (λ_home, λ_away)，用泊松分布计算胜平负概率

        Returns: (p_home, p_draw, p_away)
        """
        p_home = 0.0
        p_draw = 0.0
        p_away = 0.0

        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                p = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
                if i > j:
                    p_home += p
                elif i == j:
                    p_draw += p
                else:
                    p_away += p

        # 归一化（截断可能导致总和 < 1）
        total = p_home + p_draw + p_away
        return p_home / total, p_draw / total, p_away / total

    def _loss_function(self, lambdas: np.ndarray, target_probs: np.ndarray) -> float:
        """
        损失函数: 泊松概率 vs 目标隐含概率的均方误差

        lambdas: [λ_home, λ_away]
        target_probs: [p_home, p_draw, p_away]
        """
        lam_h, lam_a = lambdas
        # 约束: λ > 0
        if lam_h <= 0 or lam_a <= 0:
            return 1e10

        pred_h, pred_d, pred_a = self._poisson_outcome_probs(lam_h, lam_a)
        target_h, target_d, target_a = target_probs

        return (pred_h - target_h) ** 2 + (pred_d - target_d) ** 2 + (pred_a - target_a) ** 2

    def infer_lambda(self, h_odds: float, d_odds: float, a_odds: float) -> tuple:
        """
        从 1X2 赔率反推两队预期进球数 (λ_home, λ_away)

        方法: 优化搜索使泊松分布胜平负概率最接近赔率隐含概率

        Returns: (λ_home, λ_away)
        """
        target = np.array(self.odds_to_implied_probs(h_odds, d_odds, a_odds))

        # 初始猜测: 强队 1.5 球，弱队 1.0 球
        x0 = np.array([1.5, 1.0])

        result = minimize(
            self._loss_function,
            x0,
            args=(target,),
            method="Nelder-Mead",
            options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 1000},
        )

        lam_h, lam_a = result.x
        return max(lam_h, 0.05), max(lam_a, 0.05)

    # ═══════════════════════════════════════════════════════════
    #  蒙特卡洛模拟
    # ═══════════════════════════════════════════════════════════

    def simulate(
        self,
        b365h: float,
        b365d: float,
        b365a: float,
        n_simulations: int = None,
    ) -> dict:
        """
        从赔率出发，蒙特卡洛模拟比赛结果

        Args:
            b365h/d/a: Bet365 胜平负赔率
            n_simulations: 模拟次数（None 用默认 10000）

        Returns:
            {
                "home_win_prob": 0.45,
                "draw_prob": 0.28,
                "away_win_prob": 0.27,
                "wdl_prediction": "H",
                "expected_goals_home": 1.5,
                "expected_goals_away": 1.0,
                "most_likely_score": "2:1",
                "score_distribution": {"2:1": 0.12, "1:1": 0.11, ...},
                "lambda_home": 1.5,
                "lambda_away": 1.0,
                "n_simulations": 10000,
            }
        """
        n = n_simulations or self.n_simulations

        # Step 1: 反推 λ
        lam_h, lam_a = self.infer_lambda(b365h, b365d, b365a)

        # Step 2: 蒙特卡洛模拟
        home_goals = self.rng.poisson(lam_h, n)
        away_goals = self.rng.poisson(lam_a, n)

        # Step 3: 统计结果
        home_wins = np.sum(home_goals > away_goals)
        draws = np.sum(home_goals == away_goals)
        away_wins = np.sum(home_goals < away_goals)

        p_home = home_wins / n
        p_draw = draws / n
        p_away = away_wins / n

        # 最可能的比分
        score_counts = {}
        for hg, ag in zip(home_goals, away_goals):
            key = f"{hg}:{ag}"
            score_counts[key] = score_counts.get(key, 0) + 1

        # 取概率最高的前 5 个比分
        top_scores = sorted(score_counts.items(), key=lambda x: -x[1])[:5]
        score_dist = {k: v / n for k, v in top_scores}

        most_likely = top_scores[0][0] if top_scores else "1:1"

        # 预测结果
        if p_home >= p_draw and p_home >= p_away:
            pred = "H"
        elif p_away >= p_home and p_away >= p_draw:
            pred = "A"
        else:
            pred = "D"

        return {
            "home_win_prob": round(p_home, 4),
            "draw_prob": round(p_draw, 4),
            "away_win_prob": round(p_away, 4),
            "wdl_prediction": pred,
            "expected_goals_home": round(float(lam_h), 3),
            "expected_goals_away": round(float(lam_a), 3),
            "most_likely_score": most_likely,
            "score_distribution": {k: round(v, 4) for k, v in score_dist.items()},
            "lambda_home": round(float(lam_h), 3),
            "lambda_away": round(float(lam_a), 3),
            "n_simulations": n,
        }

    def simulate_from_probs(
        self,
        p_home: float,
        p_draw: float,
        p_away: float,
        n_simulations: int = None,
    ) -> dict:
        """
        从已知的胜平负概率出发模拟（用于 RF 模型输出 → 比分分布）

        先反推等效赔率，再调用 simulate()
        """
        # 概率 → 等效赔率（假设 overround = 1.0）
        h_odds = 1.0 / max(p_home, 0.01)
        d_odds = 1.0 / max(p_draw, 0.01)
        a_odds = 1.0 / max(p_away, 0.01)
        return self.simulate(h_odds, d_odds, a_odds, n_simulations)


# ═══════════════════════════════════════════════════════════════
#  命令行测试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sim = MonteCarloSimulator(n_simulations=20000)

    print("=" * 60)
    print("  蒙特卡洛模拟器 · 测试")
    print("=" * 60)

    test_cases = [
        ("强主队", 1.50, 4.20, 6.00),
        ("势均力敌", 2.50, 3.20, 2.80),
        ("强客队", 4.00, 3.50, 1.85),
        ("超级热门", 1.20, 6.50, 12.00),
    ]

    for name, h, d, a in test_cases:
        print(f"\n--- {name}: {h}/{d}/{a} ---")
        result = sim.simulate(h, d, a)
        print(f"  胜平负概率: 主胜={result['home_win_prob']:.1%} "
              f"平={result['draw_prob']:.1%} 客胜={result['away_win_prob']:.1%}")
        print(f"  预测: {result['wdl_prediction']}")
        print(f"  预期进球: 主队={result['expected_goals_home']} "
              f"客队={result['expected_goals_away']}")
        print(f"  最可能比分: {result['most_likely_score']}")
        print(f"  Top5 比分: {result['score_distribution']}")
