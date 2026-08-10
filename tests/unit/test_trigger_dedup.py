# -*- coding: utf-8 -*-
"""
预测触发去重契约测试。

conftest 已把 Redis 指向不存在的端口 → 这里覆盖的是
"Redis 降级时纯文件去重"的兜底路径（与改造前行为一致，绝不漏/重触发）。
"""

import prediction_trigger as pt


class TestTriggerDedup:
    def test_mark_then_should_not_trigger(self):
        assert pt._should_trigger("m100", 3.5) is True
        pt._mark_triggered("m100", 3.5)
        assert pt._should_trigger("m100", 3.5) is False

    def test_different_trigger_point_still_triggers(self):
        pt._mark_triggered("m200", 3.5)
        assert pt._should_trigger("m200", 2.5) is True  # 不同触发点独立

    def test_tolerance_18min(self):
        pt._mark_triggered("m300", 3.5)
        assert pt._should_trigger("m300", 3.4) is False  # 容差内视为同一触发点
        assert pt._should_trigger("m300", 3.1) is True   # 容差外

    def test_unmark_allows_retry(self):
        """预测失败 → 移除标记 → 下次调度可重试（漏推事故的自愈路径）"""
        pt._mark_triggered("m400", 1.5)
        assert pt._should_trigger("m400", 1.5) is False
        pt._unmark_triggered("m400", 1.5)
        assert pt._should_trigger("m400", 1.5) is True

    def test_state_survives_reload(self):
        """状态持久化在文件，进程重启（重新读文件）后仍然去重"""
        pt._mark_triggered("m500", 0.5)
        state = pt._load_triggered_state()
        assert any(abs(t - 0.5) < 0.3 for t in state.get("m500", []))
