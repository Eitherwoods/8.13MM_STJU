# -*- coding: utf-8 -*-
"""
问题3代码包：带时间要求的多日排班
核心:
  - problem3_service_day.py: 最终求解器（服务日入列 + 事件驱动时刻 + 覆盖兜底 + 逐日阶段二）
推广实验（适配新框架）:
  - robust_dynamic_test.py: 鲁棒性（可行集稳定率）+ 动态调度模拟
  - test/robust_demand.py: 需求人数波动下的单日鲁棒排班
"""
