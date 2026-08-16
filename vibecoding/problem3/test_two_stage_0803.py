# -*- coding: utf-8 -*-
"""
文件: test_two_stage_0803.py
功能: 单日（2026-08-03）两层求解验证脚本：确认第一层严格 T* 与第二层
      T<=T* 硬约束下二级优化均正常工作
"""

import os
import sys
import time
from datetime import date

_HERE = os.path.dirname(os.path.abspath(__file__))   # 本文件所在目录 problem3/
_VIBE = os.path.dirname(_HERE)                        # vibecoding/
sys.path.insert(0, _VIBE)
sys.path.insert(0, _HERE)

from problem3_solution import (Q3DataLoader, Q3DemandDayAllocator,
                               Q3RouteGenerator, Q3DailySolver,
                               PLANNING_DATES)


def main():
    """主函数：08-03 单日两层求解并校验 T <= T*"""
    loader = Q3DataLoader()
    distances = loader.load_distances()
    all_demands = loader.load_demands_q3()
    non_temp = [d for d in all_demands if d['task_type'] != 'temporary']
    allocation, _ = Q3DemandDayAllocator.allocate(non_temp, PLANNING_DATES)

    day = date(2026, 8, 3)
    day_demands = [d for d in non_temp if allocation[d['person_id']] == day]
    gen = Q3RouteGenerator(distances)
    routes_info = gen.generate_daily_routes(day_demands, day, route_id_start=1)
    print('候选路径:', len(routes_info))

    Q3DailySolver.TWO_STAGE_TIME_LIMITS = [7200, 7200]
    Q3DailySolver.USE_ADAPTIVE = False
    Q3DailySolver.USE_TWO_STAGE = True
    Q3DailySolver.LOG_TO_CONSOLE = 1
    solver = Q3DailySolver()
    t0 = time.time()
    result = solver.solve(routes_info, day_demands, day, distances)
    print('总耗时 %.1fs' % (time.time() - t0))
    sel, yo, yi, ys, t_star = result
    t_actual = sum(ri['route'].time for ri in sel)
    n_people = sum(yo.values()) + sum(yi.values()) + sum(ys.values())
    print('T*=%d 实际T=%d 架次=%d 人=%d' % (t_star, t_actual, len(sel), n_people))
    assert t_actual <= t_star, '第二层违反时间硬约束!'
    assert n_people == len(day_demands), '需求未全覆盖!'
    print('单日验证通过')


if __name__ == '__main__':
    main()
