# -*- coding: utf-8 -*-
"""
文件: validate_alns_0803.py
功能: ALNS 热启动消融实验 — 2026-08-03 单日
项目: 2026策联杯 B题 - 海上油田人员直升机运载计划编排

实验设计:
  在相同候选路径池（routes_info）与相同求解时限下对比：
    A. 基线: COPT MIP 无热启动
    B. 实验: ALNS 搜索出的高质量解作为 COPT MipStart
  记录加权目标值、总时间、利用率、架次、燃油与求解耗时，
  用于论文"ALNS 热启动 vs 无热启动"的消融分析。

运行方式:
    E:\\Python\\python.exe validate_alns_0803.py
"""

import os
import sys
import time as time_module
from datetime import date

# ==================== 模块路径 ====================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from problem3_solution import (
    Q3DataLoader, Q3DemandDayAllocator, Q3RouteGenerator,
    Q3DailySolver, PLANNING_DATES,
)
from alns_improver import AlnsImprover
from config import DATA_DIR


# ==================== 全局配置 ====================
TARGET_DAY = date(2026, 8, 3)
MIP_TIME_LIMIT = 60          # 单次 MIP 求解时限（秒）
ALNS_ITERATIONS = 300        # ALNS 迭代次数
OUTPUT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_metrics(sel_routes, yo_vals, yi_vals, ys_vals, routes_info):
    """从求解结果提取 (总时间, 利用率, 架次, 燃油)"""
    idx_map = {id(ri): idx for idx, ri in enumerate(routes_info)}
    sel_idx = set(idx_map[id(ri)] for ri in sel_routes)

    total_time = sum(ri['route'].time for ri in sel_routes)
    total_fuel = sum(ri['route'].fuel for ri in sel_routes)
    seat_km = sum(ri['route'].capacity * ri['route'].total_distance for ri in sel_routes)

    pass_km = 0.0
    for (ri_idx, f), v in yo_vals.items():
        if ri_idx in sel_idx:
            pass_km += v * routes_info[ri_idx]['precomputed']['dist_to'].get(f, 0)
    for (ri_idx, f), v in yi_vals.items():
        if ri_idx in sel_idx:
            pass_km += v * routes_info[ri_idx]['precomputed']['dist_from'].get(f, 0)
    for (ri_idx, f1, f2), v in ys_vals.items():
        if ri_idx in sel_idx:
            si = routes_info[ri_idx]['precomputed']['shuttle_info'].get((f1, f2), {})
            pass_km += v * si.get('dist', 0)

    util = pass_km / seat_km if seat_km > 0 else 0.0
    return total_time, util, len(sel_routes), total_fuel


def main():
    """主函数：数据准备 -> ALNS 热启动 -> 消融实验 -> 结果对比"""
    print("=" * 70)
    print("ALNS 热启动消融实验 — 2026-08-03 单日")
    print("=" * 70)

    # ---------- 1. 数据加载与路径生成 ----------
    print("\n[1] 加载数据并生成候选路径...")
    loader = Q3DataLoader()
    distances = loader.load_distances()
    all_demands = loader.load_demands_q3()
    non_temp = [d for d in all_demands if d['task_type'] != 'temporary']

    allocation, _ = Q3DemandDayAllocator.allocate(non_temp, PLANNING_DATES)
    for d in non_temp:
        d['_assigned_day'] = allocation[d['person_id']]
    day_demands = [d for d in non_temp if allocation[d['person_id']] == TARGET_DAY]
    print(f"  {TARGET_DAY} 分配需求: {len(day_demands)} 人")

    t0 = time_module.time()
    generator = Q3RouteGenerator(distances)
    routes_info = generator.generate_daily_routes(day_demands, TARGET_DAY, route_id_start=1)
    t1 = time_module.time()
    print(f"  候选路径数: {len(routes_info)}  (生成耗时 {t1-t0:.1f}s)")

    # ---------- 2. MIP 加速设置 ----------
    Q3DailySolver.DINKELBACH_MAX_ITERS = 1
    Q3DailySolver.TIME_LIMITS = [MIP_TIME_LIMIT]
    Q3DailySolver.RETRY_TIME_LIMIT = MIP_TIME_LIMIT
    Q3DailySolver.USE_WEIGHTED = True

    # ---------- 3. ALNS 求热启动解 ----------
    print("\n[2] ALNS 搜索热启动解...")
    improver = AlnsImprover(routes_info, day_demands, distances, TARGET_DAY)
    t0 = time_module.time()
    alns_sol, alns_obj = improver.run(iterations=ALNS_ITERATIONS)
    t1 = time_module.time()
    warm_start = improver.to_mip_start(alns_sol)
    print(f"  ALNS 耗时 {t1-t0:.1f}s, 目标值 {alns_obj:.2f}, "
          f"选中路径 {len(alns_sol.selected)} 条")

    # ---------- 4. 实验 A: 无热启动 ----------
    print("\n[3] 实验 A: COPT MIP 无热启动（基线）...")
    solver_a = Q3DailySolver()
    t0 = time_module.time()
    result_a = solver_a.solve(routes_info, day_demands, TARGET_DAY, distances,
                              warm_start=None)
    t1 = time_module.time()
    time_a = t1 - t0
    if result_a is None:
        print("  实验 A 求解失败!")
        metrics_a = None
    else:
        sel_routes, yo, yi, ys = result_a
        metrics_a = extract_metrics(sel_routes, yo, yi, ys, routes_info)
        print(f"  A: 时间={metrics_a[0]}min, 利用率={metrics_a[1]*100:.2f}%, "
              f"架次={metrics_a[2]}, 燃油={metrics_a[3]:.1f}kg, 耗时={time_a:.1f}s")

    # ---------- 5. 实验 B: ALNS 热启动 ----------
    print("\n[4] 实验 B: COPT MIP + ALNS 热启动...")
    solver_b = Q3DailySolver()
    t0 = time_module.time()
    result_b = solver_b.solve(routes_info, day_demands, TARGET_DAY, distances,
                              warm_start=warm_start)
    t1 = time_module.time()
    time_b = t1 - t0
    if result_b is None:
        print("  实验 B 求解失败!")
        metrics_b = None
    else:
        sel_routes, yo, yi, ys = result_b
        metrics_b = extract_metrics(sel_routes, yo, yi, ys, routes_info)
        print(f"  B: 时间={metrics_b[0]}min, 利用率={metrics_b[1]*100:.2f}%, "
              f"架次={metrics_b[2]}, 燃油={metrics_b[3]:.1f}kg, 耗时={time_b:.1f}s")

    # ---------- 6. 对比汇总 ----------
    print("\n" + "=" * 70)
    print("消融实验结果")
    print("=" * 70)
    if metrics_a and metrics_b:
        dt = metrics_b[0] - metrics_a[0]
        du = (metrics_b[1] - metrics_a[1]) * 100
        dn = metrics_b[2] - metrics_a[2]
        df = metrics_b[3] - metrics_a[3]
        print(f"  总时间:  A={metrics_a[0]}min  B={metrics_b[0]}min  (Δ{dt:+d}min)")
        print(f"  利用率:  A={metrics_a[1]*100:.2f}%  B={metrics_b[1]*100:.2f}%  (Δ{du:+.2f}pp)")
        print(f"  架次:    A={metrics_a[2]}  B={metrics_b[2]}  (Δ{dn:+d})")
        print(f"  燃油:    A={metrics_a[3]:.1f}kg  B={metrics_b[3]:.1f}kg  (Δ{df:+.1f}kg)")
        print(f"  求解耗时: A={time_a:.1f}s  B={time_b:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
