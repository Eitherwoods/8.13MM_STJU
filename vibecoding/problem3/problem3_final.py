# -*- coding: utf-8 -*-
"""
文件: problem3_final.py
功能: 问题3最终版求解器 — 整合全部优化手段
项目: 2026策联杯 B题 - 海上油田人员直升机运载计划编排

设计思想（相对原 problem3_solution.py 的增强）:
  F1. 路径池智能精简: 用"固定关键时刻 + 需求时间窗聚类中心"替代 14 个均匀偏移，
      将候选路径从 ~5 万条压缩到可控规模，大幅降低 MIP 变量/约束数；
  F2. ALNS 三阶段接入:
       ① 热启动: MIP 前先跑 ALNS 快速构造可行解，注入 COPT MipStart；
       ② 抛光: MIP 后从 MIP 解出发继续 ALNS 迭代，在不劣化总时间的前提下
         改善二级指标（人员在途/利用率/燃油）；
       ③ 降级: MIP 无可行解时，直接采用 ALNS 可行解兜底，避免整日失败；
  F3. 动态列生成（C8 落地）: 第一轮 MIP 后分析薄弱设施（出海载客率低），
      为薄弱设施补充密集起飞时刻与多设施组合路径，合并后再求解第二轮，
      让路径池随对偶信息"生长"而非一次性铺满；
  F4. COPT 大模型参数调优: MIPFocus/Cuts/Heuristics/Presolve/Threads 钩子；
  F5. 逐日即时输出: 每天求解完成立即分配飞机编号并写 q3-routes-{date}.csv
      与 q3-assignments-{date}.csv，防止终端中断导致结果丢失。

运行（Powershell）:
  E:\Python\python.exe problem3_final.py                 # 完整模式
  E:\Python\python.exe problem3_final.py --fast          # 快速预览（单日缩短时限）
  E:\Python\python.exe problem3_final.py --no-cg         # 关闭列生成增强
  E:\Python\python.exe problem3_final.py --no-polish     # 关闭 ALNS 抛光
  E:\Python\python.exe problem3_final.py --day 2026-08-03  # 只跑指定日期
"""

import os
import sys
import time as time_module
from datetime import datetime, timedelta, date
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional

# ==================== 模块路径 ====================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VIBECODING_DIR = os.path.dirname(_THIS_DIR)
if _VIBECODING_DIR not in sys.path:
    sys.path.insert(0, _VIBECODING_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
# problem12 子包路径（跨问题导入 Q2RoutePreprocessor）
_PROBLEM12_DIR = os.path.join(_VIBECODING_DIR, 'problem12')
if _PROBLEM12_DIR not in sys.path:
    sys.path.insert(0, _PROBLEM12_DIR)

import coptpy as cp

from config import AIRCRAFT, AIRPORTS
from base_models import make_route, TSPSolver
from problem2_solution import Q2RoutePreprocessor
from problem3_solution import (
    Q3DataLoader,
    Q3DemandDayAllocator,
    Q3ScheduleHelper,
    Q3RouteGenerator,
    Q3DailySolver,
    Q3AircraftAssigner,
    Q3Phase2Solver,
    Q3OutputWriter,
    FLEET,
    PLANNING_DATES,
    OPERATION_START_HOUR,
    DEPARTURE_DEADLINE_HOUR,
    RETURN_DEADLINE_HOUR,
    TURNAROUND_MINUTES,
    DEPARTURE_OFFSETS,
    OUTPUT_DIR_Q3,
)
from alns_improver import AlnsImprover, AlnsSolution

# ==================== 最终版配置 ====================

# 固定关键时刻（分钟偏移，06:00 起）: 覆盖早高峰与均匀分布的兜底
FIXED_ESSENTIAL_OFFSETS = [0, 60, 120, 240, 360, 480, 600]

# 智能偏移上限（含固定 + 聚类中心）
SMART_MAX_OFFSETS = 10

# 多设施路径剪枝上限（pair/triple/quad/quint）
# 完整原始值: (300, 300, 600, 900) → 池约 5 万条
# 精简值:     (150, 150, 300, 450) → 池约 2.2 万条（压缩 56%）
PRUNE_CAPS = (150, 150, 300, 450)
PRUNE_CAPS_FAST = (80, 80, 160, 240)

# COPT 调参：针对 10 万级变量模型的经验值（COPT 8.0 参数名）
COPT_TUNING = {
    'RootCutLevel': 3,      # 根节点割平面增强（0-4）
    'TreeCutLevel': 3,      # 分支树割平面增强（0-4）
    'CutLevel': 3,          # 全局割平面水平
    'HeurLevel': 2,         # 启发式水平（0-4）
    'Presolve': 2,          # 2=加强预处理
    'Threads': 32,          # 并行线程（i9-14900HX 32 逻辑核）
}

# 每日时间预算（秒）
DAY_TIME_LIMIT_1 = 3600     # 第一轮 MIP（智能池）
DAY_TIME_LIMIT_2 = 1800     # 第二轮 MIP（列生成增强池）
ALNS_WARM_ITERS = 200       # MIP 前 ALNS 热启动迭代数
ALNS_POLISH_ITERS = 150     # MIP 后 ALNS 抛光迭代数

# 列生成参数
CG_WEAK_THRESHOLD = 0.5     # 薄弱设施载客率阈值
CG_MAX_NEW_ROUTES = 4000    # 补列上限（防止第二轮规模失控）


def compute_smart_offsets(day_demands, day, max_offsets=SMART_MAX_OFFSETS):
    """
    计算智能起飞偏移列表：固定关键时刻 + 需求时间窗聚类中心。

    参数:
        day_demands: 当日需求列表
        day: 日期
        max_offsets: 偏移数量上限

    返回:
        list[int]: 升序偏移分钟列表
    """
    # 复用 C7 聚类方法（仅依赖 day_demands / day，不依赖 self）
    cluster_offsets = Q3RouteGenerator._compute_cluster_offsets(
        object.__new__(Q3RouteGenerator), day_demands, day)
    merged = sorted(set(FIXED_ESSENTIAL_OFFSETS) | set(cluster_offsets))
    if len(merged) <= max_offsets:
        return merged
    # 超出上限：优先保留聚类中心（最贴合需求），再补固定关键时刻
    result = sorted(set(cluster_offsets))
    for off in FIXED_ESSENTIAL_OFFSETS:
        if len(result) >= max_offsets:
            break
        result.append(off)
    return sorted(result)[:max_offsets]


class TunedQ3DailySolver(Q3DailySolver):
    """
    COPT 参数调优版单日求解器（F4）。

    覆盖 _apply_solver_tuning 钩子，为每个 COPT 模型注入针对大模型的
    MIPFocus/Cuts/Heuristics/Presolve/Threads 参数。
    """

    def _apply_solver_tuning(self, model):
        """为 COPT 模型应用大模型调参（F4）"""
        for name, value in COPT_TUNING.items():
            try:
                model.setParam(getattr(cp.COPT.Param, name), value)
            except Exception as exc:
                print(f"  [调参] 参数 {name}={value} 设置失败: {exc}")

    def set_round_limits(self, time_limit_1, time_limit_2):
        """
        设置自适应轮次时限（覆盖硬编码 ADAPTIVE_ROUNDS）。

        参数:
            time_limit_1: 第一轮（粗搜）时限
            time_limit_2: 后续轮次（精搜/极精搜）时限
        """
        self.ADAPTIVE_ROUNDS = [
            (time_limit_1, 0.05, '粗搜'),
            (time_limit_2, 0.02, '精搜'),
            (time_limit_2, 0.01, '极精搜'),
        ]
        self.TIME_LIMITS = [time_limit_1, time_limit_2]
        self.TWO_STAGE_TIME_LIMITS = [time_limit_1, time_limit_2]
        self.RETRY_TIME_LIMIT = time_limit_2


def alns_solution_to_result(alns_sol, routes_info, day, distances):
    """
    把 ALNS 解（AlnsSolution）转换为与 Q3DailySolver.solve 一致的返回格式。

    参数:
        alns_sol: AlnsSolution 对象
        routes_info: 路径池（索引对齐）
        day: 日期
        distances: 距离矩阵

    返回:
        (sel_routes, yo_vals, yi_vals, ys_vals) 或 None（解不可用）
    """
    if alns_sol is None:
        return None
    sel_indices = sorted(alns_sol.selected)
    if not sel_indices:
        return None
    sel_routes = [routes_info[i] for i in sel_indices]
    return sel_routes, alns_sol.yo, alns_sol.yi, alns_sol.ys


def build_initial_solution_from_mip(sel_routes, yo_vals, yi_vals, ys_vals,
                                    routes_info):
    """
    由 MIP 解构造 AlnsSolution（供 ALNS 抛光使用）。

    参数:
        sel_routes: MIP 选中的 routes_info 子集
        yo_vals/yi_vals/ys_vals: MIP 人员分配
        routes_info: 完整路径池

    返回:
        AlnsSolution
    """
    idx_map = {id(ri): idx for idx, ri in enumerate(routes_info)}
    selected = {idx_map[id(ri)] for ri in sel_routes}
    return AlnsSolution(selected, dict(yo_vals), dict(yi_vals), dict(ys_vals))


def solve_day_final(routes_info, day_demands, day, distances, solver,
                    use_cg=True, use_polish=True, seed=20260814):
    """
    单日完整求解流程（F1+F2+F3）:
        ALNS 热启动 → 第一轮 MIP → 列生成补列 → 第二轮 MIP → ALNS 抛光

    参数:
        routes_info: 第一轮路径池（智能偏移已精简）
        day_demands: 当日需求
        day: 日期
        distances: 距离矩阵
        solver: TunedQ3DailySolver 实例
        use_cg: 是否启用列生成增强
        use_polish: 是否启用 ALNS 抛光
        seed: ALNS 随机种子

    返回:
        (sel_routes, yo_vals, yi_vals, ys_vals, t_star, alns_used, final_routes)
        或 None
        final_routes: 实际使用的路径池（可能被列生成扩展，用于索引映射）
    """
    print(f"\n----- 单日完整求解: {day} ({len(day_demands)}人, 池={len(routes_info)}) -----")

    # ---------- F2① ALNS 热启动 ----------
    alns = AlnsImprover(routes_info, day_demands, distances, day,
                        random_seed=seed)
    print("\n[ALNS 热启动] 快速构造可行解...")
    t0 = time_module.time()
    alns_sol, alns_obj = alns.run(iterations=ALNS_WARM_ITERS,
                                  q_min=2, q_max=8, t0=5000.0, cooling=0.99,
                                  segment=30)
    print(f"  [ALNS 热启动] 耗时 {time_module.time()-t0:.1f}s, "
          f"目标={alns_obj:.2f}")
    warm_start = alns.to_mip_start(alns_sol)

    # ---------- 第一轮 MIP ----------
    print(f"\n[第一轮 MIP] TL={DAY_TIME_LIMIT_1}s ...")
    solver.set_round_limits(DAY_TIME_LIMIT_1, DAY_TIME_LIMIT_2)
    result1 = solver.solve(routes_info, day_demands, day, distances,
                           warm_start=warm_start)

    current_routes = routes_info
    current_result = result1

    # ---------- F3 列生成增强 ----------
    if use_cg:
        # 复用 C8 的薄弱设施分析与补列逻辑
        from column_generation import (
            analyze_weak_facilities, generate_new_routes, build_existing_keys)
        # 第一轮失败时：基于 ALNS 解分析薄弱设施（不放弃，先补列再重试）
        if result1 is None:
            print("  [列生成] 第一轮 MIP 无解，基于 ALNS 解分析薄弱设施")
            alns_fallback = alns_solution_to_result(
                alns_sol, routes_info, day, distances)
            if alns_fallback is None:
                return None
            f_sel, f_yo, f_yi, f_ys = alns_fallback
            weak_facs, load_info = analyze_weak_facilities(
                f_sel, f_yo, routes_info, day_demands)
            weak_facs = [f for f in weak_facs
                         if load_info[f]['load'] < CG_WEAK_THRESHOLD][:5]
            if weak_facs:
                print(f"  [列生成] 薄弱设施: {weak_facs}")
                rid_counter = [max((ri['route'].route_id for ri in routes_info),
                                   default=0) + 1]
                existing_keys = build_existing_keys(routes_info, day)
                new_routes, _ = generate_new_routes(
                    weak_facs, day_demands, day, distances,
                    existing_keys, rid_counter)
                if len(new_routes) > CG_MAX_NEW_ROUTES:
                    new_routes = new_routes[:CG_MAX_NEW_ROUTES]
                if new_routes:
                    routes2 = routes_info + new_routes
                    print(f"  [列生成] 补充 {len(new_routes)} 条路径 "
                          f"(池 {len(routes_info)}→{len(routes2)})")
                    print(f"  [第二轮 MIP] TL={DAY_TIME_LIMIT_2}s ...")
                    solver.set_round_limits(DAY_TIME_LIMIT_2, DAY_TIME_LIMIT_2)
                    ws2 = alns.to_mip_start(alns_sol)
                    result2 = solver.solve(routes2, day_demands, day, distances,
                                           warm_start=ws2)
                    if result2 is not None:
                        current_routes = routes2
                        current_result = result2
                        print(f"  [列生成] 补列后重试成功!")
            if current_result is None:
                print("  [降级] 补列后仍无解，采用 ALNS 解兜底")
                fallback = alns_solution_to_result(
                    alns_sol, routes_info, day, distances)
                if fallback is None:
                    return None
                sel_routes, yo_vals, yi_vals, ys_vals = fallback
                t_star = sum(ri['route'].time for ri in sel_routes)
                return sel_routes, yo_vals, yi_vals, ys_vals, t_star, True, routes_info
            sel, yo, yi, ys, t_star = current_result
            return sel, yo, yi, ys, t_star, True, current_routes

        # result1 非 None 分支：第一轮成功后的薄弱设施补列
        sel1, yo1, yi1, ys1, t_star1 = result1
        weak_facs, load_info = analyze_weak_facilities(
            sel1, yo1, current_routes, day_demands)
        weak_facs = [f for f in weak_facs
                     if load_info[f]['load'] < CG_WEAK_THRESHOLD][:5]
        if weak_facs:
            print(f"\n[列生成] 薄弱设施: {weak_facs}")
            rid_counter = [max((ri['route'].route_id for ri in current_routes),
                               default=0) + 1]
            existing_keys = build_existing_keys(current_routes, day)
            new_routes, n_base = generate_new_routes(
                weak_facs, day_demands, day, distances,
                existing_keys, rid_counter)
            # 控制补列规模
            if len(new_routes) > CG_MAX_NEW_ROUTES:
                new_routes = new_routes[:CG_MAX_NEW_ROUTES]
            if new_routes:
                routes2 = current_routes + new_routes
                print(f"  [列生成] 补充 {len(new_routes)} 条路径 "
                      f"(池 {len(current_routes)}→{len(routes2)})")
                # 用第一轮解做热启动（索引前缀不变）
                idx_map1 = {id(ri): i for i, ri in enumerate(current_routes)}
                ws2 = {
                    'selected': {idx_map1[id(ri)] for ri in sel1},
                    'yo': yo1, 'yi': yi1, 'ys': ys1,
                }
                print(f"  [第二轮 MIP] TL={DAY_TIME_LIMIT_2}s ...")
                solver.set_round_limits(DAY_TIME_LIMIT_2, DAY_TIME_LIMIT_2)
                result2 = solver.solve(routes2, day_demands, day, distances,
                                       warm_start=ws2)
                if result2 is not None:
                    sel2, yo2, yi2, ys2, t_star2 = result2
                    if t_star2 <= t_star1 * 1.0001:
                        current_routes = routes2
                        current_result = result2
                        print(f"  [列生成] 采用增强池解: T={t_star2} (原 {t_star1})")
                    else:
                        print(f"  [列生成] 增强池解 T={t_star2} 未改善，保留第一轮解")
                else:
                    print("  [列生成] 第二轮 MIP 无可行解，保留第一轮解")
        else:
            print("\n[列生成] 无薄弱设施，跳过补列")

    sel, yo, yi, ys, t_star = current_result

    # ---------- F2② ALNS 抛光 ----------
    if use_polish:
        print("\n[ALNS 抛光] 从 MIP 解出发改进二级指标...")
        mip_time = sum(ri['route'].time for ri in sel)
        init_sol = build_initial_solution_from_mip(sel, yo, yi, ys, current_routes)
        # 若列生成扩展了池，需基于 current_routes 重建 ALNS（索引对齐）
        if current_routes is not routes_info:
            alns = AlnsImprover(current_routes, day_demands, distances, day,
                                random_seed=seed)
        t0 = time_module.time()
        polished, polished_obj = alns.run(iterations=ALNS_POLISH_ITERS,
                                          q_min=2, q_max=8,
                                          t0=2000.0, cooling=0.99,
                                          segment=30,
                                          initial_solution=init_sol)
        pol_time = sum(current_routes[i]['route'].time for i in polished.selected)
        cov_v = alns._check_coverage(polished)
        fleet_v = alns._check_fleet(polished)
        print(f"  [ALNS 抛光] 耗时 {time_module.time()-t0:.1f}s, "
              f"T={pol_time} (MIP={mip_time}), 覆盖违规={cov_v}, 机队违规={fleet_v}")
        # 采用准则：不劣化总时间且无覆盖/机队违规
        if (pol_time <= mip_time * 1.0001 and cov_v == 0 and fleet_v == 0):
            sel, yo, yi, ys = alns_solution_to_result(
                polished, current_routes, day, distances)
            t_star = min(pol_time, t_star)
            print(f"  [ALNS 抛光] 采用抛光解: T={t_star}")
        else:
            print("  [ALNS 抛光] 抛光解不满足条件，保留 MIP 解")

    return sel, yo, yi, ys, t_star, False, current_routes


def main():
    """
    最终版主函数：加载数据 → 需求-日期分配 → 逐日完整求解 →
    失败日重分配 → 阶段二临时任务 → 最终输出与指标汇总。
    """
    print("=" * 60)
    print("问题3 最终版: 带时间要求的多日排班（F1-F5 全部优化）")
    print("=" * 60)

    # 解析命令行参数
    fast_mode = '--fast' in sys.argv
    use_cg = '--no-cg' not in sys.argv
    use_polish = '--no-polish' not in sys.argv
    only_day = None
    for i, a in enumerate(sys.argv):
        if a == '--day' and i + 1 < len(sys.argv):
            only_day = datetime.strptime(sys.argv[i + 1], '%Y-%m-%d').date()

    global DAY_TIME_LIMIT_1, DAY_TIME_LIMIT_2, ALNS_WARM_ITERS, ALNS_POLISH_ITERS
    prune_caps = PRUNE_CAPS
    if fast_mode:
        DAY_TIME_LIMIT_1 = 300
        DAY_TIME_LIMIT_2 = 150
        ALNS_WARM_ITERS = 80
        ALNS_POLISH_ITERS = 60
        prune_caps = PRUNE_CAPS_FAST
        print("[模式] 快速预览（时限缩短，仅验证流程）")
    else:
        print(f"[模式] 完整求解 (MIP1={DAY_TIME_LIMIT_1}s, MIP2={DAY_TIME_LIMIT_2}s, "
              f"CG={'开' if use_cg else '关'}, 抛光={'开' if use_polish else '关'})")

    # 1. 加载数据
    print("\n[1] 加载数据...")
    loader = Q3DataLoader()
    distances = loader.load_distances()
    all_demands = loader.load_demands_q3()
    print(f"  总需求: {len(all_demands)}")

    non_temp = [d for d in all_demands if d['task_type'] != 'temporary']
    temp_demands = [d for d in all_demands if d['task_type'] == 'temporary']
    print(f"  非临时(阶段一): {len(non_temp)}, 临时(阶段二): {len(temp_demands)}")

    # 2. 需求-日期分配
    print("\n[2] 需求-日期分配...")
    allocation, daily_load = Q3DemandDayAllocator.allocate(non_temp, PLANNING_DATES)
    for d in non_temp:
        d['_assigned_day'] = allocation[d['person_id']]
    for d in PLANNING_DATES:
        print(f"  {d}: {daily_load[d]}人")

    # 3. 逐日完整求解
    print("\n[3] 逐日完整求解...")
    all_selected_routes = []
    total_time = 0
    t_star_sum = 0
    total_fuel = 0.0
    total_pass_km = 0.0
    total_seat_km = 0.0
    total_transit = 0
    solved_days = set()
    failed_day_demands = defaultdict(list)
    alns_fallback_days = []

    route_id_counter = [0]
    solver = TunedQ3DailySolver()
    solver.USE_ADAPTIVE = True
    solver.USE_TWO_STAGE = False
    solver.USE_WEIGHTED = True

    dates_to_solve = [only_day] if only_day else PLANNING_DATES

    for day in dates_to_solve:
        day_demands = [d for d in non_temp if allocation[d['person_id']] == day]
        if not day_demands:
            print(f"\n--- {day}: 无分配需求，跳过 ---")
            continue
        print(f"\n--- {day} ({len(day_demands)}人) ---")

        # F1: 智能偏移生成路径池（精简）
        smart_offsets = compute_smart_offsets(day_demands, day)
        print(f"  智能起飞偏移({len(smart_offsets)}个): {smart_offsets}")
        generator = Q3RouteGenerator(distances, offsets=smart_offsets,
                                     pair_cap=prune_caps[0],
                                     triple_cap=prune_caps[1],
                                     quad_cap=prune_caps[2],
                                     quint_cap=prune_caps[3])
        routes_info = generator.generate_daily_routes(day_demands, day,
                                                       route_id_counter[0])
        route_id_counter[0] = (max((r['route'].route_id for r in routes_info),
                                   default=route_id_counter[0]))
        print(f"  候选路径(智能池): {len(routes_info)}")

        if not routes_info:
            # 兜底：智能偏移可能因偏移过少导致无路径，回退完整偏移
            print("  智能池为空，回退完整 DEPARTURE_OFFSETS ...")
            generator_full = Q3RouteGenerator(distances)
            routes_info = generator_full.generate_daily_routes(
                day_demands, day, route_id_counter[0])
            route_id_counter[0] = (max((r['route'].route_id for r in routes_info),
                                       default=route_id_counter[0]))
            print(f"  候选路径(完整池): {len(routes_info)}")

        if not routes_info:
            print(f"  警告: {day} 无候选路径!")
            failed_day_demands[day] = day_demands
            continue

        # 单日完整求解
        result = solve_day_final(routes_info, day_demands, day, distances,
                                 solver, use_cg=use_cg, use_polish=use_polish)
        if result is None:
            print(f"  警告: {day} 全部方法无可行解!")
            failed_day_demands[day] = day_demands
            continue

        sel_routes, yo_vals, yi_vals, ys_vals, t_star_day, alns_used, final_routes = result
        if alns_used:
            alns_fallback_days.append(day)

        solved_days.add(day)
        t_star_sum += t_star_day
        for idx, ri in enumerate(sel_routes):
            ri['route_id'] = ri['route'].route_id

        # 索引映射（使用实际求解池 final_routes，可能被列生成扩展）
        idx_map = {id(ri): idx for idx, ri in enumerate(final_routes)}
        for ri in sel_routes:
            orig_idx = idx_map.get(id(ri))
            if orig_idx is None:
                # 路径可能来自列生成增强池（routes_info 之外）
                continue
            ri['_orig_idx'] = orig_idx

        # 重新构建 yo/yi/ys 键（对齐当前池索引）
        for ri in sel_routes:
            oi = idx_map.get(id(ri))
            if oi is None:
                continue
            ri['yo'] = {(ri_idx, f): v for (ri_idx, f), v in yo_vals.items()
                        if ri_idx == oi}
            ri['yi'] = {(ri_idx, f): v for (ri_idx, f), v in yi_vals.items()
                        if ri_idx == oi}
            ri['ys'] = {(ri_idx, f1, f2): v for (ri_idx, f1, f2), v in ys_vals.items()
                        if ri_idx == oi}

        # 统计
        for ri in sel_routes:
            r = ri['route']
            pre = ri.get('precomputed', {})
            total_time += r.time
            total_fuel += r.fuel
            total_seat_km += r.capacity * r.total_distance
            for (_, f), v in ri.get('yo', {}).items():
                total_pass_km += v * pre.get('dist_to', {}).get(f, 0)
                total_transit += v * pre.get('time_to', {}).get(f, 0)
            for (_, f), v in ri.get('yi', {}).items():
                total_pass_km += v * pre.get('dist_from', {}).get(f, 0)
                total_transit += v * pre.get('time_from', {}).get(f, 0)
            for (_, f1, f2), v in ri.get('ys', {}).items():
                si = pre.get('shuttle_info', {}).get((f1, f2), {})
                total_pass_km += v * si.get('dist', 0)
                total_transit += v * si.get('time', 0)
            all_selected_routes.append({'route_info': ri, 'day': day,
                                        'yo': ri['yo'], 'yi': ri['yi'],
                                        'ys': ri['ys']})

        n_flights = len(sel_routes)
        n_people = sum(sum(ri['yo'].values()) + sum(ri['yi'].values()) +
                       sum(ri['ys'].values()) for ri in sel_routes)
        print(f"  架次: {n_flights}, 人数: {n_people}, "
              f"时间: {sum(ri['route'].time for ri in sel_routes)}min")

        # F5: 逐日即时输出
        day_routes_for_assign = [
            {'route_info': ri, 'day': day, 'yo': ri['yo'],
             'yi': ri['yi'], 'ys': ri['ys']}
            for ri in sel_routes
        ]
        day_flights = Q3AircraftAssigner.assign(day_routes_for_assign)
        Q3OutputWriter.write_daily_output(day, day_flights, day_demands,
                                          distances, OUTPUT_DIR_Q3)

    # 3.1 失败日需求重分配
    if failed_day_demands:
        print(f"\n[3.1] 失败日需求重分配: {len(failed_day_demands)} 个日期")
        day_load = defaultdict(int)
        for sr in all_selected_routes:
            day = sr['day']
            n = (sum(sr['yo'].values()) + sum(sr['yi'].values()) +
                 sum(sr['ys'].values()))
            day_load[day] += n
        for failed_day, demands in failed_day_demands.items():
            reallocated = 0
            for d in demands:
                feasible = [pd for pd in solved_days
                            if pd in Q3DemandDayAllocator.compute_feasible_days(
                                d, PLANNING_DATES)]
                if not feasible:
                    continue
                best_day = min(feasible, key=lambda pd: day_load[pd])
                d['_assigned_day'] = best_day
                day_load[best_day] += 1
                reallocated += 1
            print(f"  {failed_day}: {len(demands)}人 -> 重分配 {reallocated}人")

    T_star = t_star_sum
    print(f"\n阶段一总飞机使用时间 T* = {T_star} 分钟 ({T_star/60:.1f}h)")
    print(f"总架次: {len(all_selected_routes)}, 总燃油: {total_fuel:.1f}kg")
    if alns_fallback_days:
        print(f"ALNS 兜底天数: {[str(d) for d in alns_fallback_days]}")

    # 4. 飞机分配
    print("\n[4] 飞机编号分配...")
    flights = Q3AircraftAssigner.assign(all_selected_routes)
    print(f"  分配飞机: {len(set(f['aircraft_id'] for f in flights))}架")

    # 5. 阶段二临时任务
    print("\n[5] 阶段二: 临时任务背包化...")
    phase2 = Q3Phase2Solver()
    temp_satisfied, temp_unsatisfied = phase2.solve(
        flights, temp_demands, T_star, distances, PLANNING_DATES)

    new_flights = [f for f in flights if f.get('aircraft_id') is None]
    if new_flights:
        for f in new_flights:
            r = f['route_info']['route']
            for seq in range(FLEET[r.airport][r.aircraft_type]):
                aid = f"{r.airport}-{r.aircraft_type}-H{seq+1:02d}"
                used = [fl for fl in flights if fl.get('aircraft_id') == aid]
                conflict = False
                for u in used:
                    if not (f['route_info']['return_time'] + timedelta(minutes=TURNAROUND_MINUTES) <= u['route_info']['departure_time']
                            or u['route_info']['return_time'] + timedelta(minutes=TURNAROUND_MINUTES) <= f['route_info']['departure_time']):
                        conflict = True
                        break
                if not conflict:
                    f['aircraft_id'] = aid
                    max_fn = max([fl['flight_no'] for fl in flights
                                  if fl.get('aircraft_id') == aid], default=0)
                    f['flight_no'] = max_fn + 1
                    break

    # 6. 输出
    print("\n[6] 输出结果...")
    Q3OutputWriter.write_routes(flights, OUTPUT_DIR_Q3)
    Q3OutputWriter.write_assignments(flights, all_demands,
                                     temp_satisfied, temp_unsatisfied,
                                     OUTPUT_DIR_Q3)

    # 7. 指标汇总
    print("\n" + "=" * 60)
    print("问题3最终版结果汇总")
    print("=" * 60)
    actual_time = sum(f['route_info']['route'].time for f in flights)
    actual_fuel = sum(f['route_info']['route'].fuel for f in flights)
    actual_seat_km = sum(f['route_info']['route'].capacity *
                         f['route_info']['route'].total_distance for f in flights)
    temp_pass_km = 0.0
    temp_transit = 0
    for d in temp_satisfied:
        fi = d.get('_flight_idx')
        if fi is not None and fi < len(flights):
            r = flights[fi]['route_info']['route']
            o = d['origin_id']
            de = d['destination_id']
            o_actual = r.airport if o == 'LAND' else o
            de_actual = r.airport if de == 'LAND' else de
            if o_actual in r.path and de_actual in r.path:
                p1 = r.path.index(o_actual)
                p2 = r.path.index(de_actual)
                if p2 > p1:
                    speed = AIRCRAFT[r.aircraft_type]['speed']
                    for s in range(p1, p2):
                        seg_dist = distances[r.path[s]][r.path[s + 1]]
                        temp_pass_km += seg_dist
                        temp_transit += seg_dist / speed * 60
    actual_pass_km = total_pass_km + temp_pass_km
    phase1_n_flights = len(all_selected_routes)
    phase2_new_flights = len(flights) - phase1_n_flights
    phase1_seat_util = total_pass_km / total_seat_km if total_seat_km > 0 else 0
    seat_util = actual_pass_km / actual_seat_km if actual_seat_km > 0 else 0
    total_transit_all = total_transit + temp_transit

    metrics = {}
    metrics['--- 阶段一：固定排班（不含临时任务） ---'] = ''
    metrics['  总飞机使用时间 T*'] = f"{T_star} 分钟 ({T_star/60:.2f}h)"
    metrics['  总架次数'] = str(phase1_n_flights)
    metrics['  总燃油消耗量'] = f"{total_fuel:.1f} kg"
    metrics['  座位利用率'] = f"{phase1_seat_util*100:.2f}%"
    metrics['  人员在途时间'] = f"{total_transit} 分钟 ({total_transit/60:.2f}h)"
    metrics['--- 阶段二：临时任务背包化 ---'] = ''
    metrics['  新增飞机使用时间'] = f"{actual_time - T_star} 分钟 ({(actual_time - T_star)/60:.2f}h)"
    metrics['  新增架次数'] = str(phase2_new_flights)
    metrics['  新增燃油消耗量'] = f"{actual_fuel - total_fuel:.1f} kg"
    metrics['  临时任务在途时间'] = f"{temp_transit} 分钟 ({temp_transit/60:.2f}h)"
    metrics['  临时任务客公里'] = f"{temp_pass_km:.1f}"
    metrics['  临时任务满足'] = (f"{len(temp_satisfied)}/{len(temp_demands)} "
                             f"({len(temp_satisfied)/max(len(temp_demands),1)*100:.2f}%)")
    metrics['--- 合计 ---'] = ''
    metrics['  总飞机使用时间'] = f"{actual_time} 分钟 ({actual_time/60:.2f}h)"
    metrics['  总架次数'] = str(len(flights))
    metrics['  总燃油消耗量'] = f"{actual_fuel:.1f} kg"
    metrics['  座位利用率'] = f"{seat_util*100:.2f}%"
    metrics['  人员在途时间'] = f"{total_transit_all} 分钟 ({total_transit_all/60:.2f}h)"

    for k, v in metrics.items():
        if v:
            print(f"  {k}: {v}")
        else:
            print(f"\n{k}")

    metrics_path = os.path.join(OUTPUT_DIR_Q3, 'q3-metrics.txt')
    os.makedirs(OUTPUT_DIR_Q3, exist_ok=True)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        for k, v in metrics.items():
            if v:
                f.write(f"{k}: {v}\n")
    print(f"\n指标写入: {metrics_path}")


if __name__ == '__main__':
    main()
