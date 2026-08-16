# -*- coding: utf-8 -*-
"""
文件: pareto_sweep.py
功能: C10 加权灵敏度分析 Pareto 前沿
项目: 2026策联杯 B题 - 海上油田人员直升机运载计划编排

说明:
    对单日问题（以 2026-08-03 为例），通过 monkey-patching 修改 Q3DailySolver 的
    类属性 W_TIME / W_UTIL，在多组权重组合下求解同一个 MIP，记录每组方案的
    (总时间, 座位利用率, 架次, 燃油) 四项指标，并标注 Pareto 最优解。

    为保证公平比较，候选路径集 routes_info 只生成一次（路径生成不依赖权重），
    在同一可行域上对比不同加权目标的效果。

运行方式:
    python pareto_sweep.py
"""

import sys
import os
import time as time_module
from datetime import date

# ==================== 模块路径 ====================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from problem3_solution import (
    Q3DataLoader, Q3DemandDayAllocator, Q3RouteGenerator,
    Q3ScheduleHelper, Q3DailySolver, FLEET, PLANNING_DATES,
)
from config import DATA_DIR


# ==================== 全局配置 ====================
# 目标分析日期（单日问题）
TARGET_DAY = date(2026, 8, 3)

# 权重组合：(W_TIME, W_UTIL, 方案标签)
WEIGHT_CONFIGS = [
    (1.0, 0.05, '时间优先'),
    (1.0, 0.10, '偏时间'),
    (1.0, 0.15, '均衡(当前)'),
    (1.0, 0.25, '偏利用率'),
    (1.0, 0.40, '利用率优先'),
    (0.5, 0.30, '高利用率'),
    (2.0, 0.05, '极端时间优先'),
]

# 浮点比较容差
EPS = 1e-6

# 结果输出目录（最终交付物保存到工作区）
OUTPUT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==================== 指标提取 ====================
def extract_metrics(sel_routes, yo_vals, yi_vals, ys_vals, routes_info):
    """
    从 MIP 求解结果中提取四项指标。

    参数:
        sel_routes: 选中路径列表（routes_info 的子集引用）
        yo_vals: 出海客流分配 {(ri_idx, facility): count}
        yi_vals: 海返客流分配 {(ri_idx, facility): count}
        ys_vals: 穿梭客流分配 {(ri_idx, f1, f2): count}
        routes_info: 完整候选路径列表（用于索引映射）

    返回:
        (total_time, seat_util, n_flights, total_fuel)
        - total_time: 总飞机使用时间（分钟）
        - seat_util: 座位利用率（0~1）
        - n_flights: 架次数
        - total_fuel: 总燃油消耗（kg）
    """
    # 建立 id -> 原始索引 的映射（sel_routes 中的元素是 routes_info 的引用）
    idx_map = {id(ri): idx for idx, ri in enumerate(routes_info)}
    # 选中路径的原始索引集合
    sel_orig_indices = set(idx_map[id(ri)] for ri in sel_routes)
    # 原始索引 -> precomputed 的映射
    pre_by_idx = {}
    for ri in sel_routes:
        oi = idx_map[id(ri)]
        pre_by_idx[oi] = ri.get('precomputed', {})

    # 总时间、总燃油、总座公里（直接从路径对象累加）
    total_time = sum(ri['route'].time for ri in sel_routes)
    total_fuel = sum(ri['route'].fuel for ri in sel_routes)
    total_seat_km = sum(ri['route'].capacity * ri['route'].total_distance
                        for ri in sel_routes)

    # 总客公里（从客流分配变量 + precomputed 距离累加）
    total_pass_km = 0.0
    for (ri_idx, f), v in yo_vals.items():
        if ri_idx in pre_by_idx:
            total_pass_km += v * pre_by_idx[ri_idx].get('dist_to', {}).get(f, 0.0)
    for (ri_idx, f), v in yi_vals.items():
        if ri_idx in pre_by_idx:
            total_pass_km += v * pre_by_idx[ri_idx].get('dist_from', {}).get(f, 0.0)
    for (ri_idx, f1, f2), v in ys_vals.items():
        if ri_idx in pre_by_idx:
            si = pre_by_idx[ri_idx].get('shuttle_info', {}).get((f1, f2), {})
            total_pass_km += v * si.get('dist', 0.0)

    # 座位利用率 = 客公里 / 座公里
    seat_util = total_pass_km / total_seat_km if total_seat_km > 0 else 0.0
    n_flights = len(sel_routes)

    return total_time, seat_util, n_flights, total_fuel


# ==================== Pareto 支配分析 ====================
def is_dominated(i, results):
    """
    判断解 i 是否被其他解支配。

    支配定义（最小化时间/架次/燃油，最大化利用率）:
        解 j 支配解 i  当且仅当
        j.time <= i.time  且  j.fuel <= i.fuel  且  j.flights <= i.flights
        且  j.util >= i.util
        且至少一个不等式严格成立。

    参数:
        i: 待判断的解索引
        results: 所有解的指标列表

    返回:
        True 表示被支配（非 Pareto 最优），False 表示 Pareto 最优
    """
    ri = results[i]
    for j, rj in enumerate(results):
        if j == i:
            continue
        # 跳过求解失败的解
        if rj.get('failed'):
            continue
        # j 在所有最小化目标上不劣于 i，在最大化目标上不劣于 i
        cond = (rj['time'] <= ri['time'] + EPS
                and rj['fuel'] <= ri['fuel'] + EPS
                and rj['flights'] <= ri['flights'] + EPS
                and rj['util'] >= ri['util'] - EPS)
        if not cond:
            continue
        # 至少一个目标严格更优
        strict = (rj['time'] < ri['time'] - EPS
                  or rj['fuel'] < ri['fuel'] - EPS
                  or rj['flights'] < ri['flights'] - EPS
                  or rj['util'] > ri['util'] + EPS)
        if strict:
            return True
    return False


# ==================== 表格输出 ====================
def print_results_table(results, pareto_flags):
    """
    打印结果表格并标注 Pareto 最优解。

    参数:
        results: 所有解的指标列表
        pareto_flags: Pareto 最优标记列表（True/False）
    """
    header = f"| {'方案':<14s} | {'W_TIME':>6s} | {'W_UTIL':>6s} | {'总时间(min)':>11s} | {'利用率(%)':>9s} | {'架次':>4s} | {'燃油(kg)':>10s} | {'Pareto':>6s} |"
    sep    = f"|{'-'*16}|{'-'*8}|{'-'*8}|{'-'*13}|{'-'*11}|{'-'*6}|{'-'*12}|{'-'*8}|"
    print(header)
    print(sep)
    for r, is_pareto in zip(results, pareto_flags):
        if r.get('failed'):
            tag = r['label']
            row = f"| {tag:<14s} | {r['w_time']:>6.2f} | {r['w_util']:>6.2f} | {'FAILED':>11s} | {'-':>9s} | {'-':>4s} | {'-':>10s} | {'-':>6s} |"
        else:
            tag = r['label']
            p_tag = '*' if is_pareto else ''
            row = (f"| {tag:<14s} | {r['w_time']:>6.2f} | {r['w_util']:>6.2f} | "
                   f"{r['time']:>11d} | {r['util']*100:>9.2f} | {r['flights']:>4d} | "
                   f"{r['fuel']:>10.1f} | {p_tag:>6s} |")
        print(row)
    print(sep)
    print("注: * 标记为 Pareto 最优解（不被任何其他方案支配）")


def save_results_csv(results, pareto_flags, csv_path):
    """
    将结果保存为 CSV 文件。

    参数:
        results: 所有解的指标列表
        pareto_flags: Pareto 最优标记列表
        csv_path: 输出 CSV 路径
    """
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('方案,W_TIME,W_UTIL,总时间(min),利用率(%),架次,燃油(kg),Pareto最优\n')
        for r, is_pareto in zip(results, pareto_flags):
            if r.get('failed'):
                f.write(f"{r['label']},{r['w_time']},{r['w_util']},FAILED,,,,\n")
            else:
                p_tag = '是' if is_pareto else '否'
                f.write(f"{r['label']},{r['w_time']},{r['w_util']},"
                        f"{r['time']},{r['util']*100:.2f},{r['flights']},"
                        f"{r['fuel']:.1f},{p_tag}\n")


# ==================== 主函数 ====================
def main():
    """主函数：加载数据、生成路径、权重扫描、Pareto 分析、输出结果。"""
    print("=" * 70)
    print("C10 加权灵敏度分析 Pareto 前沿")
    print(f"目标日期: {TARGET_DAY}")
    print("=" * 70)

    # ---------- 1. 加载数据 ----------
    print("\n[1] 加载数据...")
    loader = Q3DataLoader()
    distances = loader.load_distances()
    all_demands = loader.load_demands_q3()
    print(f"  数据目录: {DATA_DIR}")
    print(f"  总需求: {len(all_demands)}")

    # 分离临时 / 非临时需求（阶段一仅处理非临时需求）
    non_temp = [d for d in all_demands if d['task_type'] != 'temporary']
    print(f"  非临时需求(阶段一): {len(non_temp)}")

    # ---------- 2. 需求-日期分配 ----------
    print("\n[2] 需求-日期分配...")
    allocation, daily_load = Q3DemandDayAllocator.allocate(non_temp, PLANNING_DATES)
    for d in non_temp:
        d['_assigned_day'] = allocation[d['person_id']]

    # 提取目标日需求
    day_demands = [d for d in non_temp if allocation[d['person_id']] == TARGET_DAY]
    print(f"  {TARGET_DAY} 分配需求: {len(day_demands)} 人")
    if not day_demands:
        print("  错误: 目标日无需求，退出。")
        return

    # ---------- 3. 生成候选路径（仅一次，保证公平比较） ----------
    print("\n[3] 生成候选路径...")
    t0 = time_module.time()
    generator = Q3RouteGenerator(distances)
    routes_info = generator.generate_daily_routes(day_demands, TARGET_DAY, route_id_start=1)
    t1 = time_module.time()
    print(f"  候选路径数: {len(routes_info)}  (生成耗时 {t1-t0:.1f}s)")
    if not routes_info:
        print("  错误: 无候选路径，退出。")
        return

    # ---------- 4. 加速设置 ----------
    print("\n[4] 加速设置...")
    # Dinkelbach 迭代次数设为 1（单次求解，不做利用率收敛迭代）
    Q3DailySolver.DINKELBACH_MAX_ITERS = 1
    # 单次求解时间限制 7200 秒（2小时）
    Q3DailySolver.TIME_LIMITS = [7200]
    # 失败重试时间限制 7200 秒（2小时）
    Q3DailySolver.RETRY_TIME_LIMIT = 7200
    # 确保使用加权单目标模式
    Q3DailySolver.USE_WEIGHTED = True
    # Pareto 扫描为权重灵敏度对比实验，关闭两层模式（保持加权单目标口径）
    Q3DailySolver.USE_TWO_STAGE = False
    print(f"  DINKELBACH_MAX_ITERS = {Q3DailySolver.DINKELBACH_MAX_ITERS}")
    print(f"  TIME_LIMITS = {Q3DailySolver.TIME_LIMITS}")
    print(f"  RETRY_TIME_LIMIT = {Q3DailySolver.RETRY_TIME_LIMIT}")
    print(f"  USE_WEIGHTED = {Q3DailySolver.USE_WEIGHTED}")

    # ---------- 5. 权重扫描 ----------
    print("\n[5] 权重扫描求解...")
    results = []
    for w_time, w_util, label in WEIGHT_CONFIGS:
        print(f"\n{'='*50}")
        print(f"  方案: {label}  (W_TIME={w_time}, W_UTIL={w_util})")
        print(f"{'='*50}")

        # monkey-patching: 修改类属性以设置权重
        Q3DailySolver.W_TIME = w_time
        Q3DailySolver.W_UTIL = w_util

        solver = Q3DailySolver()
        t0 = time_module.time()
        result = solver.solve(routes_info, day_demands, TARGET_DAY, distances)
        t1 = time_module.time()

        if result is None:
            print(f"  求解失败! 耗时 {t1-t0:.1f}s")
            results.append({
                'label': label, 'w_time': w_time, 'w_util': w_util,
                'failed': True,
            })
            continue

        sel_routes, yo_vals, yi_vals, ys_vals, _ = result
        total_time, seat_util, n_flights, total_fuel = extract_metrics(
            sel_routes, yo_vals, yi_vals, ys_vals, routes_info)

        print(f"  结果: 时间={total_time}min, 利用率={seat_util*100:.2f}%, "
              f"架次={n_flights}, 燃油={total_fuel:.1f}kg  (耗时 {t1-t0:.1f}s)")

        results.append({
            'label': label, 'w_time': w_time, 'w_util': w_util,
            'time': total_time, 'util': seat_util,
            'flights': n_flights, 'fuel': total_fuel,
            'failed': False,
        })

    # ---------- 6. Pareto 前沿分析 ----------
    print("\n[6] Pareto 前沿分析...")
    pareto_flags = []
    for i, r in enumerate(results):
        if r.get('failed'):
            pareto_flags.append(False)
            continue
        dominated = is_dominated(i, results)
        pareto_flags.append(not dominated)

    pareto_solutions = [r for r, p in zip(results, pareto_flags) if p]
    print(f"  Pareto 最优解数量: {len(pareto_solutions)} / {len([r for r in results if not r.get('failed')])}")
    for r, p in zip(results, pareto_flags):
        if p:
            print(f"    * {r['label']}: 时间={r['time']}min, 利用率={r['util']*100:.2f}%, "
                  f"架次={r['flights']}, 燃油={r['fuel']:.1f}kg")

    # ---------- 7. 输出结果表格 ----------
    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)
    print_results_table(results, pareto_flags)

    # ---------- 8. Pareto 前沿详细分析 ----------
    print("\n" + "-" * 70)
    print("Pareto 前沿分析（最小化时间/架次/燃油，最大化利用率）")
    print("-" * 70)
    if not pareto_solutions:
        print("  无 Pareto 最优解（可能存在求解失败）。")
    else:
        # 按 W_UTIL 排序展示前沿
        pareto_sorted = sorted(pareto_solutions, key=lambda r: r['w_util'])
        prev = None
        for r in pareto_sorted:
            if prev is not None:
                # 展示相邻 Pareto 解的权衡
                dt = r['time'] - prev['time']
                du = (r['util'] - prev['util']) * 100
                df = r['fuel'] - prev['fuel']
                dn = r['flights'] - prev['flights']
                print(f"  {prev['label']} -> {r['label']}: "
                      f"时间{dt:+d}min, 利用率{du:+.2f}%, "
                      f"架次{dn:+d}, 燃油{df:+.1f}kg")
            prev = r
        print()
        print("  权衡解读:")
        print("    - 增大 W_UTIL（利用率权重）倾向于选择座位利用率更高的方案，")
        print("      通常以增加总时间为代价（更紧凑的拼机带来更长绕行）。")
        print("    - 增大 W_TIME（时间权重）倾向于最小化总飞机使用时间，")
        print("      可能导致利用率下降（更多空座或更多架次）。")
        print("    - Pareto 前沿上的解代表了时间与利用率之间不可同时改善的权衡边界。")

    # ---------- 9. 保存结果到文件 ----------
    csv_path = os.path.join(OUTPUT_DIR, 'pareto_sweep_results.csv')
    save_results_csv(results, pareto_flags, csv_path)
    print(f"\n结果已保存: {csv_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
