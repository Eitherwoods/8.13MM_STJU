# -*- coding: utf-8 -*-
"""
文件: column_generation.py
功能: C8 简化列生成（Reactive Column Generation）单日验证脚本
项目: 2026策联杯 B题 - 海上油田人员直升机运载计划编排

思路:
  第一步: 用初始路径池（原始 DEPARTURE_OFFSETS）求解 MIP，得到解
  第二步: 分析解中哪些设施的服务质量差（出海载客率 < 0.5 即为薄弱）
  第三步: 为薄弱设施生成新路径（更密集起飞时刻 + 更多多设施组合）
  第四步: 将新路径加入池中重新求解 MIP
  第五步: 比较两次求解结果（路径数、架次、利用率、总时间、燃油）

运行:
  E:\\Python\\python.exe column_generation.py
"""

import os
import sys
from datetime import datetime, timedelta, date
from collections import defaultdict

# ==================== 模块路径 ====================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from config import AIRCRAFT_TYPES, AIRPORTS
from base_models import make_route, TSPSolver
from problem2_solution import Q2RoutePreprocessor
from problem3_solution import (
    Q3DataLoader,
    Q3DemandDayAllocator,
    Q3ScheduleHelper,
    Q3RouteGenerator,
    Q3DailySolver,
    DEPARTURE_OFFSETS,
    PLANNING_DATES,
    OPERATION_START_HOUR,
    RETURN_DEADLINE_HOUR,
    DEPARTURE_DEADLINE_HOUR,
)

# ==================== 列生成参数 ====================
TEST_DAY = date(2026, 8, 3)          # 单日测试日期
WEAK_THRESHOLD = 0.5                 # 薄弱设施载客率阈值
DENSE_OFFSET_STEP = 30               # 密集起飞时刻步长（分钟）
WEAK_FALLBACK_TOPK = 3              # 无薄弱设施时取载客率最低的 K 个设施
WEAK_TOPK = 5                        # 参与列生成的薄弱设施数量上限（控制新路径规模）
MAX_NEW_ROUTES = 6000                # 新生成路径数量上限（防止第二次MIP规模过大）


def patch_solver_params():
    """通过修改类属性设置求解器参数（monkey-patching，不改动原文件）"""
    # 单次 Dinkelbach 迭代，单档时间限制 90 秒
    Q3DailySolver.DINKELBACH_MAX_ITERS = 1
    Q3DailySolver.TIME_LIMITS = [90]
    # 加权单目标权重
    Q3DailySolver.USE_WEIGHTED = True
    Q3DailySolver.W_TIME = 1.0
    Q3DailySolver.W_UTIL = 0.15
    Q3DailySolver.W_TRANSIT = 0.01
    Q3DailySolver.W_FUEL = 0.005


def build_existing_keys(routes_info, day):
    """
    构建已存在路径的去重键集合，用于新路径生成时去重

    去重键: (机场, 机型, 完整路径元组, 加油点集合, 起飞偏移分钟)
    """
    day_start = datetime(day.year, day.month, day.day, OPERATION_START_HOUR, 0)
    keys = set()
    for ri in routes_info:
        r = ri['route']
        offset_min = int((ri['departure_time'] - day_start).total_seconds() / 60)
        key = (r.airport, r.aircraft_type, tuple(r.path), frozenset(r.refuel_at), offset_min)
        keys.add(key)
    return keys


def compute_metrics(sel_routes, yo_vals, yi_vals, ys_vals, routes_info, pool_size, distances):
    """
    从求解结果计算对比指标

    参数:
        sel_routes: 选中的 routes_info 子集（与 routes_info 中元素为同一对象）
        yo_vals/yi_vals/ys_vals: 求解器返回的人员分配字典，键中 ri_idx 为 routes_info 索引
        routes_info: 当前路径池
        pool_size: 路径池大小
        distances: 距离矩阵

    返回:
        指标字典
    """
    # 建立 routes_info 对象 id -> 索引 的映射，用于定位 yo/yi/ys 中的键
    idx_map = {id(ri): idx for idx, ri in enumerate(routes_info)}

    total_time = 0
    total_fuel = 0.0
    total_transit = 0
    pass_km = 0.0
    seat_km = 0.0
    n_people = 0

    for ri in sel_routes:
        idx = idx_map[id(ri)]
        r = ri['route']
        # 确保 precomputed 已生成（求解过程通常会生成，此处兜底）
        pre = ri.get('precomputed')
        if pre is None:
            pre = Q2RoutePreprocessor.preprocess(r, distances)
            ri['precomputed'] = pre

        total_time += r.time
        total_fuel += r.fuel
        seat_km += r.capacity * r.total_distance

        # 出海/海返人员：按设施累加客公里与在途时间
        for f in pre['facility_pos']:
            v_o = yo_vals.get((idx, f), 0)
            v_i = yi_vals.get((idx, f), 0)
            n_people += v_o + v_i
            pass_km += v_o * pre['dist_to'].get(f, 0)
            pass_km += v_i * pre['dist_from'].get(f, 0)
            total_transit += v_o * pre['time_to'].get(f, 0)
            total_transit += v_i * pre['time_from'].get(f, 0)

        # 穿梭人员：按 OD 对累加
        for (f1, f2) in pre['shuttle_info']:
            v_s = ys_vals.get((idx, f1, f2), 0)
            n_people += v_s
            si = pre['shuttle_info'][(f1, f2)]
            pass_km += v_s * si['dist']
            total_transit += v_s * si['time']

    utilization = pass_km / seat_km if seat_km > 0 else 0.0
    return {
        'pool_size': pool_size,
        'n_flights': len(sel_routes),
        'n_people': n_people,
        'total_time': total_time,
        'total_transit': total_transit,
        'total_fuel': total_fuel,
        'pass_km': pass_km,
        'seat_km': seat_km,
        'utilization': utilization,
    }


def analyze_weak_facilities(sel_routes, yo_vals, routes_info, day_demands):
    """
    薄弱环节分析：对每个出海目的设施 f 计算
        载客率 = sum(yo[r,f]) / sum(capacity * x_r)
    若载客率 < WEAK_THRESHOLD 则判定为薄弱

    参数:
        sel_routes: 选中的路径
        yo_vals: 出海人员分配
        routes_info: 路径池
        day_demands: 当日需求

    返回:
        (weak_facs, load_info)
        weak_facs: 薄弱设施列表
        load_info: {设施: {'load': 载客率, 'pax': 人数, 'cap': 容量}} 全部设施信息
    """
    idx_map = {id(ri): idx for idx, ri in enumerate(routes_info)}

    # 当日所有出海目的设施
    ob_facs = set()
    for d in day_demands:
        if d['destination_id'].startswith('F'):
            ob_facs.add(d['destination_id'])

    # 按设施汇总：服务该设施的选中路径的容量与出海人数
    fac_pax = defaultdict(int)
    fac_cap = defaultdict(int)
    for ri in sel_routes:
        idx = idx_map[id(ri)]
        r = ri['route']
        pre = ri.get('precomputed', {})
        for f in pre.get('facility_pos', {}):
            if f in ob_facs:
                fac_pax[f] += yo_vals.get((idx, f), 0)
                fac_cap[f] += r.capacity

    load_info = {}
    weak_facs = []
    for f in ob_facs:
        cap = fac_cap.get(f, 0)
        pax = fac_pax.get(f, 0)
        load = pax / cap if cap > 0 else 0.0
        load_info[f] = {'load': load, 'pax': pax, 'cap': cap}
        if cap > 0 and load < WEAK_THRESHOLD:
            weak_facs.append(f)

    return weak_facs, load_info


def generate_new_routes(weak_facs, day_demands, day, distances, existing_keys, rid_counter):
    """
    为薄弱设施生成新路径（列生成核心）

    生成策略:
      1. 更密集起飞时刻：每 30 分钟一个，覆盖 06:00-18:00
      2. 更多多设施组合：薄弱设施与近邻设施的单/双/三设施组合

    参数:
        weak_facs: 薄弱设施列表
        day_demands: 当日需求（用于兼容性检查）
        day: 日期
        distances: 距离矩阵
        existing_keys: 已存在路径去重键集合（会被本函数更新）
        rid_counter: [当前可用 route_id]，单元素列表用于可变传递

    返回:
        (new_routes_info, n_base)
        new_routes_info: 新生成的 routes_info 列表
        n_base: 生成的基础路径（Route 对象）数量
    """
    # 密集起飞时刻：06:00(offset=0) 到 18:00(offset=720)，每 30 分钟
    dense_offsets = list(range(0, (DEPARTURE_DEADLINE_HOUR - OPERATION_START_HOUR) * 60 + 1,
                               DENSE_OFFSET_STEP))
    day_start = datetime(day.year, day.month, day.day, OPERATION_START_HOUR, 0)
    day_end = datetime(day.year, day.month, day.day, RETURN_DEADLINE_HOUR, 0)

    # 当日涉及的所有设施
    all_facs = set()
    for d in day_demands:
        if d['origin_id'].startswith('F'):
            all_facs.add(d['origin_id'])
        if d['destination_id'].startswith('F'):
            all_facs.add(d['destination_id'])
    all_facs = sorted(all_facs)

    new_routes_info = []
    n_base = 0

    for airport in AIRPORTS:
        # 汇总薄弱设施相关的设施组合（按 frozenset 去重，避免重复生成）
        specs = set()
        for wf in weak_facs:
            specs.add(frozenset([wf]))  # 单设施
            # 按距离排序取近邻，构建双设施组合（仅取最近 3 个，控制规模）
            others = sorted([f for f in all_facs if f != wf],
                            key=lambda f: distances[wf][f])
            for f2 in others[:3]:
                specs.add(frozenset([wf, f2]))
            # 三设施组合：薄弱设施 + 最近邻 + 次近邻（仅 2 种，控制规模）
            if len(others) >= 2:
                specs.add(frozenset([wf, others[0], others[1]]))
            if len(others) >= 3:
                specs.add(frozenset([wf, others[0], others[2]]))

        # 为每个设施组合生成路径及起飞时刻变体
        for spec in specs:
            facs = list(spec)
            best_order, _ = TSPSolver.best_order(airport, facs, distances)
            for atype in AIRCRAFT_TYPES:
                r = make_route(rid_counter[0], airport, atype, best_order, distances)
                rid_counter[0] += 1
                if r is None:
                    continue
                n_base += 1
                # 最晚可行起飞偏移：保证返航不超时且不晚于起飞截止
                latest_offset = min(
                    (RETURN_DEADLINE_HOUR - OPERATION_START_HOUR) * 60 - r.time,
                    (DEPARTURE_DEADLINE_HOUR - OPERATION_START_HOUR) * 60)
                for offset in dense_offsets:
                    if offset > latest_offset:
                        continue
                    # 去重：与已有路径（含初始池与已生成新路径）比对
                    key = (airport, atype, tuple(r.path), frozenset(r.refuel_at), offset)
                    if key in existing_keys:
                        continue
                    dep_time = day_start + timedelta(minutes=offset)
                    schedule = Q3ScheduleHelper.compute_schedule(r, dep_time, distances)
                    ret_time = schedule[len(r.path) - 1][0]
                    if ret_time > day_end:
                        continue
                    # 时间窗兼容性检查：仅保留能服务至少一条需求的路径
                    compatible = [d for d in day_demands
                                  if Q3ScheduleHelper.check_compatibility(d, r, schedule, day)[0]]
                    if compatible:
                        slots = Q3ScheduleHelper.get_route_time_slots(dep_time, ret_time, day)
                        new_routes_info.append({
                            'route': r,
                            'schedule': schedule,
                            'departure_time': dep_time,
                            'return_time': ret_time,
                            'compatible_demands': compatible,
                            'time_slots': slots,
                        })
                        existing_keys.add(key)
                        # 数量上限保护：防止第二次 MIP 规模过大导致求解质量下降
                        if len(new_routes_info) >= MAX_NEW_ROUTES:
                            return new_routes_info, n_base

    return new_routes_info, n_base


def print_load_info(load_info):
    """打印各设施载客率明细"""
    print("\n  设施载客率明细（出海）:")
    print(f"  {'设施':<8}{'人数':>6}{'容量':>8}{'载客率':>10}")
    for f in sorted(load_info.keys()):
        info = load_info[f]
        mark = ' *薄弱*' if info['load'] < WEAK_THRESHOLD else ''
        print(f"  {f:<8}{info['pax']:>6}{info['cap']:>8}{info['load']:>10.4f}{mark}")


def print_comparison(m1, m2):
    """打印两次求解的对比表"""
    def fmt_delta(new, old, better_is_lower=True):
        """格式化变化量与变化率，标注改善/恶化"""
        delta = new - old
        if abs(old) > 1e-9:
            pct = delta / old * 100
        else:
            pct = 0.0
        sign = '+' if delta >= 0 else ''
        # 判断方向：better_is_lower 表示数值越小越好
        if abs(delta) < 1e-9:
            tag = '-'
        elif (delta < 0) == better_is_lower:
            tag = '改善'
        else:
            tag = '恶化'
        return f"{sign}{delta:.2f} ({sign if pct>=0 else ''}{pct:.2f}%) [{tag}]"

    print("\n" + "=" * 78)
    print("C8 简化列生成 - 两次求解对比")
    print("=" * 78)
    header = f"  {'指标':<16}{'初始池求解':>16}{'增强池求解':>16}{'变化':>30}"
    print(header)
    print("  " + "-" * 74)

    rows = [
        ('路径池大小', m1['pool_size'], m2['pool_size'], fmt_delta(m2['pool_size'], m1['pool_size'], True), 'd'),
        ('选中架次', m1['n_flights'], m2['n_flights'], fmt_delta(m2['n_flights'], m1['n_flights'], True), 'd'),
        ('运送人数', m1['n_people'], m2['n_people'], fmt_delta(m2['n_people'], m1['n_people'], False), 'd'),
        ('总时间(min)', m1['total_time'], m2['total_time'], fmt_delta(m2['total_time'], m1['total_time'], True), 'd'),
        ('人员在途(min)', m1['total_transit'], m2['total_transit'], fmt_delta(m2['total_transit'], m1['total_transit'], True), 'd'),
        ('燃油(kg)', m1['total_fuel'], m2['total_fuel'], fmt_delta(m2['total_fuel'], m1['total_fuel'], True), '.1f'),
        ('客座公里', m1['pass_km'], m2['pass_km'], fmt_delta(m2['pass_km'], m1['pass_km'], False), '.0f'),
        ('座位公里', m1['seat_km'], m2['seat_km'], fmt_delta(m2['seat_km'], m1['seat_km'], True), '.0f'),
        ('利用率', m1['utilization'], m2['utilization'], fmt_delta(m2['utilization'], m1['utilization'], False), '.4f'),
    ]
    for name, v1, v2, delta_str, num_fmt in rows:
        s1 = format(v1, num_fmt)
        s2 = format(v2, num_fmt)
        print(f"  {name:<16}{s1:>16}{s2:>16}{delta_str:>30}")
    print("=" * 78)


def print_interpretation(m1, m2, weak_facs):
    """打印列生成效果解读说明"""
    print("\n" + "-" * 78)
    print("结果解读:")
    print("-" * 78)
    # 超集性质：增强池包含初始池，故真实最优不劣于第一次
    print("  1. 增强路径池是初始池的超集，故第二次求解的真实最优不劣于第一次。")
    print("     两次求解均受 90s 时间限制（MIP 间隙约 80%），绝对目标值含求解器噪声，")
    print("     个别指标的微小波动属正常现象。")
    # 列生成的核心价值：密集起飞时刻改善需求时间窗匹配
    d_transit = m2['total_transit'] - m1['total_transit']
    d_seatkm = m2['seat_km'] - m1['seat_km']
    print(f"  2. 列生成核心价值在于为薄弱设施（{weak_facs}）补充更密集的起飞时刻变体，")
    print("     使需求时间窗与架次时刻更匹配。重点关注两项服务质量指标:")
    transit_tag = '下降(改善)' if d_transit < 0 else '上升'
    seatkm_tag = '下降(改善)' if d_seatkm < 0 else '上升'
    print(f"     - 人员在途时间: {m1['total_transit']} -> {m2['total_transit']} min ({transit_tag})")
    print(f"     - 座位公里(空载成本): {m1['seat_km']:.0f} -> {m2['seat_km']:.0f} ({seatkm_tag})")
    print("  3. 若人员在途时间与座位公里下降，说明密集起飞时刻确实改善了薄弱环节的")
    print("     服务质量——这正是简化列生成（reactive column generation）的目标。")
    print("-" * 78)


def main():
    """主函数：编排简化列生成流程"""
    print("=" * 78)
    print("C8 简化列生成（Reactive Column Generation）- 单日验证")
    print(f"测试日期: {TEST_DAY}")
    print("=" * 78)

    # 0. 配置求解器参数
    patch_solver_params()

    # 1. 加载数据
    print("\n[1] 加载数据...")
    loader = Q3DataLoader()
    distances = loader.load_distances()
    all_demands = loader.load_demands_q3()
    non_temp = [d for d in all_demands if d['task_type'] != 'temporary']
    print(f"  总需求: {len(all_demands)}, 非临时(阶段一): {len(non_temp)}")

    # 2. 需求-日期分配（沿用原模型的多日分配逻辑，保证与主流程一致）
    print("\n[2] 需求-日期分配...")
    allocation, daily_load = Q3DemandDayAllocator.allocate(non_temp, PLANNING_DATES)
    day_demands = [d for d in non_temp if allocation[d['person_id']] == TEST_DAY]
    print(f"  {TEST_DAY} 分配需求: {len(day_demands)} 人")
    if not day_demands:
        print("  错误: 该日无需求，终止")
        return

    # 3. 生成初始路径池（使用原始 DEPARTURE_OFFSETS）
    print("\n[3] 生成初始路径池（原始 DEPARTURE_OFFSETS）...")
    generator = Q3RouteGenerator(distances)
    # monkey-patch: 让 _compute_cluster_offsets 返回 DEPARTURE_OFFSETS，
    # 使得 all_offsets = sorted(set(fixed_essential + DEPARTURE_OFFSETS)) = DEPARTURE_OFFSETS
    # （因 fixed_essential 是其子集），从而初始池使用原始起飞偏移
    original_cluster_fn = Q3RouteGenerator._compute_cluster_offsets
    Q3RouteGenerator._compute_cluster_offsets = lambda self, dd, day: list(DEPARTURE_OFFSETS)
    try:
        rid_counter = [0]
        routes_info = generator.generate_daily_routes(day_demands, TEST_DAY, rid_counter[0])
    finally:
        # 恢复原始方法，避免影响后续逻辑
        Q3RouteGenerator._compute_cluster_offsets = original_cluster_fn

    if routes_info:
        rid_counter[0] = max(r['route'].route_id for r in routes_info)
    print(f"  初始候选路径: {len(routes_info)} 条")

    # 4. 第一次求解 MIP
    print("\n[4] 第一次求解 MIP（初始路径池）...")
    solver = Q3DailySolver()
    result1 = solver.solve(routes_info, day_demands, TEST_DAY, distances)
    if result1 is None:
        print("  错误: 第一次求解无可行解，终止")
        return
    sel_routes_1, yo1, yi1, ys1 = result1

    metrics1 = compute_metrics(sel_routes_1, yo1, yi1, ys1, routes_info,
                               pool_size=len(routes_info), distances=distances)
    print(f"\n  [第一次求解] 架次={metrics1['n_flights']}, 人数={metrics1['n_people']}, "
          f"时间={metrics1['total_time']}min, 燃油={metrics1['total_fuel']:.1f}kg, "
          f"利用率={metrics1['utilization']:.4f}")

    # 5. 薄弱环节分析
    print("\n[5] 薄弱环节分析...")
    weak_facs, load_info = analyze_weak_facilities(sel_routes_1, yo1, routes_info, day_demands)
    print_load_info(load_info)

    if not weak_facs:
        # 兜底：若无设施低于阈值，取载客率最低的若干设施演示列生成
        ranked = sorted(load_info.items(), key=lambda kv: kv[1]['load'])
        weak_facs = [f for f, _ in ranked[:WEAK_FALLBACK_TOPK] if load_info[f]['cap'] > 0]
        print(f"\n  无设施载客率 < {WEAK_THRESHOLD}，取载客率最低 {len(weak_facs)} 个设施作为薄弱: {weak_facs}")
    else:
        # 按载客率升序，仅取最薄弱的 WEAK_TOPK 个设施（控制新路径规模）
        weak_facs = sorted(weak_facs, key=lambda f: load_info[f]['load'])[:WEAK_TOPK]
        print(f"\n  薄弱设施（载客率 < {WEAK_THRESHOLD}），取最薄弱 {len(weak_facs)} 个: {weak_facs}")

    # 6. 为薄弱设施生成新路径
    print("\n[6] 为薄弱设施生成新路径（密集起飞时刻 + 多设施组合）...")
    existing_keys = build_existing_keys(routes_info, TEST_DAY)
    new_routes, n_base = generate_new_routes(weak_facs, day_demands, TEST_DAY, distances,
                                             existing_keys, rid_counter)
    print(f"  生成基础路径(Route): {n_base} 个")
    print(f"  新增候选路径(含时刻变体): {len(new_routes)} 条")

    if not new_routes:
        print("  未生成新路径，跳过第二次求解")
        return

    # 7. 将新路径加入池中，第二次求解
    print("\n[7] 第二次求解 MIP（增强路径池）...")
    routes_info_2 = routes_info + new_routes
    # 复用同一个 solver 实例（参数已通过类属性设置）
    result2 = solver.solve(routes_info_2, day_demands, TEST_DAY, distances)
    if result2 is None:
        print("  错误: 第二次求解无可行解，仅输出第一次结果")
        return
    sel_routes_2, yo2, yi2, ys2 = result2

    metrics2 = compute_metrics(sel_routes_2, yo2, yi2, ys2, routes_info_2,
                               pool_size=len(routes_info_2), distances=distances)
    print(f"\n  [第二次求解] 架次={metrics2['n_flights']}, 人数={metrics2['n_people']}, "
          f"时间={metrics2['total_time']}min, 燃油={metrics2['total_fuel']:.1f}kg, "
          f"利用率={metrics2['utilization']:.4f}")

    # 8. 对比
    print_comparison(metrics1, metrics2)
    print_interpretation(metrics1, metrics2, weak_facs)


if __name__ == '__main__':
    main()
