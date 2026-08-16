# -*- coding: utf-8 -*-
"""
文件: problem3_solution.py
功能: 问题3求解 — 带时间要求的多日排班
方法: 按天解耦 + 两阶段法（受限路径池 + COPT加权单目标MIP + 背包化临时任务插入）
创新点:
  C1. 按天解耦定理：多日问题分解为7个独立单日问题
  C2. 时间窗CPM收缩预处理
  C3. 受限路径池+时间窗过滤
  C4. 阶段二临时任务背包化
  C5. 拉格朗日下界+最优性间隙认证（见lagrangian_bound.py）
  C6. 时间扩展网络统一路由+调度+机队（按机型分解约束）
  C7. 时间窗感知需求聚类路径生成（实验为负结果，方法保留供对比）
  C8. 反应式列生成（见column_generation.py，边际改善）
  C9. 加权灵敏度分析Pareto前沿（见pareto_sweep.py，验证权重Pareto最优）
项目: 2026策联杯 B题
"""

import csv
import os
import sys
import math
import time as time_module
from datetime import datetime, timedelta, date
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple, Optional

# ==================== 模块路径 ====================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VIBECODING_DIR = os.path.dirname(_THIS_DIR)
if _VIBECODING_DIR not in sys.path:
    sys.path.insert(0, _VIBECODING_DIR)
# problem12 子包路径（跨问题导入 problem1_solution/problem2_solution）
_PROBLEM12_DIR = os.path.join(_VIBECODING_DIR, 'problem12')
if _PROBLEM12_DIR not in sys.path:
    sys.path.insert(0, _PROBLEM12_DIR)

import coptpy as cp
from config import (
    AIRCRAFT, AIRCRAFT_TYPES, AIRPORTS, REFUEL_FACILITIES,
    MAX_LANDINGS, DATA_DIR, STOP_TIME_NO_REFUEL, STOP_TIME_REFUEL, fmt6
)
from base_models import (
    DataLoader, FuelChecker, TSPSolver, Route,
    calculate_time, make_route
)
from problem2_solution import Q2RoutePreprocessor
from problem1_solution import RouteGenerator

# ==================== Q3配置 ====================
FLEET = {
    'A01': {'T1': 3, 'T2': 3, 'T3': 2},
    'A02': {'T1': 2, 'T2': 4, 'T3': 2},
    'A03': {'T1': 2, 'T2': 3, 'T3': 3},
}
OPERATION_START_HOUR = 6
DEPARTURE_DEADLINE_HOUR = 18
RETURN_DEADLINE_HOUR = 20
TURNAROUND_MINUTES = 30
TIME_SLOT_MINUTES = 10
PLANNING_DATES = [date(2026, 8, d) for d in range(3, 10)]
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR_Q3 = os.path.join(PROJECT_DIR, 'output', 'Problem3')
# 起飞时刻偏移列表（分钟），06:00-10:00用30min粒度，10:00后用60min
DEPARTURE_OFFSETS = [0, 30, 60, 90, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660]


# ==================== Q3数据加载器 ====================
class Q3DataLoader(DataLoader):
    """Q3数据加载器：加载带时间窗和任务类型的4000条需求"""

    def load_demands_q3(self):
        """加载Q3需求数据，返回含时间窗和任务类型的需求列表"""
        demands = []
        path = os.path.join(DATA_DIR, 'peopleQ3.csv')
        with open(path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                demands.append({
                    'person_id': row['person_id'],
                    'origin_id': row['origin_id'],
                    'destination_id': row['destination_id'],
                    'earliest_pickup': datetime.strptime(
                        row['earliest_pickup_time'].strip(), "%Y-%m-%d %H:%M"),
                    'latest_arrival': datetime.strptime(
                        row['latest_arrival_time'].strip(), "%Y-%m-%d %H:%M"),
                    'task_type': row['task_type'],
                })
        return demands


# ==================== 需求-日期分配器（定理1推论1.2） ====================
class Q3DemandDayAllocator:
    """计算每条需求的可行日期集合D_p，贪心分配到负载最低的可行日期"""

    @staticmethod
    def compute_feasible_days(demand, dates):
        """计算需求可行日期集合：时间窗与运营时段有交集的日期"""
        feasible = []
        for d in dates:
            day_start = datetime(d.year, d.month, d.day, OPERATION_START_HOUR, 0)
            day_end = datetime(d.year, d.month, d.day, RETURN_DEADLINE_HOUR, 0)
            if demand['earliest_pickup'] < day_end and demand['latest_arrival'] > day_start:
                feasible.append(d)
        return feasible

    @staticmethod
    def _time_window_bucket(demand):
        """将需求按最早接送时间归入2小时桶，用于时间窗聚类均衡"""
        hour = demand['earliest_pickup'].hour
        return hour // 2

    @staticmethod
    def allocate(demands, dates):
        """
        贪心分配需求到日期，按可行日期数升序分配。
        负载度量同时考虑：
          1. 总人数（避免某日过载）
          2. 需求类型分布（出海/海返/穿梭，避免类型集中）
          3. 时间窗桶分布（2小时粒度，避免同时段需求集中导致机队冲突）
        """
        demand_days = {}
        for d in demands:
            demand_days[d['person_id']] = Q3DemandDayAllocator.compute_feasible_days(d, dates)

        sorted_demands = sorted(demands, key=lambda d: len(demand_days[d['person_id']]))

        daily_load = {d: 0 for d in dates}
        daily_type_load = {d: defaultdict(int) for d in dates}
        daily_bucket_load = {d: defaultdict(int) for d in dates}
        allocation = {}

        for d in sorted_demands:
            feasible = demand_days[d['person_id']]
            if not feasible:
                forced = d['earliest_pickup'].date()
                target = forced if forced in daily_load else dates[0]
                allocation[d['person_id']] = target
                daily_load[target] += 1
                daily_type_load[target][d['task_type']] += 1
                bucket = Q3DemandDayAllocator._time_window_bucket(d)
                daily_bucket_load[target][bucket] += 1
                continue

            bucket = Q3DemandDayAllocator._time_window_bucket(d)

            def load_key(dd):
                total = daily_load[dd]
                type_count = daily_type_load[dd][d['task_type']]
                bucket_count = daily_bucket_load[dd][bucket]
                return (total, type_count, bucket_count)

            best_date = min(feasible, key=load_key)
            allocation[d['person_id']] = best_date
            daily_load[best_date] += 1
            daily_type_load[best_date][d['task_type']] += 1
            daily_bucket_load[best_date][bucket] += 1

        return allocation, daily_load


# ==================== 时间窗处理器（命题2） ====================
class Q3ScheduleHelper:
    """时刻表计算与需求-路径时间窗兼容性检查"""

    @staticmethod
    def get_effective_window(demand, day):
        """获取需求在某日的有效时间窗（收缩到运营时段内）"""
        day_start = datetime(day.year, day.month, day.day, OPERATION_START_HOUR, 0)
        day_end = datetime(day.year, day.month, day.day, RETURN_DEADLINE_HOUR, 0)
        e = max(demand['earliest_pickup'], day_start)
        l = min(demand['latest_arrival'], day_end)
        return e, l

    @staticmethod
    def compute_schedule(route, departure_time, distances):
        """计算路径在给定起飞时刻下各停靠点的时刻表"""
        path = route.path
        speed = AIRCRAFT[route.aircraft_type]['speed']
        schedule = {}
        schedule[0] = (None, departure_time)
        current_time = departure_time

        for i in range(1, len(path)):
            flight_min = distances[path[i-1]][path[i]] / speed * 60.0
            arrival = current_time + timedelta(minutes=math.ceil(flight_min))
            if i == len(path) - 1:
                schedule[i] = (arrival, None)
            else:
                stop = STOP_TIME_REFUEL if i in route.refuel_at else STOP_TIME_NO_REFUEL
                departure = arrival + timedelta(minutes=stop)
                schedule[i] = (arrival, departure)
                current_time = departure
        return schedule

    @staticmethod
    def check_compatibility(demand, route, schedule, day):
        """检查需求与路径固定时刻表的时间窗兼容性，返回(compatible, pickup_idx, delivery_idx)"""
        e, l = Q3ScheduleHelper.get_effective_window(demand, day)
        path = route.path
        o = demand['origin_id']
        de = demand['destination_id']
        o_actual = route.airport if o == 'LAND' else o
        de_actual = route.airport if de == 'LAND' else de

        pickup_idx, delivery_idx = None, None
        for i in range(len(path)):
            if path[i] == o_actual and pickup_idx is None:
                pickup_idx = i
            if path[i] == de_actual and pickup_idx is not None and i > pickup_idx:
                delivery_idx = i
                break

        if pickup_idx is None or delivery_idx is None:
            return False, None, None

        leave_time = schedule[pickup_idx][1]
        arrive_time = schedule[delivery_idx][0]
        if leave_time >= e and arrive_time <= l:
            return True, pickup_idx, delivery_idx
        return False, None, None

    @staticmethod
    def get_route_time_slots(departure_time, return_time, day):
        """计算路径覆盖的时间槽索引列表（含30分钟周转）"""
        day_start = datetime(day.year, day.month, day.day, OPERATION_START_HOUR, 0)
        occupied_end = return_time + timedelta(minutes=TURNAROUND_MINUTES)
        start_min = int((departure_time - day_start).total_seconds() / 60)
        end_min = int((occupied_end - day_start).total_seconds() / 60)
        start_slot = max(0, start_min // TIME_SLOT_MINUTES)
        end_slot = min(84, (end_min + TIME_SLOT_MINUTES - 1) // TIME_SLOT_MINUTES)
        return list(range(start_slot, end_slot))


# ==================== Q3路径生成器（创新点C3 + C7） ====================
class Q3RouteGenerator:
    """
    Q3路径生成器：复用节省算法 + 时间窗过滤 + 起飞时刻变体 + 时间窗聚类(C7)

    剪枝策略:
    1. 双设施路径: 节省值排序，取前300对（与Q1规模对齐）
    2. 三设施路径: 距离阈值 ≤ 120km，数量上限 300条
    3. 四设施路径: 距离阈值 ≤ 100km，数量上限 300条（从最优三设施扩展）
    4. 五设施路径: 距离阈值 ≤ 80km，累计上限 300条（从最优四设施扩展）
    5. 穿梭OD对专用路径: 确保所有有穿梭需求的设施对至少有一条覆盖路径
    6. 燃油检查剪枝: 通过make_route函数调用FuelChecker.check_fuel，自动应用提前剪枝
    7. 起飞偏移: 14个偏移（06:00-10:00用30min粒度），平衡覆盖率与MIP规模

    设计原理:
    - Q3有4000人（Q1的2.5倍），路径池必须相应扩大
    - 4-5设施多停靠路径能显著提升架次效率，降低总飞机使用时间
    - 渐进式距离阈值保证燃油可行性
    - 14个偏移控制MIP规模，确保2h内可收敛
    """

    def __init__(self, distances, offsets=None,
                 pair_cap=300, triple_cap=300, quad_cap=600, quint_cap=900):
        """
        参数:
            distances: 距离矩阵
            offsets: 可选起飞偏移列表（分钟），None 时使用模块级 DEPARTURE_OFFSETS。
                传入自定义偏移列表可精简路径池（如按需求时间窗聚类后的关键时刻），
                大幅压缩 MIP 规模。
            pair_cap/triple_cap/quad_cap/quint_cap: 多设施路径数量上限。
                默认 300/300/600/900（与原始行为一致）；传入更小值可进一步
                压缩路径池规模（牺牲部分多停靠覆盖换取 MIP 可解性）。
        """
        self.distances = distances
        self.offsets = offsets if offsets is not None else DEPARTURE_OFFSETS
        self.pair_cap = pair_cap
        self.triple_cap = triple_cap
        self.quad_cap = quad_cap
        self.quint_cap = quint_cap

    def _compute_cluster_offsets(self, day_demands, day):
        """
        C7: 基于需求时间窗聚类计算起飞时刻偏移
        将需求按时间窗中心聚类，取簇中心作为起飞时刻
        """
        day_start = datetime(day.year, day.month, day.day, OPERATION_START_HOUR, 0)
        day_end = datetime(day.year, day.month, day.day, RETURN_DEADLINE_HOUR, 0)

        centers = []
        for d in day_demands:
            e = d['earliest_pickup']
            l = d['latest_arrival']
            e_eff = max(e, day_start)
            l_eff = min(l, day_end)
            if l_eff > e_eff:
                center_ts = (e_eff.timestamp() + l_eff.timestamp()) / 2
                center = datetime.fromtimestamp(center_ts)
                offset = int((center - day_start).total_seconds() / 60)
                if 0 <= offset <= (DEPARTURE_DEADLINE_HOUR - OPERATION_START_HOUR) * 60:
                    centers.append(offset)

        if not centers:
            return []

        centers.sort()
        K = min(12, max(4, len(centers) // 50))

        cluster_offsets = []
        for k in range(K):
            idx = int(k * len(centers) / K)
            cluster_offsets.append(centers[idx])

        cluster_offsets = sorted(set(cluster_offsets))
        return cluster_offsets

    def generate_daily_routes(self, day_demands, day, route_id_start=1):
        """为某日生成候选路径（含多起飞时刻变体 + C7聚类时刻）"""
        airport_set = set(AIRPORTS) | {'LAND'}
        all_facilities = set()
        for d in day_demands:
            if d['origin_id'].startswith('F'):
                all_facilities.add(d['origin_id'])
            if d['destination_id'].startswith('F'):
                all_facilities.add(d['destination_id'])
        if not all_facilities:
            return []

        fac_list = sorted(all_facilities)
        base_routes = []
        rid = [route_id_start]
        def next_id():
            rid[0] += 1
            return rid[0]

        for airport in AIRPORTS:
            # 单设施路径
            for fac in fac_list:
                for atype in AIRCRAFT_TYPES:
                    r = make_route(next_id(), airport, atype, [fac], self.distances)
                    if r is not None:
                        base_routes.append(r)
            # 双设施路径（节省算法）
            savings = []
            for i in range(len(fac_list)):
                for j in range(i + 1, len(fac_list)):
                    f1, f2 = fac_list[i], fac_list[j]
                    s = self.distances[f1][airport] + self.distances[airport][f2] - self.distances[f1][f2]
                    if s > 0:
                        savings.append((s, f1, f2))
            savings.sort(reverse=True)
            for s, f1, f2 in savings[:self.pair_cap]:
                best_order, _ = TSPSolver.best_order(airport, [f1, f2], self.distances)
                for atype in AIRCRAFT_TYPES:
                    r = make_route(next_id(), airport, atype, best_order, self.distances)
                    if r is not None:
                        base_routes.append(r)
            # 三设施路径
            pair_groups = defaultdict(list)
            for r in base_routes:
                if r.airport == airport and len(r.dest_facilities) == 2:
                    key = (r.airport, frozenset(r.dest_facilities))
                    pair_groups[key].append(r)
            best_pairs = [min(g, key=lambda r: r.time) for g in pair_groups.values()]
            count = 0
            for r2 in best_pairs:
                if count >= self.triple_cap:
                    break
                existing = list(r2.dest_facilities)
                for f3 in fac_list:
                    if f3 in r2.dest_facilities:
                        continue
                    if min(self.distances[f3][f] for f in existing) > 120:
                        continue
                    best_order, _ = TSPSolver.best_order(airport, existing + [f3], self.distances)
                    for atype in AIRCRAFT_TYPES:
                        r = make_route(next_id(), airport, atype, best_order, self.distances)
                        if r is not None:
                            base_routes.append(r)
                            count += 1

            # 四设施路径（从最优三设施路径扩展，距离阈值 ≤ 100km）
            triple_groups = defaultdict(list)
            for r in base_routes:
                if r.airport == airport and len(r.dest_facilities) == 3:
                    key = (r.airport, frozenset(r.dest_facilities))
                    triple_groups[key].append(r)
            best_triples = [min(g, key=lambda r: r.time) for g in triple_groups.values()]
            for r3 in best_triples:
                if count >= self.quad_cap:
                    break
                existing = list(r3.dest_facilities)
                for f4 in fac_list:
                    if f4 in r3.dest_facilities:
                        continue
                    if min(self.distances[f4][f] for f in existing) > 100:
                        continue
                    best_order, _ = TSPSolver.best_order(airport, existing + [f4], self.distances)
                    for atype in AIRCRAFT_TYPES:
                        r = make_route(next_id(), airport, atype, best_order, self.distances)
                        if r is not None:
                            base_routes.append(r)
                            count += 1

            # 五设施路径（从最优四设施路径扩展，距离阈值 ≤ 80km）
            quad_groups = defaultdict(list)
            for r in base_routes:
                if r.airport == airport and len(r.dest_facilities) == 4:
                    key = (r.airport, frozenset(r.dest_facilities))
                    quad_groups[key].append(r)
            best_quads = [min(g, key=lambda r: r.time) for g in quad_groups.values()]
            for r4 in best_quads:
                if count >= self.quint_cap:
                    break
                existing = list(r4.dest_facilities)
                for f5 in fac_list:
                    if f5 in r4.dest_facilities:
                        continue
                    if min(self.distances[f5][f] for f in existing) > 80:
                        continue
                    best_order, _ = TSPSolver.best_order(airport, existing + [f5], self.distances)
                    for atype in AIRCRAFT_TYPES:
                        r = make_route(next_id(), airport, atype, best_order, self.distances)
                        if r is not None:
                            base_routes.append(r)
                            count += 1

        # 去重
        seen = set()
        unique = []
        for r in base_routes:
            key = (r.airport, r.aircraft_type, tuple(r.path), frozenset(r.refuel_at))
            if key not in seen:
                seen.add(key)
                unique.append(r)
        base_routes = unique

        # 为穿梭OD对生成专用直达路径（确保设施顺序匹配需求方向）
        shuttle_ods = set()
        urgent_shuttle_ods = set()
        for d in day_demands:
            if d['origin_id'].startswith('F') and d['destination_id'].startswith('F'):
                shuttle_ods.add((d['origin_id'], d['destination_id']))
                # 紧急/增储上产穿梭需求时间窗紧，必须生成专用直达路径，
                # 否则多停靠路径的中间停靠会导致到达过晚、需求漏配
                if d['task_type'] in ('emergency', 'production'):
                    urgent_shuttle_ods.add((d['origin_id'], d['destination_id']))
        covered_ods = set()
        for r in base_routes:
            facs = [f for f in r.path if f.startswith('F')]
            for i in range(len(facs)):
                for j in range(i + 1, len(facs)):
                    covered_ods.add((facs[i], facs[j]))
        uncovered_ods = shuttle_ods - covered_ods
        # 未覆盖 OD 与紧急 OD 都需要专用直达路径
        dedicated_ods = uncovered_ods | urgent_shuttle_ods
        for (f1, f2) in dedicated_ods:
            for airport in AIRPORTS:
                for atype in AIRCRAFT_TYPES:
                    r = make_route(next_id(), airport, atype, [f1, f2], self.distances)
                    if r is not None:
                        base_routes.append(r)
                    if (f1, f2) in urgent_shuttle_ods:
                        # 紧急穿梭：额外生成"去程直达 + 返程段加油"的路径，
                        # 避免加油点被自动插入 f1->f2 之间而破坏紧急时间窗
                        for refuel_fac in REFUEL_FACILITIES:
                            if refuel_fac in (f1, f2):
                                continue
                            r2 = make_route(next_id(), airport, atype,
                                            [f1, f2, refuel_fac], self.distances)
                            if r2 is not None:
                                base_routes.append(r2)
        if dedicated_ods:
            print(f"  补充穿梭专用路径: {len(dedicated_ods)}个OD对 "
                  f"(未覆盖{len(uncovered_ods)} + 紧急{len(urgent_shuttle_ods)})")

        # C7实验结论：聚类偏移替换固定偏移导致覆盖率下降，利用率反降
        # 保留原始固定偏移策略（C7方法保留供论文对比分析）
        all_offsets = self.offsets

        # 生成起飞时刻变体并检查兼容性
        routes_info = []
        for r in base_routes:
            latest_offset = min(
                (RETURN_DEADLINE_HOUR - OPERATION_START_HOUR) * 60 - r.time,
                (DEPARTURE_DEADLINE_HOUR - OPERATION_START_HOUR) * 60
            )
            for offset in all_offsets:
                if offset > latest_offset:
                    continue
                dep_time = datetime(day.year, day.month, day.day,
                                    OPERATION_START_HOUR, 0) + timedelta(minutes=offset)
                schedule = Q3ScheduleHelper.compute_schedule(r, dep_time, self.distances)
                ret_time = schedule[len(r.path) - 1][0]
                day_end = datetime(day.year, day.month, day.day, RETURN_DEADLINE_HOUR, 0)
                if ret_time > day_end:
                    continue

                # 检查兼容需求
                compatible = []
                for d in day_demands:
                    compat, _, _ = Q3ScheduleHelper.check_compatibility(d, r, schedule, day)
                    if compat:
                        compatible.append(d)
                if compatible:
                    slots = Q3ScheduleHelper.get_route_time_slots(dep_time, ret_time, day)
                    routes_info.append({
                        'route': r,
                        'schedule': schedule,
                        'departure_time': dep_time,
                        'return_time': ret_time,
                        'compatible_demands': compatible,
                        'time_slots': slots,
                    })
        return routes_info


# ==================== Q3每日求解器（创新点C6） ====================
class Q3DailySolver:
    """单日COPT MIP求解器：集合划分 + 航段容量 + 按机型分解机队约束 + 加权单目标(C9验证Pareto最优)"""

    DINKELBACH_MAX_ITERS = 2
    DINKELBACH_TOL = 0.005
    LAMBDA_INIT = 0.45
    TIME_LIMITS = [7200, 7200]
    TWO_STAGE_TIME_LIMITS = [7200, 7200]  # 两层模式：第一层严格min T / 第二层固定路径优化二级指标
    RETRY_TIME_LIMIT = 7200  # 失败后重试的时间限制
    USE_TWO_STAGE = True  # True=严格时间优先两层求解；False=加权单目标（对比实验用）
    USE_WEIGHTED = True  # True=加权单目标, False=字典序多目标
    LOG_TO_CONSOLE = 0  # COPT 日志开关（调试 MipStart 时可临时置 1）
    # 加权目标权重（基于历史运行数据标定）
    W_TIME = 1.0       # 总时间权重（~10000量级）
    W_TRANSIT = 0.01   # 在途时间权重（~35000量级）
    W_UTIL = 0.15      # 利用率项权重（λ*seat_km - pass_km ~105000量级）
    W_FUEL = 0.005     # 燃油权重（~90000量级）

    @staticmethod
    def _has_solution(model):
        if model.status in (cp.COPT.OPTIMAL, cp.COPT.TIMEOUT, cp.COPT.NODELIMIT, cp.COPT.INTERRUPTED):
            try:
                _ = model.objval
                return True
            except Exception:
                return False
        return False

    def _apply_solver_tuning(self, model):
        """
        求解器参数调优钩子：在每次 COPT 模型求解前调用。

        基类默认不额外调参（保持原有行为）；子类可覆盖此方法注入
        MIPFocus / Cuts / Heuristics / Presolve 等针对大模型的参数。
        """
        pass

    def solve(self, routes_info, day_demands, day, distances, warm_start=None):
        """
        求解单日MIP，返回(selected_routes_info, yo_vals, yi_vals, ys_vals)或None

        参数:
            warm_start: 可选热启动数据（由 ALNS 改进器生成），格式
                {'selected': {ri_idx}, 'yo': {(ri_idx,f):n},
                 'yi': {(ri_idx,f):n}, 'ys': {(ri_idx,f1,f2):n}}
                仅在第一轮 Dinkelbach 迭代中注入 COPT MipStart。
        """
        print(f"\n===== COPT MIP 求解 (日期={day}, 路径数={len(routes_info)}) =====")

        # 分类当日需求
        airport_set = set(AIRPORTS) | {'LAND'}
        outbound = [d for d in day_demands if d['origin_id'] in airport_set and d['destination_id'].startswith('F')]
        inbound = [d for d in day_demands if d['origin_id'].startswith('F') and d['destination_id'] in airport_set]
        shuttle = [d for d in day_demands if d['origin_id'].startswith('F') and d['destination_id'].startswith('F')]

        # 按设施分组需求
        ob_by_fac = defaultdict(list)
        for d in outbound:
            ob_by_fac[d['destination_id']].append(d)
        ib_by_fac = defaultdict(list)
        for d in inbound:
            ib_by_fac[d['origin_id']].append(d)
        sh_by_od = defaultdict(list)
        for d in shuttle:
            sh_by_od[(d['origin_id'], d['destination_id'])].append(d)

        # 预处理所有路径（避免Dinkelbach迭代中重复计算）
        for ri in routes_info:
            if 'precomputed' not in ri:
                ri['precomputed'] = Q2RoutePreprocessor.preprocess(ri['route'], distances)

        # 自适应渐进搜索（创新点C10）：三轮迭代，每轮用上一轮解热启动
        if self.USE_ADAPTIVE:
            return self._solve_adaptive(routes_info, day, distances,
                                        ob_by_fac, ib_by_fac, sh_by_od,
                                        warm_start=warm_start)

        # 两层模式（默认）：第一层严格最小化总飞机使用时间，第二层在
        # sum(t_r x_r) <= T* 硬约束（无上界差）下优化二级指标
        if self.USE_TWO_STAGE:
            return self._solve_two_stage(routes_info, day, distances,
                                         ob_by_fac, ib_by_fac, sh_by_od,
                                         warm_start=warm_start)

        lam = self.LAMBDA_INIT
        best_result = None
        best_util = -1.0

        for iteration in range(self.DINKELBACH_MAX_ITERS):
            tl = self.TIME_LIMITS[min(iteration, len(self.TIME_LIMITS) - 1)]
            print(f"\n--- Dinkelbach迭代 {iteration+1}/{self.DINKELBACH_MAX_ITERS} (λ={lam:.4f}, TL={tl}s) ---")
            result = self._solve_with_lambda(routes_info, day, distances,
                                             ob_by_fac, ib_by_fac, sh_by_od, lam, tl,
                                             warm_start=warm_start if iteration == 0 else None)
            if result is None:
                print("  无可行解，终止")
                break
            sel_routes, yo_vals, yi_vals, ys_vals, pass_km, seat_km = result
            actual_util = pass_km / seat_km if seat_km > 0 else 0.0
            print(f"  利用率: {actual_util:.4f}, pass_km={pass_km:.0f}, seat_km={seat_km:.0f}")
            if actual_util > best_util:
                best_util = actual_util
                best_result = result
            if abs(actual_util - lam) < self.DINKELBACH_TOL:
                print(f"  Dinkelbach收敛!")
                break
            lam = actual_util

        if best_result is None:
            # 重试：用更长时间和简化模型（仅第一目标）重新求解
            print(f"\n  首次求解失败，重试中 (TL={self.RETRY_TIME_LIMIT}s, 仅min时间目标)...")
            result = self._solve_fallback(routes_info, day, distances,
                                          ob_by_fac, ib_by_fac, sh_by_od)
            if result is None:
                return None
            sel_routes, yo_vals, yi_vals, ys_vals = result
            t_star = sum(ri['route'].time for ri in sel_routes)
            print(f"\n选中路径(重试): {len(sel_routes)}")
            return sel_routes, yo_vals, yi_vals, ys_vals, t_star
        sel_routes, yo_vals, yi_vals, ys_vals, _, _ = best_result
        t_star = sum(ri['route'].time for ri in sel_routes)
        print(f"\n选中路径: {len(sel_routes)}")
        return sel_routes, yo_vals, yi_vals, ys_vals, t_star

    def _solve_two_stage(self, routes_info, day, distances,
                         ob_by_fac, ib_by_fac, sh_by_od, warm_start=None):
        """
        两层求解（严格时间优先）。

        第一层：单目标严格最小化总飞机使用时间，得到 T*；
        第二层：在硬约束 sum(t_r * x_r) <= T*（无任何上界差/容差放松）下，
                优化在途时间、座位利用率与燃油的加权二级目标。

        返回:
            (sel_routes, yo_vals, yi_vals, ys_vals, t_star)
        """
        print("\n===== 两层求解（严格时间优先，无上界差） =====")

        # ---------- 第一层：严格 min T ----------
        t0 = time_module.time()
        result1 = self._solve_with_lambda(routes_info, day, distances,
                                          ob_by_fac, ib_by_fac, sh_by_od,
                                          self.LAMBDA_INIT,
                                          self.TWO_STAGE_TIME_LIMITS[0],
                                          warm_start=warm_start,
                                          objective='time')
        if result1 is None:
            print("  第一层（min T）无可行解，进入单目标重试...")
            result = self._solve_fallback(routes_info, day, distances,
                                          ob_by_fac, ib_by_fac, sh_by_od)
            if result is None:
                return None
            sel_routes, yo_vals, yi_vals, ys_vals = result
            t_star = sum(ri['route'].time for ri in sel_routes)
            return sel_routes, yo_vals, yi_vals, ys_vals, t_star

        sel1, yo1, yi1, ys1, _, _ = result1
        t_star = sum(ri['route'].time for ri in sel1)
        print(f"  第一层完成: T*={t_star} min, 架次={len(sel1)}, "
              f"耗时={time_module.time()-t0:.1f}s")

        # ---------- 第二层：T <= T* 下优化二级指标 ----------
        t1 = time_module.time()
        # 第一层选中路径索引全集（含"x=1 但 y=0"的个人级覆盖路径）
        idx_of = {id(ri): i for i, ri in enumerate(routes_info)}
        sel_idx = {idx_of[id(ri)] for ri in sel1}
        # 注入第一层解作为热启动：保证第二层立即可行，再在 T<=T* 内优化二级目标
        ws_second = {
            'selected': sel_idx,
            'yo': yo1,
            'yi': yi1,
            'ys': ys1,
        }
        result2 = self._solve_with_lambda(routes_info, day, distances,
                                          ob_by_fac, ib_by_fac, sh_by_od,
                                          self.LAMBDA_INIT,
                                          self.TWO_STAGE_TIME_LIMITS[1],
                                          warm_start=ws_second,
                                          objective='second',
                                          time_cap=t_star,
                                          mipstart_mode=1,
                                          fixed_selected=sel_idx)
        if result2 is None:
            print("  第二层无可行解，返回第一层解")
            return sel1, yo1, yi1, ys1, t_star

        sel2, yo2, yi2, ys2, _, _ = result2
        t2_val = sum(ri['route'].time for ri in sel2)
        assert t2_val <= t_star, f"第二层违反时间硬约束: {t2_val} > {t_star}"
        print(f"  第二层完成: T={t2_val} min (≤ T*={t_star}), 架次={len(sel2)}, "
              f"耗时={time_module.time()-t1:.1f}s")
        return sel2, yo2, yi2, ys2, t_star

    # ==================== 自适应渐进搜索配置（创新点C10） ====================
    ADAPTIVE_ROUNDS = [
        (7200, 0.05, '粗搜'),    # 2h找可行解
        (7200, 0.02, '精搜'),    # 2h收紧搜索
        (7200, 0.01, '极精搜'),  # 2h逼近最优
    ]
    USE_ADAPTIVE = True  # True=自适应三轮搜索；False=原两层搜索

    def _solve_adaptive(self, routes_info, day, distances,
                        ob_by_fac, ib_by_fac, sh_by_od, warm_start=None):
        """
        自适应渐进搜索（创新点C10）：三轮迭代，每轮用上一轮的解做热启动。

        策略：粗搜(60s/30%) → 精搜(180s/15%) → 极精搜(600s/5%)
        每轮都是两层结构：第一层严格min T，第二层T<=T*下优化二级指标。
        上一轮的解作为下一轮的MipStart，让求解器从已有可行解出发继续优化。

        返回:
            (sel_routes, yo_vals, yi_vals, ys_vals, t_star) 或 None
        """
        print("\n===== 自适应渐进搜索（三轮迭代） =====")

        best_result = None
        best_t_star = float('inf')
        prev_warm = warm_start  # 初始热启动（可能为None）

        for round_idx, (time_limit, rel_gap, round_name) in enumerate(self.ADAPTIVE_ROUNDS):
            print(f"\n{'='*50}")
            print(f"  第{round_idx+1}轮 [{round_name}]: TL={time_limit}s, RelGap={rel_gap:.0%}")
            print(f"{'='*50}")

            t0 = time_module.time()

            # 第一层：严格 min T（使用当前轮的时限和间隙）
            result1 = self._solve_with_lambda(routes_info, day, distances,
                                              ob_by_fac, ib_by_fac, sh_by_od,
                                              self.LAMBDA_INIT,
                                              time_limit,
                                              warm_start=prev_warm,
                                              objective='time',
                                              rel_gap=rel_gap)
            t_stage1 = time_module.time() - t0

            if result1 is None:
                print(f"  [{round_name}] 第一层无可行解")
                # 如果是第一轮就失败，尝试fallback
                if round_idx == 0:
                    result = self._solve_fallback(routes_info, day, distances,
                                                  ob_by_fac, ib_by_fac, sh_by_od)
                    if result is not None:
                        sel, yo, yi, ys = result
                        t_star = sum(ri['route'].time for ri in sel)
                        best_result = (sel, yo, yi, ys)
                        best_t_star = t_star
                        prev_warm = {
                            'selected': {id(ri) for ri in sel},
                            'yo': yo, 'yi': yi, 'ys': ys
                        }
                continue

            sel1, yo1, yi1, ys1, _, _ = result1
            t_star = sum(ri['route'].time for ri in sel1)
            print(f"  [{round_name}] 第一层完成: T*={t_star}min, 架次={len(sel1)}, "
                  f"耗时={t_stage1:.1f}s")

            # 第二层：T <= T* 下优化二级指标
            t1 = time_module.time()
            idx_of = {id(ri): i for i, ri in enumerate(routes_info)}
            sel_idx = {idx_of[id(ri)] for ri in sel1}
            ws_second = {'selected': sel_idx, 'yo': yo1, 'yi': yi1, 'ys': ys1}

            result2 = self._solve_with_lambda(routes_info, day, distances,
                                              ob_by_fac, ib_by_fac, sh_by_od,
                                              self.LAMBDA_INIT,
                                              time_limit,  # 第二层使用完整时限（2h+）
                                              warm_start=ws_second,
                                              objective='second',
                                              time_cap=t_star,
                                              mipstart_mode=1,
                                              fixed_selected=sel_idx,
                                              rel_gap=rel_gap)
            t_stage2 = time_module.time() - t1

            if result2 is not None:
                sel2, yo2, yi2, ys2, _, _ = result2
                t2_val = sum(ri['route'].time for ri in sel2)
                print(f"  [{round_name}] 第二层完成: T={t2_val}min (≤T*={t_star}), "
                      f"架次={len(sel2)}, 耗时={t_stage2:.1f}s")
                current_result = (sel2, yo2, yi2, ys2)
                current_t_star = t_star
            else:
                print(f"  [{round_name}] 第二层无可行解，使用第一层解")
                current_result = (sel1, yo1, yi1, ys1)
                current_t_star = t_star

            # 更新最优解（T*更小的解更优）
            if current_t_star < best_t_star:
                best_t_star = current_t_star
                best_result = current_result
                print(f"  [{round_name}] ★ 新最优: T*={best_t_star}min")

            # 构建下一轮的热启动
            sel_cur = current_result[0]
            prev_warm = {
                'selected': {idx_of[id(ri)] for ri in sel_cur},
                'yo': current_result[1],
                'yi': current_result[2],
                'ys': current_result[3],
            }

            # 如果T*未改善且已到最后一轮，提前终止
            if round_idx > 0 and current_t_star >= best_t_star:
                print(f"  [{round_name}] T*未改善，提前终止搜索")
                break

        if best_result is None:
            print("\n  自适应搜索全部失败，进入最终fallback...")
            result = self._solve_fallback(routes_info, day, distances,
                                          ob_by_fac, ib_by_fac, sh_by_od)
            if result is None:
                return None
            sel, yo, yi, ys = result
            t_star = sum(ri['route'].time for ri in sel)
            return sel, yo, yi, ys, t_star

        sel, yo, yi, ys = best_result
        print(f"\n  自适应搜索完成: 最优T*={best_t_star}min, 架次={len(sel)}")
        return sel, yo, yi, ys, best_t_star

    def _solve_fallback(self, routes_info, day, distances,
                        ob_by_fac, ib_by_fac, sh_by_od):
        """失败重试：仅优化第一目标（min总时间），更长时限"""
        result = self._solve_with_lambda_single(routes_info, day, distances,
                                                ob_by_fac, ib_by_fac, sh_by_od,
                                                self.RETRY_TIME_LIMIT)
        if result is None:
            return None
        sel_routes, yo_vals, yi_vals, ys_vals, _, _ = result
        return sel_routes, yo_vals, yi_vals, ys_vals

    def _solve_with_lambda_single(self, routes_info, day, distances,
                                   ob_by_fac, ib_by_fac, sh_by_od, time_limit):
        """简化求解：仅第一目标（min总时间），用于失败重试"""
        env = cp.Envr()
        model = env.createModel('Q3_Fallback')
        model.setParam(cp.COPT.Param.TimeLimit, time_limit)
        model.setParam(cp.COPT.Param.RelGap, 0.01)
        model.setParam(cp.COPT.Param.LogToConsole, 0)

        x = {}
        yo, yi, ys = {}, {}, {}
        compat_ob = defaultdict(dict)
        compat_ib = defaultdict(dict)
        compat_sh = defaultdict(dict)
        demand_routes = defaultdict(list)

        for ri_idx, ri in enumerate(routes_info):
            r = ri['route']
            x[ri_idx] = model.addVar(vtype=cp.COPT.BINARY, name=f'x_{ri_idx}')
            pre = ri.get('precomputed')
            if pre is None:
                pre = Q2RoutePreprocessor.preprocess(r, distances)
                ri['precomputed'] = pre
            for d in ri['compatible_demands']:
                demand_routes[d['person_id']].append(ri_idx)
                o, de = d['origin_id'], d['destination_id']
                if o in (set(AIRPORTS) | {'LAND'}) and de.startswith('F'):
                    compat_ob[ri_idx][de] = compat_ob[ri_idx].get(de, 0) + 1
                elif o.startswith('F') and de in (set(AIRPORTS) | {'LAND'}):
                    compat_ib[ri_idx][o] = compat_ib[ri_idx].get(o, 0) + 1
                elif o.startswith('F') and de.startswith('F'):
                    compat_sh[ri_idx][(o, de)] = compat_sh[ri_idx].get((o, de), 0) + 1
            for f in pre['facility_pos']:
                if f in ob_by_fac and compat_ob[ri_idx].get(f, 0) > 0:
                    yo[(ri_idx, f)] = model.addVar(vtype=cp.COPT.INTEGER, lb=0,
                        ub=compat_ob[ri_idx][f], name=f'yo_{ri_idx}_{f}')
                if f in ib_by_fac and compat_ib[ri_idx].get(f, 0) > 0:
                    yi[(ri_idx, f)] = model.addVar(vtype=cp.COPT.INTEGER, lb=0,
                        ub=compat_ib[ri_idx][f], name=f'yi_{ri_idx}_{f}')
            for (f1, f2) in pre['shuttle_info']:
                if (f1, f2) in sh_by_od and compat_sh[ri_idx].get((f1, f2), 0) > 0:
                    ys[(ri_idx, f1, f2)] = model.addVar(vtype=cp.COPT.INTEGER, lb=0,
                        ub=compat_sh[ri_idx][(f1, f2)], name=f'ys_{ri_idx}_{f1}_{f2}')

        # 需求覆盖约束
        for f, dlist in ob_by_fac.items():
            terms = [yo[(ri, f)] for ri in range(len(routes_info)) if (ri, f) in yo]
            if terms:
                model.addConstr(cp.quicksum(terms) == len(dlist))
        for f, dlist in ib_by_fac.items():
            terms = [yi[(ri, f)] for ri in range(len(routes_info)) if (ri, f) in yi]
            if terms:
                model.addConstr(cp.quicksum(terms) == len(dlist))
        for (f1, f2), dlist in sh_by_od.items():
            terms = [ys[(ri, f1, f2)] for ri in range(len(routes_info)) if (ri, f1, f2) in ys]
            if terms:
                model.addConstr(cp.quicksum(terms) == len(dlist))

        # 个人级覆盖约束（修复紧需求漏配）
        for pid, ri_list in demand_routes.items():
            if ri_list:
                model.addConstr(cp.quicksum(x[ri] for ri in ri_list) >= 1)

        # 起点感知约束
        for f, dlist in ob_by_fac.items():
            by_origin = defaultdict(int)
            for d in dlist:
                by_origin[d['origin_id']] += 1
            for origin, cnt in by_origin.items():
                if origin != 'LAND' and cnt > 0:
                    terms = [yo[(ri, f)] for ri in range(len(routes_info))
                             if (ri, f) in yo and routes_info[ri]['route'].airport == origin]
                    if terms:
                        model.addConstr(cp.quicksum(terms) >= cnt)
        for f, dlist in ib_by_fac.items():
            by_dest = defaultdict(int)
            for d in dlist:
                by_dest[d['destination_id']] += 1
            for dest, cnt in by_dest.items():
                if dest != 'LAND' and cnt > 0:
                    terms = [yi[(ri, f)] for ri in range(len(routes_info))
                             if (ri, f) in yi and routes_info[ri]['route'].airport == dest]
                    if terms:
                        model.addConstr(cp.quicksum(terms) >= cnt)

        # 航段容量约束
        for ri_idx, ri in enumerate(routes_info):
            r = ri['route']
            pre = ri['precomputed']
            cap = r.capacity
            for k, (out_after, in_before, shut_cross) in enumerate(pre['segments']):
                terms = []
                for f in out_after:
                    if (ri_idx, f) in yo:
                        terms.append(yo[(ri_idx, f)])
                for f in in_before:
                    if (ri_idx, f) in yi:
                        terms.append(yi[(ri_idx, f)])
                for (f1, f2) in shut_cross:
                    if (ri_idx, f1, f2) in ys:
                        terms.append(ys[(ri_idx, f1, f2)])
                if terms:
                    model.addConstr(cp.quicksum(terms) <= cap * x[ri_idx])

        # 机队约束
        fleet_groups = defaultdict(list)
        for ri_idx, ri in enumerate(routes_info):
            r = ri['route']
            fleet_groups[(r.airport, r.aircraft_type)].append(ri_idx)
        for (airport, atype), ri_indices in fleet_groups.items():
            max_count = FLEET[airport][atype]
            slot_to_routes = defaultdict(list)
            for ri_idx in ri_indices:
                for slot in routes_info[ri_idx]['time_slots']:
                    slot_to_routes[slot].append(ri_idx)
            for slot, ri_list in slot_to_routes.items():
                if len(ri_list) > max_count:
                    model.addConstr(cp.quicksum(x[ri] for ri in ri_list) <= max_count)

        # 仅第一目标
        time_expr = cp.quicksum(ri['route'].time * x[idx] for idx, ri in enumerate(routes_info))
        model.setObjective(time_expr, sense=cp.COPT.MINIMIZE)

        print(f"  重试求解中 (单目标, TL={time_limit}s)...")
        t0 = time_module.time()
        model.solve()
        t1 = time_module.time()

        if not self._has_solution(model):
            print(f"  重试仍失败: 状态={model.status}")
            return None
        print(f"  重试成功, 耗时 {t1-t0:.1f}s")

        try:
            sel_indices = [idx for idx in range(len(routes_info)) if x[idx].x > 0.5]
        except Exception as e:
            print(f"  错误: 无法提取变量值: {e}")
            return None
        sel_routes = [routes_info[idx] for idx in sel_indices]

        yo_vals, yi_vals, ys_vals = {}, {}, {}
        for idx in sel_indices:
            ri = routes_info[idx]
            pre = ri['precomputed']
            for f in pre['facility_pos']:
                if (idx, f) in yo:
                    v = int(round(yo[(idx, f)].x))
                    if v > 0:
                        yo_vals[(idx, f)] = v
                if (idx, f) in yi:
                    v = int(round(yi[(idx, f)].x))
                    if v > 0:
                        yi_vals[(idx, f)] = v
            for (f1, f2) in pre['shuttle_info']:
                if (idx, f1, f2) in ys:
                    v = int(round(ys[(idx, f1, f2)].x))
                    if v > 0:
                        ys_vals[(idx, f1, f2)] = v

        pass_km, seat_km = 0.0, 0.0
        for idx in sel_indices:
            ri = routes_info[idx]
            pre = ri['precomputed']
            seat_km += ri['route'].capacity * ri['route'].total_distance
            for f in pre['facility_pos']:
                pass_km += yo_vals.get((idx, f), 0) * pre['dist_to'].get(f, 0)
                pass_km += yi_vals.get((idx, f), 0) * pre['dist_from'].get(f, 0)
            for (f1, f2) in pre['shuttle_info']:
                pass_km += ys_vals.get((idx, f1, f2), 0) * pre['shuttle_info'][(f1, f2)]['dist']

        return sel_routes, yo_vals, yi_vals, ys_vals, pass_km, seat_km

    def _solve_with_lambda(self, routes_info, day, distances,
                           ob_by_fac, ib_by_fac, sh_by_od, lam, time_limit,
                           warm_start=None, objective='weighted',
                           time_cap=None, mipstart_mode=1,
                           fixed_selected=None, rel_gap=None):
        """
        构建并求解给定λ下的COPT模型。

        参数:
            warm_start: 可选热启动数据 {'selected', 'yo', 'yi', 'ys'}，
                在 model.solve() 前注入 COPT MipStart。
            objective: 'time'（第一层严格min T）| 'second'（第二层T<=time_cap下
                优化二级指标）| 'weighted'（加权单目标）| 'lexicographic'（字典序）
            time_cap: 第二层总飞机使用时间硬上限（无上界差约束），None 表示不限制
            mipstart_mode: COPT MipStartMode 参数（0=禁用初始解，1=仅完整可行解，
                2=可行解+子MIP补全）；第二层注入第一层完整解时使用 1
            fixed_selected: 第二层固定路径集合（x 变量固定为第一层选择），
                使总飞机使用时间严格等于 T*（无上界差），并大幅缩小模型
        """
        env = cp.Envr()
        model = env.createModel('Q3_Daily')
        model.setParam(cp.COPT.Param.TimeLimit, time_limit)
        model.setParam(cp.COPT.Param.RelGap, rel_gap if rel_gap is not None else 0.01)
        model.setParam(cp.COPT.Param.LogToConsole, self.LOG_TO_CONSOLE)
        self._apply_solver_tuning(model)

        # --- 变量 ---
        x = {}
        yo, yi, ys = {}, {}, {}
        # 兼容性缓存
        compat_ob = defaultdict(dict)  # compat_ob[ri_idx][fac] = count
        compat_ib = defaultdict(dict)
        compat_sh = defaultdict(dict)
        demand_routes = defaultdict(list)  # person_id -> 兼容路径索引

        for ri_idx, ri in enumerate(routes_info):
            r = ri['route']
            x[ri_idx] = model.addVar(vtype=cp.COPT.BINARY, name=f'x_{ri_idx}')
            pre = ri.get('precomputed')
            if pre is None:
                pre = Q2RoutePreprocessor.preprocess(r, distances)
                ri['precomputed'] = pre

            # 统计兼容需求
            for d in ri['compatible_demands']:
                demand_routes[d['person_id']].append(ri_idx)
                o, de = d['origin_id'], d['destination_id']
                if o in (set(AIRPORTS) | {'LAND'}) and de.startswith('F'):
                    compat_ob[ri_idx][de] = compat_ob[ri_idx].get(de, 0) + 1
                elif o.startswith('F') and de in (set(AIRPORTS) | {'LAND'}):
                    compat_ib[ri_idx][o] = compat_ib[ri_idx].get(o, 0) + 1
                elif o.startswith('F') and de.startswith('F'):
                    compat_sh[ri_idx][(o, de)] = compat_sh[ri_idx].get((o, de), 0) + 1

            for f in pre['facility_pos']:
                if f in ob_by_fac and compat_ob[ri_idx].get(f, 0) > 0:
                    yo[(ri_idx, f)] = model.addVar(vtype=cp.COPT.INTEGER, lb=0,
                        ub=compat_ob[ri_idx][f], name=f'yo_{ri_idx}_{f}')
                if f in ib_by_fac and compat_ib[ri_idx].get(f, 0) > 0:
                    yi[(ri_idx, f)] = model.addVar(vtype=cp.COPT.INTEGER, lb=0,
                        ub=compat_ib[ri_idx][f], name=f'yi_{ri_idx}_{f}')
            for (f1, f2) in pre['shuttle_info']:
                if (f1, f2) in sh_by_od and compat_sh[ri_idx].get((f1, f2), 0) > 0:
                    ys[(ri_idx, f1, f2)] = model.addVar(vtype=cp.COPT.INTEGER, lb=0,
                        ub=compat_sh[ri_idx][(f1, f2)], name=f'ys_{ri_idx}_{f1}_{f2}')

        n_vars = len(x) + len(yo) + len(yi) + len(ys)
        print(f"  变量: {len(x)} binary + {len(yo)+len(yi)+len(ys)} integer = {n_vars}")

        # --- 约束1: 需求覆盖 ---
        for f, dlist in ob_by_fac.items():
            terms = [yo[(ri, f)] for ri in range(len(routes_info)) if (ri, f) in yo]
            if terms:
                model.addConstr(cp.quicksum(terms) == len(dlist))
            else:
                print(f"  警告: 设施{f}出海无覆盖路径!")
        for f, dlist in ib_by_fac.items():
            terms = [yi[(ri, f)] for ri in range(len(routes_info)) if (ri, f) in yi]
            if terms:
                model.addConstr(cp.quicksum(terms) == len(dlist))
            else:
                print(f"  警告: 设施{f}海返无覆盖路径!")
        for (f1, f2), dlist in sh_by_od.items():
            terms = [ys[(ri, f1, f2)] for ri in range(len(routes_info)) if (ri, f1, f2) in ys]
            if terms:
                model.addConstr(cp.quicksum(terms) == len(dlist))
            else:
                print(f"  警告: 穿梭({f1},{f2})无覆盖路径!")

        # --- 约束1.5: 个人级覆盖（修复紧需求被设施级人数约束掩盖而漏配的问题） ---
        n_person_constr = 0
        for pid, ri_list in demand_routes.items():
            if ri_list:
                model.addConstr(cp.quicksum(x[ri] for ri in ri_list) >= 1)
                n_person_constr += 1
        print(f"  个人级覆盖约束: {n_person_constr}条")

        # --- 约束2: 起点感知 ---
        for f, dlist in ob_by_fac.items():
            by_origin = defaultdict(int)
            for d in dlist:
                by_origin[d['origin_id']] += 1
            for origin, cnt in by_origin.items():
                if origin != 'LAND' and cnt > 0:
                    terms = [yo[(ri, f)] for ri in range(len(routes_info))
                             if (ri, f) in yo and routes_info[ri]['route'].airport == origin]
                    if terms:
                        model.addConstr(cp.quicksum(terms) >= cnt)
        for f, dlist in ib_by_fac.items():
            by_dest = defaultdict(int)
            for d in dlist:
                by_dest[d['destination_id']] += 1
            for dest, cnt in by_dest.items():
                if dest != 'LAND' and cnt > 0:
                    terms = [yi[(ri, f)] for ri in range(len(routes_info))
                             if (ri, f) in yi and routes_info[ri]['route'].airport == dest]
                    if terms:
                        model.addConstr(cp.quicksum(terms) >= cnt)

        # --- 约束3: 航段容量 ---
        n_cap = 0
        for ri_idx, ri in enumerate(routes_info):
            r = ri['route']
            pre = ri['precomputed']
            cap = r.capacity
            for k, (out_after, in_before, shut_cross) in enumerate(pre['segments']):
                terms = []
                for f in out_after:
                    if (ri_idx, f) in yo:
                        terms.append(yo[(ri_idx, f)])
                for f in in_before:
                    if (ri_idx, f) in yi:
                        terms.append(yi[(ri_idx, f)])
                for (f1, f2) in shut_cross:
                    if (ri_idx, f1, f2) in ys:
                        terms.append(ys[(ri_idx, f1, f2)])
                if terms:
                    model.addConstr(cp.quicksum(terms) <= cap * x[ri_idx])
                    n_cap += 1
        print(f"  容量约束: {n_cap}条")

        # --- 约束4: 机队约束（按机型分解，时间扩展网络） ---
        n_fleet = 0
        # 按机场+机型分组路径
        fleet_groups = defaultdict(list)
        for ri_idx, ri in enumerate(routes_info):
            r = ri['route']
            fleet_groups[(r.airport, r.aircraft_type)].append(ri_idx)

        for (airport, atype), ri_indices in fleet_groups.items():
            max_count = FLEET[airport][atype]
            # 收集所有涉及的时间槽
            slot_to_routes = defaultdict(list)
            for ri_idx in ri_indices:
                for slot in routes_info[ri_idx]['time_slots']:
                    slot_to_routes[slot].append(ri_idx)
            for slot, ri_list in slot_to_routes.items():
                if len(ri_list) > max_count:
                    model.addConstr(cp.quicksum(x[ri] for ri in ri_list) <= max_count)
                    n_fleet += 1
        print(f"  机队约束: {n_fleet}条")

        # --- 目标函数 ---
        time_expr = cp.quicksum(ri['route'].time * x[idx] for idx, ri in enumerate(routes_info))

        transit_terms = []
        for idx, ri in enumerate(routes_info):
            pre = ri['precomputed']
            for f in pre['facility_pos']:
                if (idx, f) in yo and f in pre['time_to']:
                    transit_terms.append(pre['time_to'][f] * yo[(idx, f)])
                if (idx, f) in yi and f in pre['time_from']:
                    transit_terms.append(pre['time_from'][f] * yi[(idx, f)])
            for (f1, f2) in pre['shuttle_info']:
                if (idx, f1, f2) in ys:
                    transit_terms.append(pre['shuttle_info'][(f1, f2)]['time'] * ys[(idx, f1, f2)])
        transit_expr = cp.quicksum(transit_terms) if transit_terms else 0

        passkm_terms = []
        for idx, ri in enumerate(routes_info):
            pre = ri['precomputed']
            for f in pre['facility_pos']:
                if (idx, f) in yo and f in pre['dist_to']:
                    passkm_terms.append(pre['dist_to'][f] * yo[(idx, f)])
                if (idx, f) in yi and f in pre['dist_from']:
                    passkm_terms.append(pre['dist_from'][f] * yi[(idx, f)])
            for (f1, f2) in pre['shuttle_info']:
                if (idx, f1, f2) in ys:
                    passkm_terms.append(pre['shuttle_info'][(f1, f2)]['dist'] * ys[(idx, f1, f2)])
        passkm_expr = cp.quicksum(passkm_terms) if passkm_terms else 0
        seatkm_expr = cp.quicksum(ri['route'].capacity * ri['route'].total_distance * x[idx]
                                  for idx, ri in enumerate(routes_info))

        fuel_expr = cp.quicksum(ri['route'].fuel * x[idx] for idx, ri in enumerate(routes_info))

        # --- 约束5: 总飞机使用时间硬上限（保证"严格无上界差"） ---
        if time_cap is not None:
            model.addConstr(time_expr <= time_cap)
            print(f"  时间硬约束: T <= {time_cap} min")

        # --- 固定第一层路径选择（第二层：x 固定，仅优化人员分配） ---
        if fixed_selected is not None:
            n_fixed = 0
            for ri_idx in range(len(routes_info)):
                if ri_idx in fixed_selected:
                    x[ri_idx].setInfo('LB', 1.0)
                    x[ri_idx].setInfo('UB', 1.0)
                else:
                    x[ri_idx].setInfo('LB', 0.0)
                    x[ri_idx].setInfo('UB', 0.0)
                n_fixed += 1
            print(f"  固定路径: {len(fixed_selected)} 条选中 / {n_fixed} 条全部固定")

        if objective == 'time':
            # 第一层：严格最小化总飞机使用时间（不掺入任何二级目标）
            # RelGap 由调用方传入（自适应搜索中由 ADAPTIVE_ROUNDS 控制）
            model.setObjective(time_expr, sense=cp.COPT.MINIMIZE)
            print(f"  目标: 严格最小化总飞机使用时间 (RelGap={rel_gap if rel_gap is not None else 0.01})")
        elif objective == 'second':
            # 第二层：T <= time_cap 硬约束下优化"人员在途时间 + 燃油"加权。
            # 注意不加入利用率项（λ*seat_km - pass_km）：该项使目标矩阵 100%
            # 稠密，根节点 LP 在时限内无法收敛、MipStart 验证被无限推迟；
            # 在途时间与燃油为稀疏目标，能保证第二层立即可行并真正完成优化。
            second_obj = (
                self.W_TRANSIT * transit_expr
                + self.W_FUEL * fuel_expr
            )
            model.setObjective(second_obj, sense=cp.COPT.MINIMIZE)
            print(f"  目标: 二级指标加权 (w_transit={self.W_TRANSIT}, "
                  f"w_fuel={self.W_FUEL}, T≤{time_cap})")
        elif self.USE_WEIGHTED:
            # 加权单目标：同时优化时间、在途、利用率、燃油
            weighted_obj = (
                self.W_TIME * time_expr
                + self.W_TRANSIT * transit_expr
                + self.W_UTIL * (lam * seatkm_expr - passkm_expr)
                + self.W_FUEL * fuel_expr
            )
            model.setObjective(weighted_obj, sense=cp.COPT.MINIMIZE)
            print(f"  目标: 加权单目标 (w_time={self.W_TIME}, w_transit={self.W_TRANSIT}, "
                  f"w_util={self.W_UTIL}, w_fuel={self.W_FUEL}, λ={lam:.4f})")
        else:
            # 字典序四目标
            model.setObjectiveN(0, time_expr, sense=cp.COPT.MINIMIZE, priority=10, reltol=0.01)
            if transit_terms:
                model.setObjectiveN(1, transit_expr, sense=cp.COPT.MINIMIZE,
                                    priority=7, reltol=0.01)
            model.setObjectiveN(2, lam * seatkm_expr - passkm_expr, sense=cp.COPT.MINIMIZE,
                                priority=4, reltol=0.01)
            model.setObjectiveN(3, fuel_expr, sense=cp.COPT.MINIMIZE, priority=1, reltol=0.01)
            print(f"  目标: 字典序四目标 (λ={lam:.4f})")

        # --- 热启动注入（ALNS 提供的初始可行解） ---
        if warm_start is not None:
            start_vars, start_vals = [], []
            selected = warm_start.get('selected', set())
            yo_ws = warm_start.get('yo', {})
            yi_ws = warm_start.get('yi', {})
            ys_ws = warm_start.get('ys', {})
            # 路径选择变量 x
            for ri_idx in range(len(routes_info)):
                start_vars.append(x[ri_idx])
                start_vals.append(1.0 if ri_idx in selected else 0.0)
            # 出海/海返/穿梭人员分配变量（完整提供，缺失补零）
            for key, var in yo.items():
                start_vars.append(var)
                start_vals.append(float(yo_ws.get(key, 0)))
            for key, var in yi.items():
                start_vars.append(var)
                start_vals.append(float(yi_ws.get(key, 0)))
            for key, var in ys.items():
                start_vars.append(var)
                start_vals.append(float(ys_ws.get(key, 0)))
            if start_vars:
                # 显式开启 MipStart，并允许求解器修复轻微不可行的起始解
                model.setParam(cp.COPT.Param.MipStartMode, mipstart_mode)
                if mipstart_mode == 1:
                    model.setParam(cp.COPT.Param.MipRepair, 1)
                model.setMipStart(start_vars, start_vals)
                print(f"  热启动注入: {len(start_vars)} 个变量 "
                      f"(选中路径 {len(selected)} 条, mode={mipstart_mode})")

        # --- 求解 ---
        print("  求解中...")
        t0 = time_module.time()
        model.solve()
        t1 = time_module.time()

        if not self._has_solution(model):
            print(f"  错误: 状态={model.status}, 无可行解")
            return None
        print(f"  求解完成, 耗时 {t1-t0:.1f}s")

        # --- 提取解 ---
        try:
            sel_indices = [idx for idx in range(len(routes_info)) if x[idx].x > 0.5]
        except Exception as e:
            print(f"  错误: 无法提取变量值: {e}")
            return None
        sel_routes = [routes_info[idx] for idx in sel_indices]

        yo_vals, yi_vals, ys_vals = {}, {}, {}
        for idx in sel_indices:
            ri = routes_info[idx]
            pre = ri['precomputed']
            for f in pre['facility_pos']:
                if (idx, f) in yo:
                    v = int(round(yo[(idx, f)].x))
                    if v > 0:
                        yo_vals[(idx, f)] = v
                if (idx, f) in yi:
                    v = int(round(yi[(idx, f)].x))
                    if v > 0:
                        yi_vals[(idx, f)] = v
            for (f1, f2) in pre['shuttle_info']:
                if (idx, f1, f2) in ys:
                    v = int(round(ys[(idx, f1, f2)].x))
                    if v > 0:
                        ys_vals[(idx, f1, f2)] = v

        # 计算pass_km和seat_km
        pass_km, seat_km = 0.0, 0.0
        for idx in sel_indices:
            ri = routes_info[idx]
            pre = ri['precomputed']
            seat_km += ri['route'].capacity * ri['route'].total_distance
            for f in pre['facility_pos']:
                pass_km += yo_vals.get((idx, f), 0) * pre['dist_to'].get(f, 0)
                pass_km += yi_vals.get((idx, f), 0) * pre['dist_from'].get(f, 0)
            for (f1, f2) in pre['shuttle_info']:
                pass_km += ys_vals.get((idx, f1, f2), 0) * pre['shuttle_info'][(f1, f2)]['dist']

        return sel_routes, yo_vals, yi_vals, ys_vals, pass_km, seat_km


# ==================== 飞机分配器 ====================
class Q3AircraftAssigner:
    """将MIP选中的路径分配到具体飞机编号（贪心区间调度）"""

    @staticmethod
    def assign(all_selected_routes):
        """
        为所有选中路径分配具体飞机编号

        参数:
            all_selected_routes: [{route_info, day, yo, yi, ys}, ...]

        返回:
            flights: [{aircraft_id, flight_no, route_info, day, ...}, ...]
        """
        # 按机场+机型分组
        groups = defaultdict(list)
        for idx, item in enumerate(all_selected_routes):
            r = item['route_info']['route']
            groups[(r.airport, r.aircraft_type)].append(item)

        flights = []
        flight_counter = defaultdict(int)  # (airport, type, seq) -> flight_no

        for (airport, atype), items in groups.items():
            # 按起飞时间排序
            items.sort(key=lambda x: x['route_info']['departure_time'])
            max_count = FLEET[airport][atype]
            # 每架飞机的可用时间
            aircraft_available = [None] * max_count  # None表示可用

            for item in items:
                dep = item['route_info']['departure_time']
                ret = item['route_info']['return_time']
                assigned = False
                for seq in range(max_count):
                    if aircraft_available[seq] is None or aircraft_available[seq] <= dep:
                        aircraft_id = f"{airport}-{atype}-H{seq+1:02d}"
                        flight_no = flight_counter[(airport, atype, seq)] + 1
                        flight_counter[(airport, atype, seq)] = flight_no
                        item['aircraft_id'] = aircraft_id
                        item['flight_no'] = flight_no
                        flights.append(item)
                        aircraft_available[seq] = ret + timedelta(minutes=TURNAROUND_MINUTES)
                        assigned = True
                        break
                if not assigned:
                    # 不应发生（MIP已约束），但以防万一
                    print(f"  警告: {airport}-{atype} 无法分配飞机! dep={dep}")

        return flights


# ==================== 阶段二：临时任务背包化（创新点C4） ====================
class Q3Phase2Solver:
    """临时任务背包插入：优先插入已有架次空余座位，次选新增架次"""

    def solve(self, flights, temp_demands, T_star, distances, all_dates):
        """
        在T*约束下尽可能多满足临时任务

        参数:
            flights: 阶段一已分配飞机的架次列表
            temp_demands: 临时任务列表
            T_star: 阶段一总时间
            distances: 距离矩阵
            all_dates: 规划日期列表

        返回:
            (satisfied, unsatisfied) 临时任务分配结果
        """
        print(f"\n===== 阶段二: 临时任务背包化 (T*={T_star}min, 临时={len(temp_demands)}人) =====")

        # 计算每个架次的当前载客量（按航段）
        flight_loads = []
        for f in flights:
            ri = f['route_info']
            r = ri['route']
            schedule = ri['schedule']
            path = r.path
            # 计算各航段已载人数
            seg_load = [0] * (len(path) - 1)
            for (idx, fac), cnt in f.get('yo', {}).items():
                pos = path.index(fac) if fac in path else -1
                if pos > 0:
                    for s in range(pos):
                        seg_load[s] += cnt
            for (idx, fac), cnt in f.get('yi', {}).items():
                pos = path.index(fac) if fac in path else -1
                if pos > 0:
                    for s in range(pos, len(path) - 1):
                        seg_load[s] += cnt
            for (idx, f1, f2), cnt in f.get('ys', {}).items():
                p1 = path.index(f1) if f1 in path else -1
                p2 = path.index(f2) if f2 in path else -1
                if p1 >= 0 and p2 > p1:
                    for s in range(p1, p2):
                        seg_load[s] += cnt
            flight_loads.append(seg_load)

        current_total_time = sum(f['route_info']['route'].time for f in flights)
        remaining_budget = T_star - current_total_time

        satisfied = []
        unsatisfied = []

        # 按时间窗宽度升序排序（最窄先处理）
        sorted_temp = sorted(temp_demands, key=lambda d: d['latest_arrival'] - d['earliest_pickup'])

        for d in sorted_temp:
            inserted = False
            # 尝试插入已有架次
            for fi, f in enumerate(flights):
                ri = f['route_info']
                r = ri['route']
                schedule = ri['schedule']
                path = r.path
                day = f['day']
                cap = r.capacity
                seg_load = flight_loads[fi]

                # 检查兼容性
                compat, pidx, didx = Q3ScheduleHelper.check_compatibility(d, r, schedule, day)
                if not compat:
                    continue

                # 检查各航段是否有空余座位
                has_space = True
                for s in range(pidx, didx):
                    if seg_load[s] + 1 > cap:
                        has_space = False
                        break
                if not has_space:
                    continue

                # 插入
                for s in range(pidx, didx):
                    seg_load[s] += 1
                d['_flight_idx'] = fi
                d['_pickup_idx'] = pidx
                d['_delivery_idx'] = didx
                satisfied.append(d)
                inserted = True
                break

            if not inserted:
                # 尝试新增架次（在T*预算内）
                o = d['origin_id']
                de = d['destination_id']
                # 确定OD设施
                o_fac = None
                de_fac = None
                if o.startswith('F'):
                    o_fac = o
                if de.startswith('F'):
                    de_fac = de
                # 确定有效机场：指定机场的需求只能从该机场出发
                if o in AIRPORTS:
                    valid_airports = [o]
                elif de in AIRPORTS:
                    valid_airports = [de]
                else:
                    valid_airports = AIRPORTS

                # 尝试为该临时任务创建新架次
                best_new_time = float('inf')
                best_new_route = None
                best_new_airport = None

                for airport in valid_airports:
                    facs = []
                    if o_fac and de_fac:
                        if o_fac != de_fac:
                            facs = [o_fac, de_fac]
                        else:
                            facs = [o_fac]
                    elif o_fac:
                        facs = [o_fac]
                    elif de_fac:
                        facs = [de_fac]
                    else:
                        continue

                    if not facs:
                        continue

                    for atype in AIRCRAFT_TYPES:
                        best_order, _ = TSPSolver.best_order(airport, facs, distances)
                        r = make_route(999999, airport, atype, best_order, distances)
                        if r is None:
                            continue
                        if r.time > remaining_budget:
                            continue
                        latest_offset = min(
                            (RETURN_DEADLINE_HOUR - OPERATION_START_HOUR) * 60 - r.time,
                            (DEPARTURE_DEADLINE_HOUR - OPERATION_START_HOUR) * 60)
                        # 尝试所有可行日期
                        feasible_days = Q3DemandDayAllocator.compute_feasible_days(d, all_dates)
                        for day in feasible_days:
                            for offset in DEPARTURE_OFFSETS:
                                if offset > latest_offset:
                                    continue
                                dep_time = datetime(day.year, day.month, day.day,
                                                    OPERATION_START_HOUR, 0) + timedelta(minutes=offset)
                                sched = Q3ScheduleHelper.compute_schedule(r, dep_time, distances)
                                ret_t = sched[len(r.path)-1][0]
                                day_end = datetime(day.year, day.month, day.day, RETURN_DEADLINE_HOUR, 0)
                                if ret_t > day_end:
                                    continue
                                compat, _, _ = Q3ScheduleHelper.check_compatibility(d, r, sched, day)
                                if compat and r.time < best_new_time:
                                    best_new_time = r.time
                                    best_new_route = r
                                    best_new_airport = airport
                                    best_new_schedule = sched
                                    best_new_day = day
                                    best_new_dep = dep_time
                                    best_new_ret = ret_t

                if best_new_route is not None and best_new_time <= remaining_budget:
                    # 创建新架次
                    new_item = {
                        'route_info': {
                            'route': best_new_route,
                            'schedule': best_new_schedule,
                            'departure_time': best_new_dep,
                            'return_time': best_new_ret,
                        },
                        'day': best_new_day,
                        'yo': {}, 'yi': {}, 'ys': {},
                        'aircraft_id': None,  # 后续分配
                        'flight_no': None,
                    }
                    flights.append(new_item)
                    flight_loads.append([0] * (len(best_new_route.path) - 1))
                    remaining_budget -= best_new_time

                    d['_flight_idx'] = len(flights) - 1
                    d['_pickup_idx'] = 0 if not d['origin_id'].startswith('F') else best_new_route.path.index(d['origin_id'])
                    d['_delivery_idx'] = len(best_new_route.path) - 1 if not d['destination_id'].startswith('F') else best_new_route.path.index(d['destination_id'])
                    satisfied.append(d)
                    inserted = True

            if not inserted:
                unsatisfied.append(d)

        print(f"  临时任务满足: {len(satisfied)}/{len(temp_demands)}")
        print(f"  未满足: {len(unsatisfied)}")
        print(f"  剩余时间预算: {remaining_budget:.0f}min")
        return satisfied, unsatisfied


# ==================== Q3输出器 ====================
class Q3OutputWriter:
    """Q3格式输出：q3-routes.csv（含时刻）+ q3-assignments.csv"""

    @staticmethod
    def write_routes(flights, output_dir):
        """写入q3-routes.csv"""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'q3-routes.csv')
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['aircraft_id', 'flight_no', 'stop_order', 'facility_id',
                           'arrival_time', 'departure_time', 'refuel'])
            for flight in flights:
                ri = flight['route_info']
                r = ri['route']
                schedule = ri['schedule']
                path_list = r.path
                ac_id = flight.get('aircraft_id') or 'UNASSIGNED'
                fn = flight.get('flight_no') or 0
                for i, loc in enumerate(path_list):
                    arr, dep = schedule[i]
                    refuel = 1 if i in r.refuel_at else 0
                    arr_str = arr.strftime('%Y-%m-%d %H:%M') if arr else ''
                    dep_str = dep.strftime('%Y-%m-%d %H:%M') if dep else ''
                    writer.writerow([ac_id, fn, i, loc, arr_str, dep_str, refuel])
        print(f"  写入 {path}")

    @staticmethod
    def write_assignments(flights, all_demands, temp_satisfied, temp_unsatisfied, output_dir):
        """写入q3-assignments.csv（含航段级容量保护）"""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'q3-assignments.csv')

        # 构建人员→架次映射
        person_to_flight = {}

        # 初始化每个架次的航段剩余容量，并登记临时任务（已在phase2验证过容量）
        flight_seg_remain = []
        for fi, flight in enumerate(flights):
            r = flight['route_info']['route']
            n_seg = max(0, len(r.path) - 1)
            seg_remain = [r.capacity] * n_seg if n_seg > 0 else []
            flight_seg_remain.append(seg_remain)

        # 辅助：检查区间是否都有座位 / 占用座位
        def seg_has_space(fi, pidx, didx):
            sr = flight_seg_remain[fi]
            if not sr or pidx >= didx:
                return False
            for k in range(pidx, didx):
                if k >= len(sr) or sr[k] <= 0:
                    return False
            return True

        def seg_consume(fi, pidx, didx):
            sr = flight_seg_remain[fi]
            for k in range(pidx, didx):
                sr[k] -= 1

        # 临时任务（phase2 已通过 flight_loads 检查）
        for d in temp_satisfied:
            fi = d.get('_flight_idx')
            if fi is None or fi < 0 or fi >= len(flights):
                continue
            flight = flights[fi]
            pidx = d['_pickup_idx']
            didx = d['_delivery_idx']
            person_to_flight[d['person_id']] = (
                flight['aircraft_id'], flight['flight_no'], pidx, didx)
            # 占用对应航段座位
            seg_consume(fi, pidx, didx)

        # 构建非临时人员的分配：按日期+OD分组，避免全量扫描
        temp_ids = {d['person_id'] for d in temp_satisfied + temp_unsatisfied}
        non_temp = [d for d in all_demands if d['person_id'] not in temp_ids]

        # 按日期分组非临时需求，再按OD类型细分
        day_ob = defaultdict(lambda: defaultdict(list))  # day -> dest_fac -> [demand]
        day_ib = defaultdict(lambda: defaultdict(list))  # day -> origin_fac -> [demand]
        day_sh = defaultdict(lambda: defaultdict(list))  # day -> (f1,f2) -> [demand]
        for d in non_temp:
            ad = d.get('_assigned_day')
            if ad is None:
                continue
            o, de = d['origin_id'], d['destination_id']
            if o in (set(AIRPORTS) | {'LAND'}) and de.startswith('F'):
                day_ob[ad][de].append(d)
            elif o.startswith('F') and de in (set(AIRPORTS) | {'LAND'}):
                day_ib[ad][o].append(d)
            elif o.startswith('F') and de.startswith('F'):
                day_sh[ad][(o, de)].append(d)

        assigned_persons = set()

        def process_one_flight(fi, day_map, feasible_day_check=None):
            """
            处理一个架次的 OB/IB/SH 具体人员分配：
            - day_map: {OB: day_ob[day], IB: day_ib[day], SH: day_sh[day]}
                     或 unmatched_by_od
            - feasible_day_check: 若不为 None，则 callable(demand) -> bool，必须返回 True 才尝试
            """
            flight = flights[fi]
            ri = flight['route_info']
            r = ri['route']
            schedule = ri['schedule']
            day = flight['day']
            ac_id = flight.get('aircraft_id') or 'UNASSIGNED'
            fn = flight.get('flight_no') or 0
            ob_map = day_map['ob']
            ib_map = day_map['ib']
            sh_map = day_map['sh']

            def try_assign(typ, key, cnt):
                """typ='ob'|'ib'|'sh', key=fac|(f1,f2), cnt=该类配额上限，返回实际分配数"""
                if cnt <= 0:
                    return 0
                if typ == 'ob':
                    cands_src = ob_map.get(key, [])
                elif typ == 'ib':
                    cands_src = ib_map.get(key, [])
                else:
                    cands_src = sh_map.get(key, [])
                compatible = []
                for d in cands_src:
                    if d['person_id'] in assigned_persons:
                        continue
                    # OD 端点校验
                    if typ == 'ob':
                        o = d['origin_id']
                        if o != 'LAND' and o != r.airport:
                            continue
                    elif typ == 'ib':
                        de = d['destination_id']
                        if de != 'LAND' and de != r.airport:
                            continue
                    if feasible_day_check is not None and not feasible_day_check(d):
                        continue
                    compat, pidx, didx = Q3ScheduleHelper.check_compatibility(
                        d, r, schedule, day)
                    if compat and seg_has_space(fi, pidx, didx):
                        compatible.append((d, pidx, didx))
                compatible.sort(key=lambda x: x[0]['latest_arrival'])
                added = 0
                for d, pidx, didx in compatible:
                    if added >= cnt:
                        break
                    # 再次检查（顺序分配可能导致之前的名额被占用）
                    if not seg_has_space(fi, pidx, didx):
                        continue
                    person_to_flight[d['person_id']] = (ac_id, fn, pidx, didx)
                    assigned_persons.add(d['person_id'])
                    seg_consume(fi, pidx, didx)
                    added += 1
                return added

            # OB
            for (_, fac), cnt in flight.get('yo', {}).items():
                try_assign('ob', fac, cnt)
            # IB
            for (_, fac), cnt in flight.get('yi', {}).items():
                try_assign('ib', fac, cnt)
            # SH
            for (_, f1, f2), cnt in flight.get('ys', {}).items():
                try_assign('sh', (f1, f2), cnt)

        # 第一 pass：按日匹配（_assigned_day）
        for fi in range(len(flights)):
            day = flights[fi]['day']
            dm = {'ob': day_ob.get(day, {}),
                  'ib': day_ib.get(day, {}),
                  'sh': day_sh.get(day, {})}
            process_one_flight(fi, dm, feasible_day_check=None)

        # 第二 pass：对未匹配的需求进行跨可行日的严格匹配
        unmatched_demands = []
        for d in non_temp:
            if d['person_id'] not in assigned_persons:
                ad = d.get('_assigned_day')
                if ad is None:
                    continue
                unmatched_demands.append(d)

        if unmatched_demands:
            n_before = len(unmatched_demands)
            print(f"  二次匹配（严格约束+容量）: {n_before}人待匹配")
            unmatched_by_od = {
                'ob': defaultdict(list),
                'ib': defaultdict(list),
                'sh': defaultdict(list),
            }
            for d in unmatched_demands:
                o, de = d['origin_id'], d['destination_id']
                if o in (set(AIRPORTS) | {'LAND'}) and de.startswith('F'):
                    unmatched_by_od['ob'][de].append(d)
                elif o.startswith('F') and de in (set(AIRPORTS) | {'LAND'}):
                    unmatched_by_od['ib'][o].append(d)
                elif o.startswith('F') and de.startswith('F'):
                    unmatched_by_od['sh'][(o, de)].append(d)

            feasible_cache = {}
            def get_feasible_days(demand):
                pid = demand['person_id']
                if pid not in feasible_cache:
                    days = []
                    for pd in PLANNING_DATES:
                        ds = datetime(pd.year, pd.month, pd.day, OPERATION_START_HOUR, 0)
                        de_t = datetime(pd.year, pd.month, pd.day, RETURN_DEADLINE_HOUR, 0)
                        if demand['earliest_pickup'] < de_t and demand['latest_arrival'] > ds:
                            days.append(pd)
                    feasible_cache[pid] = days
                return feasible_cache[pid]

            for fi in range(len(flights)):
                flight = flights[fi]
                day = flight['day']

                def check_day(d):
                    return day in get_feasible_days(d)

                process_one_flight(fi, unmatched_by_od, feasible_day_check=check_day)

            n_rematched = n_before - len(
                [d for d in unmatched_demands if d['person_id'] not in assigned_persons])
            print(f"  二次匹配完成: 成功匹配 {n_rematched}/{n_before}人，"
                  f"未服务 {n_before - n_rematched}人")

        # 写入
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['person_id', 'aircraft_id', 'flight_no',
                           'pickup_stop_order', 'delivery_stop_order'])
            for d in all_demands:
                pid = d['person_id']
                if pid in person_to_flight:
                    ac, fn, pidx, didx = person_to_flight[pid]
                    writer.writerow([pid, ac, fn, pidx, didx])
                else:
                    writer.writerow([pid, '', '', '', ''])
        print(f"  写入 {path}")
        print(f"  已分配: {len(person_to_flight)}/{len(all_demands)}")


    @staticmethod
    def write_daily_output(day, day_flights, day_demands, distances, output_dir):
        """
        逐日即时输出：每天MIP求解完立刻写入两个CSV。
        
        参数:
            day: 当天日期
            day_flights: 当天已分配飞机编号的架次列表
            day_demands: 当天的非临时需求列表
            distances: 距离矩阵
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        date_str = day.strftime('%Y%m%d')
        
        # --- 写 q3-routes-{date}.csv ---
        routes_path = os.path.join(output_dir, f'q3-routes-{date_str}.csv')
        with open(routes_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['aircraft_id', 'flight_no', 'stop_order', 'facility_id',
                             'arrival_time', 'departure_time', 'refuel'])
            for flight in day_flights:
                ri = flight['route_info']
                r = ri['route']
                schedule = ri['schedule']
                path_list = r.path
                ac_id = flight.get('aircraft_id') or 'UNASSIGNED'
                fn = flight.get('flight_no') or 0
                for i, loc in enumerate(path_list):
                    arr, dep = schedule[i]
                    refuel = 1 if i in r.refuel_at else 0
                    arr_str = arr.strftime('%Y-%m-%d %H:%M') if arr else ''
                    dep_str = dep.strftime('%Y-%m-%d %H:%M') if dep else ''
                    writer.writerow([ac_id, fn, i, loc, arr_str, dep_str, refuel])
        print(f"  写入 {routes_path} ({len(day_flights)} 架次)")
        
        # --- 写 q3-assignments-{date}.csv ---
        assign_path = os.path.join(output_dir, f'q3-assignments-{date_str}.csv')
        
        # 构建人员→架次映射（从yo/yi/ys值分配具体人员）
        person_to_flight = {}
        
        # 按OD类型分组当天需求
        airport_set = set(AIRPORTS) | {'LAND'}
        ob_by_fac = defaultdict(list)   # dest_fac -> [demand]
        ib_by_fac = defaultdict(list)   # origin_fac -> [demand]
        sh_by_od = defaultdict(list)    # (f1, f2) -> [demand]
        
        for d in day_demands:
            o, de = d['origin_id'], d['destination_id']
            if o in airport_set and de.startswith('F'):
                ob_by_fac[de].append(d)
            elif o.startswith('F') and de in airport_set:
                ib_by_fac[o].append(d)
            elif o.startswith('F') and de.startswith('F'):
                sh_by_od[(o, de)].append(d)
        
        # 初始化每架次航段剩余容量
        flight_seg_remain = []
        for fi, flight in enumerate(day_flights):
            r = flight['route_info']['route']
            n_seg = max(0, len(r.path) - 1)
            seg_remain = [r.capacity] * n_seg if n_seg > 0 else []
            flight_seg_remain.append(seg_remain)
        
        def seg_has_space(fi, pidx, didx):
            sr = flight_seg_remain[fi]
            if not sr or pidx >= didx:
                return False
            for k in range(pidx, didx):
                if k >= len(sr) or sr[k] <= 0:
                    return False
            return True
        
        def seg_consume(fi, pidx, didx):
            sr = flight_seg_remain[fi]
            for k in range(pidx, didx):
                sr[k] -= 1
        
        # 遍历架次，按yo/yi/ys分配人员
        for fi, flight in enumerate(day_flights):
            ri = flight['route_info']
            r = ri['route']
            schedule = ri['schedule']
            ac_id = flight.get('aircraft_id') or 'UNASSIGNED'
            fn = flight.get('flight_no') or 0
            
            def try_assign(od_type, fac_key, count):
                """尝试分配count个人到该架次的fac_key"""
                nonlocal person_to_flight
                if od_type == 'ob':
                    demands = ob_by_fac.get(fac_key, [])
                elif od_type == 'ib':
                    demands = ib_by_fac.get(fac_key, [])
                else:  # sh
                    demands = sh_by_od.get(fac_key, [])
                
                added = 0
                for d in demands:
                    if d['person_id'] in person_to_flight:
                        continue
                    # 找pickup和delivery位置
                    if od_type == 'ob':
                        pidx = 0  # 从机场出发
                        didx = r.path.index(fac_key) if fac_key in r.path else -1
                    elif od_type == 'ib':
                        pidx = r.path.index(fac_key) if fac_key in r.path else -1
                        didx = len(r.path) - 1  # 到机场
                    else:  # sh
                        f1, f2 = fac_key
                        pidx = r.path.index(f1) if f1 in r.path else -1
                        didx = r.path.index(f2) if f2 in r.path else -1
                    
                    if pidx < 0 or didx < 0 or pidx >= didx:
                        continue
                    if not seg_has_space(fi, pidx, didx):
                        continue
                    
                    person_to_flight[d['person_id']] = (ac_id, fn, pidx, didx)
                    seg_consume(fi, pidx, didx)
                    added += 1
                    if added >= count:
                        break
                return added
            
            # OB
            for (_, fac), cnt in flight.get('yo', {}).items():
                try_assign('ob', fac, cnt)
            # IB
            for (_, fac), cnt in flight.get('yi', {}).items():
                try_assign('ib', fac, cnt)
            # SH
            for (_, f1, f2), cnt in flight.get('ys', {}).items():
                try_assign('sh', (f1, f2), cnt)
        
        # 写入
        with open(assign_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['person_id', 'aircraft_id', 'flight_no',
                             'pickup_stop_order', 'delivery_stop_order'])
            for d in day_demands:
                pid = d['person_id']
                if pid in person_to_flight:
                    ac, fn_p, pidx, didx = person_to_flight[pid]
                    writer.writerow([pid, ac, fn_p, pidx, didx])
                else:
                    writer.writerow([pid, '', '', '', ''])
        
        assigned_count = len(person_to_flight)
        total_count = len(day_demands)
        print(f"  写入 {assign_path} ({assigned_count}/{total_count} 人已分配)")
        return assigned_count

# ==================== 主函数 ====================
def main():
    """主函数：编排两阶段求解流程"""
    print("=" * 60)
    print("问题3: 带时间要求的多日排班")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1] 加载数据...")
    loader = Q3DataLoader()
    distances = loader.load_distances()
    all_demands = loader.load_demands_q3()
    print(f"  总需求: {len(all_demands)}")

    # 分离临时和非临时需求
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

    # 3. 逐日求解
    print("\n[3] 逐日求解...")
    all_selected_routes = []
    total_time = 0
    t_star_sum = 0  # 第一层严格最优总时间之和（阶段二/三的时间预算）
    total_fuel = 0.0
    total_pass_km = 0.0
    total_seat_km = 0.0
    total_transit = 0
    solved_days = set()   # MIP 求解成功的日期集合
    failed_day_demands = defaultdict(list)  # 失败日 -> 该日原需求

    route_id_counter = [0]
    generator = Q3RouteGenerator(distances)
    solver = Q3DailySolver()

    for day in PLANNING_DATES:
        day_demands = [d for d in non_temp if allocation[d['person_id']] == day]
        if not day_demands:
            continue
        print(f"\n--- {day} ({len(day_demands)}人) ---")

        # 生成路径
        print("  生成路径...")
        routes_info = generator.generate_daily_routes(day_demands, day, route_id_counter[0])
        route_id_counter[0] = max(r['route'].route_id for r in routes_info) if routes_info else route_id_counter[0]
        print(f"  候选路径: {len(routes_info)}")

        if not routes_info:
            print(f"  警告: {day} 无候选路径! 该日 {len(day_demands)} 人将重新分配到其他已求解日")
            failed_day_demands[day] = day_demands
            continue

        # 求解MIP
        result = solver.solve(routes_info, day_demands, day, distances)
        if result is None:
            print(f"  警告: {day} 无可行解! 该日 {len(day_demands)} 人将重新分配到其他已求解日")
            failed_day_demands[day] = day_demands
            continue

        solved_days.add(day)
        sel_routes, yo_vals, yi_vals, ys_vals, t_star_day = result
        t_star_sum += t_star_day
        for idx, ri in enumerate(sel_routes):
            ri['route_id'] = ri['route'].route_id

        # 索引映射
        idx_map = {id(ri): idx for idx, ri in enumerate(routes_info)}
        for ri in sel_routes:
            orig_idx = idx_map[id(ri)]
            ri['_orig_idx'] = orig_idx

        # 重新构建yo/yi/ys键
        for ri in sel_routes:
            oi = ri['_orig_idx']
            ri['yo'] = {(ri_idx, f): v for (ri_idx, f), v in yo_vals.items() if ri_idx == oi}
            ri['yi'] = {(ri_idx, f): v for (ri_idx, f), v in yi_vals.items() if ri_idx == oi}
            ri['ys'] = {(ri_idx, f1, f2): v for (ri_idx, f1, f2), v in ys_vals.items() if ri_idx == oi}

        # 统计
        for ri in sel_routes:
            r = ri['route']
            pre = ri.get('precomputed', {})
            total_time += r.time
            total_fuel += r.fuel
            total_seat_km += r.capacity * r.total_distance
            # 计算客公里和人员在途时间
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
                                        'yo': ri['yo'], 'yi': ri['yi'], 'ys': ri['ys']})

        n_flights = len(sel_routes)
        n_people = sum(sum(ri['yo'].values()) + sum(ri['yi'].values()) + sum(ri['ys'].values())
                       for ri in sel_routes)
        print(f"  架次: {n_flights}, 人数: {n_people}, 时间: {sum(ri['route'].time for ri in sel_routes)}min")

        # ===== 逐日即时输出：分配飞机编号 + 写CSV =====
        # 为当天架次分配飞机编号
        day_routes_for_assign = [
            {'route_info': ri, 'day': day, 'yo': ri['yo'], 'yi': ri['yi'], 'ys': ri['ys']}
            for ri in sel_routes
        ]
        day_flights = Q3AircraftAssigner.assign(day_routes_for_assign)
        # 写入当天CSV（仅非临时需求，临时任务在阶段二处理）
        Q3OutputWriter.write_daily_output(day, day_flights, day_demands, distances, OUTPUT_DIR_Q3)

    # 3.1 失败日需求重分配：转移到其他已成功求解的日期
    # （更新需求的 _assigned_day，确保 write_assignments 能匹配到可用航班日）
    if failed_day_demands:
        print(f"\n[3.1] 失败日需求重分配: {len(failed_day_demands)} 个日期需要处理")
        # 统计各已求解日当前的"负载"（已服务人数上限 = MIP解中的y之和）
        day_load = defaultdict(int)
        for sr in all_selected_routes:
            day = sr['day']
            n = (sum(sr['yo'].values()) + sum(sr['yi'].values()) + sum(sr['ys'].values()))
            day_load[day] += n

        for failed_day, demands in failed_day_demands.items():
            reallocated_total = 0  # 每失败日单独统计，避免跨日累计污染日志
            for d in demands:
                feasible_in_solved = [pd for pd in solved_days
                                      if pd in Q3DemandDayAllocator.compute_feasible_days(d, PLANNING_DATES)]
                if not feasible_in_solved:
                    # 无可行日：保留 _assigned_day，write_assignments 中二次匹配将严格过滤
                    reallocated_total += 0
                    continue
                # 选择负载最轻的已求解日
                best_day = min(feasible_in_solved, key=lambda pd: day_load[pd])
                d['_assigned_day'] = best_day
                day_load[best_day] += 1
                reallocated_total += 1
            print(f"  {failed_day}: {len(demands)}人 -> 成功重分配 {reallocated_total}人 "
                  f"(剩余无可行日 {len(demands) - reallocated_total}人)")
        print(f"  重分配完成: {sum(day_load.values())} 人分布于 {sorted([str(d) for d in solved_days])}")

    T_star = t_star_sum
    print(f"\n阶段一总飞机使用时间 T* = {T_star} 分钟 ({T_star/60:.1f}h)")
    print(f"总架次: {len(all_selected_routes)}, 总燃油: {total_fuel:.1f}kg")

    # 4. 飞机分配
    print("\n[4] 飞机编号分配...")
    flights = Q3AircraftAssigner.assign(all_selected_routes)
    print(f"  分配飞机: {len(set(f['aircraft_id'] for f in flights))}架")

    # 5. 阶段二：临时任务
    print("\n[5] 阶段二: 临时任务背包化...")
    phase2 = Q3Phase2Solver()
    temp_satisfied, temp_unsatisfied = phase2.solve(
        flights, temp_demands, T_star, distances, PLANNING_DATES)

    # 为阶段二新增的架次分配飞机编号
    new_flights = [f for f in flights if f.get('aircraft_id') is None]
    if new_flights:
        # 重新分配
        all_for_assign = all_selected_routes + [
            {'route_info': f['route_info'], 'day': f['day'],
             'yo': f.get('yo', {}), 'yi': f.get('yi', {}), 'ys': f.get('ys', {})}
            for f in new_flights
        ]
        # 简化：直接为新增架次分配
        for f in new_flights:
            r = f['route_info']['route']
            # 找可用飞机
            for seq in range(FLEET[r.airport][r.aircraft_type]):
                aid = f"{r.airport}-{r.aircraft_type}-H{seq+1:02d}"
                # 检查是否可用
                used = [fl for fl in flights if fl.get('aircraft_id') == aid]
                conflict = False
                for u in used:
                    if not (f['route_info']['return_time'] + timedelta(minutes=TURNAROUND_MINUTES) <= u['route_info']['departure_time']
                            or u['route_info']['return_time'] + timedelta(minutes=TURNAROUND_MINUTES) <= f['route_info']['departure_time']):
                        conflict = True
                        break
                if not conflict:
                    f['aircraft_id'] = aid
                    # 找最大flight_no
                    max_fn = max([fl['flight_no'] for fl in flights
                                  if fl.get('aircraft_id') == aid], default=0)
                    f['flight_no'] = max_fn + 1
                    break

    # 6. 输出
    print("\n[6] 输出结果...")
    Q3OutputWriter.write_routes(flights, OUTPUT_DIR_Q3)
    Q3OutputWriter.write_assignments(flights, all_demands, temp_satisfied, temp_unsatisfied, OUTPUT_DIR_Q3)

    # 7. 指标汇总（分阶段汇报）
    print("\n" + "=" * 60)
    print("问题3结果汇总（分阶段）")
    print("=" * 60)
    # 重新计算总计：时间/燃油/座公里从所有flights重新计算
    actual_time = sum(f['route_info']['route'].time for f in flights)
    actual_fuel = sum(f['route_info']['route'].fuel for f in flights)
    actual_seat_km = sum(f['route_info']['route'].capacity * f['route_info']['route'].total_distance
                         for f in flights)
    # 临时任务客公里和在途时间
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
    total_flights = len(flights)
    phase1_n_flights = len(all_selected_routes)
    phase2_new_flights = total_flights - phase1_n_flights
    phase2_new_time = actual_time - T_star
    phase2_new_fuel = actual_fuel - total_fuel
    phase1_seat_util = total_pass_km / total_seat_km if total_seat_km > 0 else 0
    seat_util = actual_pass_km / actual_seat_km if actual_seat_km > 0 else 0
    total_transit_all = total_transit + temp_transit
    temp_satisfied_count = len(temp_satisfied)
    temp_total = len(temp_demands)

    metrics = {}
    # 阶段一指标
    metrics['--- 阶段一：固定排班（不含临时任务） ---'] = ''
    metrics['  总飞机使用时间 T*'] = f"{T_star} 分钟 ({T_star/60:.2f}h)"
    metrics['  总架次数'] = str(phase1_n_flights)
    metrics['  总燃油消耗量'] = f"{total_fuel:.1f} kg"
    metrics['  座位利用率'] = f"{phase1_seat_util*100:.2f}%"
    metrics['  人员在途时间'] = f"{total_transit} 分钟 ({total_transit/60:.2f}h)"
    # 阶段二指标
    metrics['--- 阶段二：临时任务背包化 ---'] = ''
    metrics['  新增飞机使用时间'] = f"{phase2_new_time} 分钟 ({phase2_new_time/60:.2f}h)"
    metrics['  新增架次数'] = str(phase2_new_flights)
    metrics['  新增燃油消耗量'] = f"{phase2_new_fuel:.1f} kg"
    metrics['  临时任务在途时间'] = f"{temp_transit} 分钟 ({temp_transit/60:.2f}h)"
    metrics['  临时任务客公里'] = f"{temp_pass_km:.1f}"
    metrics['  临时任务满足'] = f"{temp_satisfied_count}/{temp_total} ({temp_satisfied_count/temp_total*100:.2f}%)"
    # 合计指标
    metrics['--- 合计 ---'] = ''
    metrics['  总飞机使用时间'] = f"{actual_time} 分钟 ({actual_time/60:.2f}h)"
    metrics['  总架次数'] = str(total_flights)
    metrics['  总燃油消耗量'] = f"{actual_fuel:.1f} kg"
    metrics['  座位利用率'] = f"{seat_util*100:.2f}%"
    metrics['  人员在途时间'] = f"{total_transit_all} 分钟 ({total_transit_all/60:.2f}h)"

    for k, v in metrics.items():
        if v:
            print(f"  {k}: {v}")
        else:
            print(f"\n{k}")

    # 写指标文件
    metrics_path = os.path.join(OUTPUT_DIR_Q3, 'q3-metrics.txt')
    os.makedirs(OUTPUT_DIR_Q3, exist_ok=True)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        for k, v in metrics.items():
            if v:
                f.write(f"{k}: {v}\n")
            else:
                f.write(f"{k}\n")
    print(f"\n指标已写入 {metrics_path}")


if __name__ == '__main__':
    main()







