# -*- coding: utf-8 -*-
"""
文件: weighted_solver.py
功能: 问题1和问题2的加权单目标优化求解器（替代字典序多目标的额外方法）
方法: 将四个目标通过归一化加权转化为单一线性目标，一次求解无需Dinkelbach迭代
创新点:
  1. 加权归一化单目标：将time/transit/wasted_seatkm/fuel四目标归一化后加权求和
  2. 自适应归一化：先等权重求解获取参考值，再用参考值归一化进行正式实验
  3. 权重敏感性分析：系统测试8组权重方案，研究不同权重比例对解的影响
项目: 2026策联杯 B题
"""

import csv
import os
import sys
import math
import time as time_module
from collections import defaultdict

import coptpy as cp

# ==================== 模块路径 ====================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VIBECODING_DIR = os.path.dirname(_THIS_DIR)
if _VIBECODING_DIR not in sys.path:
    sys.path.insert(0, _VIBECODING_DIR)
# 同目录导入 problem1_solution/problem2_solution
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from config import (
    AIRCRAFT, AIRCRAFT_TYPES, AIRPORTS, REFUEL_FACILITIES,
    MAX_LANDINGS, OUTPUT_DIR, OUTPUT_DIR_Q2, DATA_DIR, fmt6,
    setup_matplotlib, FIGURES_DIR, FIGURES_DIR_Q2
)
from base_models import (
    DataLoader, FuelChecker, TSPSolver, Route,
    calculate_time, make_route
)

from problem1_solution import RouteGenerator, OutputWriter
from problem2_solution import (
    Q2DataLoader, Q2RoutePreprocessor, Q2RouteGenerator, Q2OutputWriter
)


# ==================== 权重方案定义 ====================
# 8组权重方案: (w_time, w_transit, w_util, w_fuel)
# w_util 对应最小化浪费座公里 (seat_km - pass_km)
WEIGHT_SCHEMES = {
    'equal':          (0.25, 0.25, 0.25, 0.25),  # 等权重
    'priority':       (0.40, 0.30, 0.20, 0.10),  # 按题目优先级递减
    'time_heavy':     (0.55, 0.20, 0.15, 0.10),  # 偏向总飞机使用时间
    'transit_heavy':  (0.20, 0.55, 0.15, 0.10),  # 偏向人员总在途时间
    'util_heavy':     (0.15, 0.20, 0.55, 0.10),  # 偏向座位利用率
    'fuel_heavy':     (0.15, 0.15, 0.20, 0.50),  # 偏向燃油消耗
    'time_fuel':      (0.45, 0.15, 0.10, 0.30),  # 偏向时间+燃油
    'balanced':       (0.35, 0.25, 0.25, 0.15),  # 均衡偏时间
}

# 权重方案的中文描述
WEIGHT_DESCRIPTIONS = {
    'equal':         '等权重',
    'priority':      '按题目优先级',
    'time_heavy':    '偏总飞机时间',
    'transit_heavy': '偏在途时间',
    'util_heavy':    '偏座位利用率',
    'fuel_heavy':    '偏燃油消耗',
    'time_fuel':     '偏时间+燃油',
    'balanced':      '均衡偏时间',
}


# ==================== Q1 加权求解器 ====================
class WeightedSolverQ1:
    """
    Q1加权单目标求解器

    将四个目标通过归一化加权转化为单一线性目标:
        min Z = w1*(T/T0) + w2*(τ/τ0) + w3*(W/W0) + w4*(F/F0)
    其中:
        T  = Σ t_r * x_r             (总飞机使用时间)
        τ  = Σ τ_{r,f} * y_{r,f}     (总在途时间)
        W  = Σ c_r*ℓ_r*x_r - Σ δ_{r,f}*y_{r,f}  (浪费座公里 = seat_km - pass_km)
        F  = Σ fuel_r * x_r           (总燃油消耗)
        T0,τ0,W0,F0 为归一化常数

    优势: 无需Dinkelbach迭代，一次求解即可获得帕累托有效解
    """

    # 默认归一化常数（从Q1字典序解获取，将自适应更新）
    NORM = {'time': 15808.0, 'transit': 119092.0, 'wasted': 417699.0, 'fuel': 127634.6}

    # 求解参数
    TIME_LIMIT = 180        # 单次求解时间限制(秒)
    REL_GAP = 0.03          # 相对间隙

    @staticmethod
    def _has_solution(model):
        """检查模型是否有可行解"""
        if model.status in (cp.COPT.OPTIMAL, cp.COPT.TIMEOUT,
                            cp.COPT.NODELIMIT, cp.COPT.INTERRUPTED):
            try:
                _ = model.objval
                return True
            except Exception:
                return False
        return False

    def solve(self, routes, demands_by_facility, weights, norm=None, time_limit=None):
        """
        使用给定权重构建并求解加权单目标COPT模型

        参数:
            routes: 候选路径列表
            demands_by_facility: 按设施分组的需求
            weights: (w_time, w_transit, w_util, w_fuel) 权重元组
            norm: 归一化常数dict，None则使用默认值
            time_limit: 求解时间限制(秒)，None则使用默认值

        返回:
            (selected_routes, assignments, metrics) 或 ([], {}, None)
        """
        if norm is None:
            norm = self.NORM
        if time_limit is None:
            time_limit = self.TIME_LIMIT

        w_time, w_transit, w_util, w_fuel = weights

        env = cp.Envr()
        model = env.createModel('Q1_Weighted')
        model.setParam(cp.COPT.Param.TimeLimit, time_limit)
        model.setParam(cp.COPT.Param.RelGap, self.REL_GAP)
        model.setParam(cp.COPT.Param.LogToConsole, 0)

        # --- 创建变量 ---
        x = {}
        y = {}
        for r in routes:
            x[r.route_id] = model.addVar(vtype=cp.COPT.BINARY, name=f'x_{r.route_id}')
            for f in r.dest_facilities:
                if f in demands_by_facility:
                    y[(r.route_id, f)] = model.addVar(
                        vtype=cp.COPT.INTEGER, lb=0, name=f'y_{r.route_id}_{f}')

        # --- 约束1: 需求满足 ---
        for f, info in demands_by_facility.items():
            terms = [y[(r.route_id, f)] for r in routes
                     if f in r.dest_facilities and (r.route_id, f) in y]
            if terms:
                model.addConstr(cp.quicksum(terms) == info['total'])

        # --- 约束2: 起点感知 ---
        for f, info in demands_by_facility.items():
            for origin, cnt in info['by_origin'].items():
                if origin != 'LAND' and cnt > 0:
                    terms = [y[(r.route_id, f)] for r in routes
                             if f in r.dest_facilities and r.airport == origin
                             and (r.route_id, f) in y]
                    if terms:
                        model.addConstr(cp.quicksum(terms) >= cnt)

        # --- 约束3: 容量 ---
        for r in routes:
            terms = [y[(r.route_id, f)] for f in r.dest_facilities
                     if (r.route_id, f) in y]
            if terms:
                model.addConstr(cp.quicksum(terms) <= r.capacity * x[r.route_id])

        # --- 构建加权单目标 ---
        # T = Σ t_r * x_r
        time_expr = cp.quicksum(r.time * x[r.route_id] for r in routes)

        # τ = Σ τ_{r,f} * y_{r,f}
        transit_terms = []
        for r in routes:
            for f in r.dest_facilities:
                if (r.route_id, f) in y and f in r.transit_times:
                    transit_terms.append(r.transit_times[f] * y[(r.route_id, f)])
        transit_expr = cp.quicksum(transit_terms) if transit_terms else 0

        # W = seat_km - pass_km = Σ c_r*ℓ_r*x_r - Σ δ_{r,f}*y_{r,f}
        seatkm_expr = cp.quicksum(r.capacity * r.total_distance * x[r.route_id]
                                  for r in routes)
        passkm_terms = []
        for r in routes:
            for f in r.dest_facilities:
                if (r.route_id, f) in y and f in r.dist_to_facility:
                    passkm_terms.append(r.dist_to_facility[f] * y[(r.route_id, f)])
        passkm_expr = cp.quicksum(passkm_terms) if passkm_terms else 0
        wasted_expr = seatkm_expr - passkm_expr

        # F = Σ fuel_r * x_r
        fuel_expr = cp.quicksum(r.fuel * x[r.route_id] for r in routes)

        # 归一化加权单目标
        obj = (w_time * time_expr / norm['time']
               + w_transit * transit_expr / norm['transit']
               + w_util * wasted_expr / norm['wasted']
               + w_fuel * fuel_expr / norm['fuel'])

        model.setObjective(obj, sense=cp.COPT.MINIMIZE)

        # --- 求解 ---
        t0 = time_module.time()
        model.solve()
        t1 = time_module.time()

        if not self._has_solution(model):
            print(f"  错误: 求解状态={model.status}, 无可行解")
            return [], {}, None

        # --- 提取解 ---
        selected_routes = [r for r in routes if x[r.route_id].x > 0.5]
        assignments = {}
        for r in routes:
            if x[r.route_id].x > 0.5:
                for f in r.dest_facilities:
                    if (r.route_id, f) in y:
                        val = int(round(y[(r.route_id, f)].x))
                        if val > 0:
                            assignments[(r.route_id, f)] = val

        # --- 计算指标 ---
        metrics = self._calc_metrics(selected_routes, assignments)

        return selected_routes, assignments, metrics

    @staticmethod
    def _calc_metrics(selected_routes, assignments):
        """计算五项指标"""
        total_time = sum(r.time for r in selected_routes)
        total_transit = 0
        for r in selected_routes:
            for f in r.dest_facilities:
                n = assignments.get((r.route_id, f), 0)
                total_transit += n * r.transit_times.get(f, 0)
        total_flights = len(selected_routes)
        total_fuel = sum(r.fuel for r in selected_routes)
        pass_km = 0.0
        seat_km = 0.0
        for r in selected_routes:
            seat_km += r.capacity * r.total_distance
            for f in r.dest_facilities:
                n = assignments.get((r.route_id, f), 0)
                pass_km += n * r.dist_to_facility.get(f, 0)
        seat_util = pass_km / seat_km if seat_km > 0 else 0
        wasted = seat_km - pass_km

        return {
            'total_time': total_time,
            'total_transit': total_transit,
            'total_flights': total_flights,
            'total_fuel': total_fuel,
            'seat_util': seat_util,
            'pass_km': pass_km,
            'seat_km': seat_km,
            'wasted': wasted,
            'solve_time': 0.0,  # 由调用者填充
        }


# ==================== Q2 加权求解器 ====================
class WeightedSolverQ2:
    """
    Q2加权单目标求解器

    与Q1类似，但需要处理出海/海返/穿梭三类需求和航段级容量约束。
    目标函数中τ和W的计算需要覆盖三类需求。
    """

    # 默认归一化常数（从Q2字典序解获取，将自适应更新）
    NORM = {'time': 18561.0, 'transit': 268468.0, 'wasted': 143861.0, 'fuel': 144970.2}

    # 求解参数
    TIME_LIMIT = 300
    REL_GAP = 0.05

    @staticmethod
    def _has_solution(model):
        """检查模型是否有可行解"""
        if model.status in (cp.COPT.OPTIMAL, cp.COPT.TIMEOUT,
                            cp.COPT.NODELIMIT, cp.COPT.INTERRUPTED):
            try:
                _ = model.objval
                return True
            except Exception:
                return False
        return False

    def solve(self, routes, precomputed, outbound_demand, inbound_demand,
              shuttle_demand, weights, norm=None, time_limit=None):
        """
        使用给定权重构建并求解Q2加权单目标COPT模型

        参数:
            routes: 路径列表
            precomputed: {route_id: 预处理信息}
            outbound_demand: {facility: {'total', 'by_origin'}}
            inbound_demand: {facility: {'total', 'by_dest'}}
            shuttle_demand: {(f1,f2): {'total'}}
            weights: (w_time, w_transit, w_util, w_fuel)
            norm: 归一化常数dict
            time_limit: 求解时间限制(秒)

        返回:
            (selected_routes, yo_vals, yi_vals, ys_vals, metrics) 或 ([],...None)
        """
        if norm is None:
            norm = self.NORM
        if time_limit is None:
            time_limit = self.TIME_LIMIT

        w_time, w_transit, w_util, w_fuel = weights

        env = cp.Envr()
        model = env.createModel('Q2_Weighted')
        model.setParam(cp.COPT.Param.TimeLimit, time_limit)
        model.setParam(cp.COPT.Param.RelGap, self.REL_GAP)
        model.setParam(cp.COPT.Param.LogToConsole, 0)

        # --- 创建变量 ---
        x = {}
        yo = {}
        yi = {}
        ys = {}

        for r in routes:
            pre = precomputed[r.route_id]
            x[r.route_id] = model.addVar(vtype=cp.COPT.BINARY, name=f'x_{r.route_id}')
            for f in pre['facility_pos']:
                if f in outbound_demand:
                    yo[(r.route_id, f)] = model.addVar(
                        vtype=cp.COPT.INTEGER, lb=0, name=f'yo_{r.route_id}_{f}')
                if f in inbound_demand:
                    yi[(r.route_id, f)] = model.addVar(
                        vtype=cp.COPT.INTEGER, lb=0, name=f'yi_{r.route_id}_{f}')
            for (f1, f2) in pre['shuttle_info']:
                if (f1, f2) in shuttle_demand:
                    ys[(r.route_id, f1, f2)] = model.addVar(
                        vtype=cp.COPT.INTEGER, lb=0,
                        name=f'ys_{r.route_id}_{f1}_{f2}')

        # --- 约束1: 需求满足 ---
        for f, info in outbound_demand.items():
            terms = [yo[(r.route_id, f)] for r in routes
                     if (r.route_id, f) in yo]
            if terms:
                model.addConstr(cp.quicksum(terms) == info['total'])

        for f, info in inbound_demand.items():
            terms = [yi[(r.route_id, f)] for r in routes
                     if (r.route_id, f) in yi]
            if terms:
                model.addConstr(cp.quicksum(terms) == info['total'])

        for (f1, f2), info in shuttle_demand.items():
            terms = [ys[(r.route_id, f1, f2)] for r in routes
                     if (r.route_id, f1, f2) in ys]
            if terms:
                model.addConstr(cp.quicksum(terms) == info['total'])

        # --- 约束2: 起点感知 ---
        for f, info in outbound_demand.items():
            for origin, cnt in info['by_origin'].items():
                if origin != 'LAND' and cnt > 0:
                    terms = [yo[(r.route_id, f)] for r in routes
                             if (r.route_id, f) in yo and r.airport == origin]
                    if terms:
                        model.addConstr(cp.quicksum(terms) >= cnt)

        for f, info in inbound_demand.items():
            for dest, cnt in info['by_dest'].items():
                if dest != 'LAND' and cnt > 0:
                    terms = [yi[(r.route_id, f)] for r in routes
                             if (r.route_id, f) in yi and r.airport == dest]
                    if terms:
                        model.addConstr(cp.quicksum(terms) >= cnt)

        # --- 约束3: 航段级容量 ---
        for r in routes:
            pre = precomputed[r.route_id]
            cap = r.capacity
            for k, (out_after, in_before, shut_cross) in enumerate(pre['segments']):
                terms = []
                for f in out_after:
                    if (r.route_id, f) in yo:
                        terms.append(yo[(r.route_id, f)])
                for f in in_before:
                    if (r.route_id, f) in yi:
                        terms.append(yi[(r.route_id, f)])
                for (f1, f2) in shut_cross:
                    if (r.route_id, f1, f2) in ys:
                        terms.append(ys[(r.route_id, f1, f2)])
                if terms:
                    model.addConstr(cp.quicksum(terms) <= cap * x[r.route_id])

        # --- 构建加权单目标 ---
        # T = Σ t_r * x_r
        time_expr = cp.quicksum(r.time * x[r.route_id] for r in routes)

        # τ = Σ(出海在途*yo) + Σ(海返在途*yi) + Σ(穿梭在途*ys)
        transit_terms = []
        for r in routes:
            pre = precomputed[r.route_id]
            for f in pre['facility_pos']:
                if (r.route_id, f) in yo and f in pre['time_to']:
                    transit_terms.append(pre['time_to'][f] * yo[(r.route_id, f)])
                if (r.route_id, f) in yi and f in pre['time_from']:
                    transit_terms.append(pre['time_from'][f] * yi[(r.route_id, f)])
            for (f1, f2) in pre['shuttle_info']:
                if (r.route_id, f1, f2) in ys:
                    transit_terms.append(pre['shuttle_info'][(f1, f2)]['time']
                                         * ys[(r.route_id, f1, f2)])
        transit_expr = cp.quicksum(transit_terms) if transit_terms else 0

        # W = seat_km - pass_km
        seatkm_expr = cp.quicksum(r.capacity * r.total_distance * x[r.route_id]
                                  for r in routes)
        passkm_terms = []
        for r in routes:
            pre = precomputed[r.route_id]
            for f in pre['facility_pos']:
                if (r.route_id, f) in yo and f in pre['dist_to']:
                    passkm_terms.append(pre['dist_to'][f] * yo[(r.route_id, f)])
                if (r.route_id, f) in yi and f in pre['dist_from']:
                    passkm_terms.append(pre['dist_from'][f] * yi[(r.route_id, f)])
            for (f1, f2) in pre['shuttle_info']:
                if (r.route_id, f1, f2) in ys:
                    passkm_terms.append(pre['shuttle_info'][(f1, f2)]['dist']
                                         * ys[(r.route_id, f1, f2)])
        passkm_expr = cp.quicksum(passkm_terms) if passkm_terms else 0
        wasted_expr = seatkm_expr - passkm_expr

        # F = Σ fuel_r * x_r
        fuel_expr = cp.quicksum(r.fuel * x[r.route_id] for r in routes)

        # 归一化加权单目标
        obj = (w_time * time_expr / norm['time']
               + w_transit * transit_expr / norm['transit']
               + w_util * wasted_expr / norm['wasted']
               + w_fuel * fuel_expr / norm['fuel'])

        model.setObjective(obj, sense=cp.COPT.MINIMIZE)

        # --- 求解 ---
        t0 = time_module.time()
        model.solve()
        t1 = time_module.time()

        if not self._has_solution(model):
            print(f"  错误: 求解状态={model.status}, 无可行解")
            return [], {}, {}, {}, None

        # --- 提取解 ---
        selected_routes = [r for r in routes if x[r.route_id].x > 0.5]
        yo_vals = {}
        yi_vals = {}
        ys_vals = {}

        for r in selected_routes:
            pre = precomputed[r.route_id]
            for f in pre['facility_pos']:
                if (r.route_id, f) in yo:
                    val = int(round(yo[(r.route_id, f)].x))
                    if val > 0:
                        yo_vals[(r.route_id, f)] = val
                if (r.route_id, f) in yi:
                    val = int(round(yi[(r.route_id, f)].x))
                    if val > 0:
                        yi_vals[(r.route_id, f)] = val
            for (f1, f2) in pre['shuttle_info']:
                if (r.route_id, f1, f2) in ys:
                    val = int(round(ys[(r.route_id, f1, f2)].x))
                    if val > 0:
                        ys_vals[(r.route_id, f1, f2)] = val

        # --- 计算指标 ---
        metrics = self._calc_metrics(
            selected_routes, yo_vals, yi_vals, ys_vals, precomputed)
        metrics['solve_time'] = t1 - t0

        return selected_routes, yo_vals, yi_vals, ys_vals, metrics

    @staticmethod
    def _calc_metrics(selected_routes, yo_vals, yi_vals, ys_vals, precomputed):
        """计算五项指标"""
        total_time = sum(r.time for r in selected_routes)
        total_transit = 0
        pass_km = 0.0
        seat_km = 0.0

        for r in selected_routes:
            pre = precomputed[r.route_id]
            seat_km += r.capacity * r.total_distance
            for f in pre['facility_pos']:
                n_out = yo_vals.get((r.route_id, f), 0)
                total_transit += n_out * pre['time_to'].get(f, 0)
                pass_km += n_out * pre['dist_to'].get(f, 0)
                n_in = yi_vals.get((r.route_id, f), 0)
                total_transit += n_in * pre['time_from'].get(f, 0)
                pass_km += n_in * pre['dist_from'].get(f, 0)
            for (f1, f2) in pre['shuttle_info']:
                n_shut = ys_vals.get((r.route_id, f1, f2), 0)
                total_transit += n_shut * pre['shuttle_info'][(f1, f2)]['time']
                pass_km += n_shut * pre['shuttle_info'][(f1, f2)]['dist']

        total_flights = len(selected_routes)
        total_fuel = sum(r.fuel for r in selected_routes)
        seat_util = pass_km / seat_km if seat_km > 0 else 0
        wasted = seat_km - pass_km

        return {
            'total_time': total_time,
            'total_transit': total_transit,
            'total_flights': total_flights,
            'total_fuel': total_fuel,
            'seat_util': seat_util,
            'pass_km': pass_km,
            'seat_km': seat_km,
            'wasted': wasted,
            'solve_time': 0.0,
        }


# ==================== 敏感性分析结果输出器 ====================
class SensitivityAnalyzer:
    """权重敏感性分析：运行多组权重方案，汇总对比结果，生成图表"""

    @staticmethod
    def write_comparison_csv(all_results, path, problem_tag):
        """写权重方案对比CSV"""
        with open(path, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow(['scheme', 'w_time', 'w_transit', 'w_util', 'w_fuel',
                        'total_time_min', 'total_time_h',
                        'total_transit_min', 'total_transit_h',
                        'total_flights', 'total_fuel_kg',
                        'seat_util_pct', 'pass_km', 'seat_km', 'wasted_seatkm',
                        'solve_time_s'])
            for scheme_name, weights, metrics in all_results:
                if metrics is None:
                    continue
                w_time, w_transit, w_util, w_fuel = weights
                w.writerow([
                    scheme_name, w_time, w_transit, w_util, w_fuel,
                    metrics['total_time'], f"{metrics['total_time']/60:.6f}",
                    metrics['total_transit'], f"{metrics['total_transit']/60:.6f}",
                    metrics['total_flights'], fmt6(metrics['total_fuel']),
                    f"{metrics['seat_util']*100:.6f}",
                    fmt6(metrics['pass_km']), fmt6(metrics['seat_km']),
                    fmt6(metrics['wasted']), f"{metrics.get('solve_time', 0):.1f}"
                ])
        print(f"  对比文件: {path}")

    @staticmethod
    def write_comparison_txt(all_results, path, problem_tag, norm):
        """写权重方案对比TXT（人类可读）"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write(f"{problem_tag} 加权单目标法 — 权重敏感性分析\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"归一化常数: T0={norm['time']:.1f}, τ0={norm['transit']:.1f}, "
                    f"W0={norm['wasted']:.1f}, F0={norm['fuel']:.1f}\n\n")
            f.write(f"{'方案':<16s} {'w_time':>6s} {'w_tran':>6s} {'w_util':>6s} {'w_fuel':>6s}"
                    f" | {'时间(min)':>10s} {'在途(min)':>10s} {'架次':>4s}"
                    f" {'燃油(kg)':>12s} {'利用率':>8s} {'求解(s)':>8s}\n")
            f.write("-" * 110 + "\n")
            for scheme_name, weights, metrics in all_results:
                if metrics is None:
                    f.write(f"{scheme_name:<16s} — 无可行解\n")
                    continue
                w_time, w_transit, w_util, w_fuel = weights
                desc = WEIGHT_DESCRIPTIONS.get(scheme_name, scheme_name)
                f.write(f"{desc:<16s} {w_time:>6.2f} {w_transit:>6.2f} "
                        f"{w_util:>6.2f} {w_fuel:>6.2f}"
                        f" | {metrics['total_time']:>10d} "
                        f"{metrics['total_transit']:>10d} "
                        f"{metrics['total_flights']:>4d} "
                        f"{fmt6(metrics['total_fuel']):>12s} "
                        f"{metrics['seat_util']*100:>7.2f}% "
                        f"{metrics.get('solve_time', 0):>8.1f}\n")
            f.write("\n")
        print(f"  对比文本: {path}")

    @staticmethod
    def plot_sensitivity(all_results, figures_dir, problem_tag):
        """生成权重敏感性分析图表（雷达图 + 折线图）"""
        plt = setup_matplotlib()
        os.makedirs(figures_dir, exist_ok=True)

        # 筛选有效结果
        valid = [(name, w, m) for name, w, m in all_results if m is not None]
        if not valid:
            print("  无有效结果，跳过绘图")
            return

        scheme_names = [WEIGHT_DESCRIPTIONS.get(n, n) for n, _, _ in valid]
        n_schemes = len(valid)

        # === 图1: 四指标归一化折线图 ===
        fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
        fig1.suptitle(f'{problem_tag} 权重敏感性分析 — 四项指标归一化对比',
                       fontsize=14, fontweight='bold')

        # 归一化各指标（除以最大值）
        times = [m['total_time'] for _, _, m in valid]
        transits = [m['total_transit'] for _, _, m in valid]
        fuels = [m['total_fuel'] for _, _, m in valid]
        utils = [m['seat_util'] * 100 for _, _, m in valid]

        max_time = max(times) if times else 1
        max_transit = max(transits) if transits else 1
        max_fuel = max(fuels) if fuels else 1
        min_util = min(utils) if utils else 0
        max_util = max(utils) if utils else 1

        x_pos = range(n_schemes)

        # 总飞机使用时间
        ax = axes1[0, 0]
        norm_times = [t / max_time for t in times]
        ax.bar(x_pos, norm_times, color='#2196F3', alpha=0.8)
        ax.set_title('总飞机使用时间 (归一化)', fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(scheme_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('归一化值')

        # 人员在途时间
        ax = axes1[0, 1]
        norm_transits = [t / max_transit for t in transits]
        ax.bar(x_pos, norm_transits, color='#4CAF50', alpha=0.8)
        ax.set_title('人员总在途时间 (归一化)', fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(scheme_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('归一化值')

        # 座位利用率
        ax = axes1[1, 0]
        ax.bar(x_pos, utils, color='#FF9800', alpha=0.8)
        ax.set_title('座位利用率 (%)', fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(scheme_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('%')
        # 标注数值
        for i, v in enumerate(utils):
            ax.text(i, v + 0.2, f'{v:.1f}%', ha='center', va='bottom', fontsize=7)

        # 总燃油消耗
        ax = axes1[1, 1]
        norm_fuels = [f / max_fuel for f in fuels]
        ax.bar(x_pos, norm_fuels, color='#f44336', alpha=0.8)
        ax.set_title('总燃油消耗 (归一化)', fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(scheme_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('归一化值')

        fig1.tight_layout(rect=[0, 0, 1, 0.96])
        path1 = os.path.join(figures_dir, f'{problem_tag.lower()}_weight_bars.png')
        fig1.savefig(path1, dpi=200, bbox_inches='tight')
        plt.close(fig1)
        print(f"  柱状图: {path1}")

        # === 图2: 雷达图 ===
        fig2, ax2 = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        # 选取有代表性的方案绘制雷达图（最多6个，避免拥挤）
        plot_schemes = ['equal', 'time_heavy', 'transit_heavy',
                        'util_heavy', 'fuel_heavy', 'priority']
        plot_data = [(WEIGHT_DESCRIPTIONS.get(n, n), m)
                     for n, _, m in valid if n in plot_schemes]

        if plot_data:
            categories = ['总时间', '在途时间', '座位利用率', '燃油经济性']
            n_cats = len(categories)
            angles = [n / n_cats * 2 * math.pi for n in range(n_cats)]
            angles += angles[:1]

            colors = ['#2196F3', '#4CAF50', '#FF9800', '#f44336', '#9C27B0', '#795548']

            for idx, (label, m) in enumerate(plot_data):
                # 归一化到[0,1]，注意利用率和燃油的方向
                vals = [
                    m['total_time'] / max_time,
                    m['total_transit'] / max_transit,
                    (m['seat_util'] - min_util) / (max_util - min_util) if max_util > min_util else 0.5,
                    1.0 - m['total_fuel'] / max_fuel,  # 燃油越低越好
                ]
                vals += vals[:1]
                ax2.plot(angles, vals, 'o-', linewidth=1.5, markersize=4,
                         label=label, color=colors[idx % len(colors)])
                ax2.fill(angles, vals, alpha=0.08, color=colors[idx % len(colors)])

            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(categories, fontsize=11)
            ax2.set_ylim(0, 1.1)
            ax2.set_title(f'{problem_tag} 不同权重方案的帕累托表现',
                          fontsize=13, fontweight='bold', pad=20)
            ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)

        fig2.tight_layout()
        path2 = os.path.join(figures_dir, f'{problem_tag.lower()}_weight_radar.png')
        fig2.savefig(path2, dpi=200, bbox_inches='tight')
        plt.close(fig2)
        print(f"  雷达图: {path2}")

        # === 图3: 权重-指标折线图 ===
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        # 按w_time排序
        sorted_results = sorted(valid, key=lambda x: x[1][0])  # 按w_time排序
        w_times_sorted = [w[0] for _, w, _ in sorted_results]

        ax3_twin = ax3.twinx()

        # 归一化指标
        norm_t = [m['total_time'] / max_time for _, _, m in sorted_results]
        norm_tr = [m['total_transit'] / max_transit for _, _, m in sorted_results]
        norm_u = [(m['seat_util'] - min_util) / (max_util - min_util)
                  if max_util > min_util else 0.5 for _, _, m in sorted_results]
        norm_f = [m['total_fuel'] / max_fuel for _, _, m in sorted_results]

        line1, = ax3.plot(w_times_sorted, norm_t, 'o-', color='#2196F3',
                          label='总时间(归一化)', linewidth=2, markersize=6)
        line2, = ax3.plot(w_times_sorted, norm_tr, 's-', color='#4CAF50',
                          label='在途时间(归一化)', linewidth=2, markersize=6)
        line4, = ax3.plot(w_times_sorted, norm_f, '^-', color='#f44336',
                          label='燃油(归一化)', linewidth=2, markersize=6)
        line3, = ax3_twin.plot(w_times_sorted, [m['seat_util']*100 for _, _, m in sorted_results],
                               'D-', color='#FF9800', label='座位利用率(%)',
                               linewidth=2, markersize=6)

        # 标注方案名
        for i, (name, _, _) in enumerate(sorted_results):
            desc = WEIGHT_DESCRIPTIONS.get(name, name)
            ax3.annotate(desc, (w_times_sorted[i], norm_t[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=7)

        ax3.set_xlabel('时间权重 $w_1$', fontsize=12)
        ax3.set_ylabel('归一化指标值', fontsize=11)
        ax3_twin.set_ylabel('座位利用率 (%)', fontsize=11, color='#FF9800')
        ax3_twin.tick_params(axis='y', labelcolor='#FF9800')
        ax3.set_title(f'{problem_tag} 权重 $w_1$ 对各指标的影响',
                      fontsize=13, fontweight='bold')

        lines = [line1, line2, line3, line4]
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='center left', fontsize=9)

        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        path3 = os.path.join(figures_dir, f'{problem_tag.lower()}_weight_lines.png')
        fig3.savefig(path3, dpi=200, bbox_inches='tight')
        plt.close(fig3)
        print(f"  折线图: {path3}")


# ==================== Q1 主函数 ====================
def run_q1_weighted():
    """Q1加权单目标法主函数：路径生成 + 自适应归一化 + 多组权重求解 + 敏感性分析"""
    print("=" * 60)
    print("问题1: 加权单目标法 — 权重敏感性分析")
    print("=" * 60)

    # 1. 数据加载
    print("\n[1] 数据加载...")
    loader = DataLoader()
    distances = loader.load_distances()
    demands = loader.load_demands_q1()
    demands_by_facility = loader.count_demands(demands)
    print(f"  需求: {len(demands)}人, 设施: {len(demands_by_facility)}个")

    # 2. 路径生成（只做一次，所有权重方案复用）
    print("\n[2] 路径生成...")
    t0 = time_module.time()
    gen = RouteGenerator(distances)
    routes = gen.generate_all()
    t1 = time_module.time()
    print(f"  耗时: {t1-t0:.1f}s, 候选路径: {len(routes)}条")

    # 3. 自适应归一化：先等权重求解获取参考值
    print("\n[3] 自适应归一化（等权重预求解）...")
    solver = WeightedSolverQ1()
    t0 = time_module.time()
    _, _, ref_metrics = solver.solve(
        routes, demands_by_facility,
        weights=WEIGHT_SCHEMES['equal'],
        time_limit=120)
    t1 = time_module.time()
    if ref_metrics is not None:
        # 更新归一化常数
        norm = {
            'time': ref_metrics['total_time'],
            'transit': ref_metrics['total_transit'],
            'wasted': ref_metrics['wasted'] if ref_metrics['wasted'] > 0 else 1.0,
            'fuel': ref_metrics['total_fuel'],
        }
        print(f"  参考值: T={norm['time']:.0f}, τ={norm['transit']:.0f}, "
              f"W={norm['wasted']:.0f}, F={norm['fuel']:.0f}")
    else:
        norm = WeightedSolverQ1.NORM
        print(f"  使用默认归一化常数")

    # 4. 多组权重方案求解
    print("\n[4] 权重敏感性分析（8组方案）...")
    output_base = os.path.join(os.path.dirname(OUTPUT_DIR), 'Problem1_weighted')
    os.makedirs(output_base, exist_ok=True)

    all_results = []
    writer = OutputWriter(distances, demands, demands_by_facility)

    for scheme_name, weights in WEIGHT_SCHEMES.items():
        desc = WEIGHT_DESCRIPTIONS[scheme_name]
        print(f"\n  --- 方案: {desc} ({scheme_name}) ---")
        print(f"  权重: w_time={weights[0]}, w_transit={weights[1]}, "
              f"w_util={weights[2]}, w_fuel={weights[3]}")

        t0 = time_module.time()
        selected, assignments, metrics = solver.solve(
            routes, demands_by_facility,
            weights=weights, norm=norm,
            time_limit=solver.TIME_LIMIT)
        t1 = time_module.time()
        metrics['solve_time'] = t1 - t0

        if metrics is None or not selected:
            print(f"  无可行解!")
            all_results.append((scheme_name, weights, None))
            continue

        print(f"  结果: 时间={metrics['total_time']}min, "
              f"在途={metrics['total_transit']}min, "
              f"架次={metrics['total_flights']}, "
              f"燃油={fmt6(metrics['total_fuel'])}kg, "
              f"利用率={metrics['seat_util']*100:.2f}%")

        # 输出每组方案的结果
        scheme_dir = os.path.join(output_base, scheme_name)
        os.makedirs(scheme_dir, exist_ok=True)

        # 写routes和assignments
        person_assignments = writer._assign_people(selected, assignments)
        flight_numbers = writer._assign_flight_numbers(selected)

        routes_path = os.path.join(scheme_dir, 'q1-routes.csv')
        writer._write_routes(selected, flight_numbers, routes_path)

        assign_path = os.path.join(scheme_dir, 'q1-assignments.csv')
        writer._write_assignments(person_assignments, flight_numbers, assign_path)

        # 写指标
        metrics_path = os.path.join(scheme_dir, 'q1-metrics.txt')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write(f"方案: {desc} ({scheme_name})\n")
            f.write(f"权重: w_time={weights[0]}, w_transit={weights[1]}, "
                    f"w_util={weights[2]}, w_fuel={weights[3]}\n\n")
            f.write(f"总飞机使用时间: {metrics['total_time']} 分钟 "
                    f"({metrics['total_time']/60:.6f} 小时)\n")
            f.write(f"人员总在途时间: {metrics['total_transit']} 分钟 "
                    f"({metrics['total_transit']/60:.6f} 小时)\n")
            f.write(f"总架次数: {metrics['total_flights']}\n")
            f.write(f"总燃油消耗量: {fmt6(metrics['total_fuel'])} kg\n")
            f.write(f"座位利用率: {metrics['seat_util']*100:.6f}%\n")
            f.write(f"  客座公里: {fmt6(metrics['pass_km'])}\n")
            f.write(f"  可用座公里: {fmt6(metrics['seat_km'])}\n")
            f.write(f"  浪费座公里: {fmt6(metrics['wasted'])}\n")
            f.write(f"求解时间: {metrics['solve_time']:.1f}s\n")

        all_results.append((scheme_name, weights, metrics))

    # 5. 汇总输出
    print("\n[5] 汇总输出...")
    comparison_csv = os.path.join(output_base, 'q1-weight-comparison.csv')
    SensitivityAnalyzer.write_comparison_csv(all_results, comparison_csv, 'Q1')

    comparison_txt = os.path.join(output_base, 'q1-weight-comparison.txt')
    SensitivityAnalyzer.write_comparison_txt(all_results, comparison_txt, 'Q1', norm)

    # 6. 生成图表
    print("\n[6] 生成敏感性分析图表...")
    SensitivityAnalyzer.plot_sensitivity(all_results, FIGURES_DIR, 'Q1')

    print("\n" + "=" * 60)
    print("Q1加权单目标法完成!")
    print("=" * 60)
    return all_results


# ==================== Q2 主函数 ====================
def run_q2_weighted():
    """Q2加权单目标法主函数：路径生成 + 自适应归一化 + 多组权重求解 + 敏感性分析"""
    print("=" * 60)
    print("问题2: 加权单目标法 — 权重敏感性分析")
    print("=" * 60)

    # 1. 数据加载与分类
    print("\n[1] 数据加载与分类...")
    loader = Q2DataLoader()
    distances = loader.load_distances()
    demands = loader.load_demands_q2()

    outbound, inbound, shuttle = Q2DataLoader.classify_demands(demands)
    outbound_demand = Q2DataLoader.group_outbound(outbound)
    inbound_demand = Q2DataLoader.group_inbound(inbound)
    shuttle_demand = Q2DataLoader.group_shuttle(shuttle)

    print(f"  总需求: {len(demands)}人")
    print(f"  出海: {len(outbound)}人, 海返: {len(inbound)}人, 穿梭: {len(shuttle)}人")

    # 2. 路径生成
    print("\n[2] 路径生成...")
    t0 = time_module.time()
    shuttle_pairs = list(shuttle_demand.keys())
    gen = Q2RouteGenerator(distances)
    routes = gen.generate_all_q2(shuttle_pairs)
    t1 = time_module.time()
    print(f"  耗时: {t1-t0:.1f}s, 候选路径: {len(routes)}条")

    # 3. 路由预处理
    print("\n[3] 路由预处理...")
    t0 = time_module.time()
    precomputed = {}
    for r in routes:
        precomputed[r.route_id] = Q2RoutePreprocessor.preprocess(r, distances)
    t1 = time_module.time()
    print(f"  预处理: {len(routes)}条, 耗时 {t1-t0:.1f}s")

    # 4. 自适应归一化
    print("\n[4] 自适应归一化（等权重预求解）...")
    solver = WeightedSolverQ2()
    _, _, _, _, ref_metrics = solver.solve(
        routes, precomputed, outbound_demand, inbound_demand, shuttle_demand,
        weights=WEIGHT_SCHEMES['equal'],
        time_limit=200)

    if ref_metrics is not None:
        norm = {
            'time': ref_metrics['total_time'],
            'transit': ref_metrics['total_transit'],
            'wasted': ref_metrics['wasted'] if ref_metrics['wasted'] > 0 else 1.0,
            'fuel': ref_metrics['total_fuel'],
        }
        print(f"  参考值: T={norm['time']:.0f}, τ={norm['transit']:.0f}, "
              f"W={norm['wasted']:.0f}, F={norm['fuel']:.0f}")
    else:
        norm = WeightedSolverQ2.NORM
        print(f"  使用默认归一化常数")

    # 5. 多组权重方案求解
    print("\n[5] 权重敏感性分析（8组方案）...")
    output_base = os.path.join(os.path.dirname(OUTPUT_DIR_Q2), 'Problem2_weighted')
    os.makedirs(output_base, exist_ok=True)

    all_results = []
    writer = Q2OutputWriter(distances, demands, precomputed, outbound, inbound, shuttle)

    for scheme_name, weights in WEIGHT_SCHEMES.items():
        desc = WEIGHT_DESCRIPTIONS[scheme_name]
        print(f"\n  --- 方案: {desc} ({scheme_name}) ---")
        print(f"  权重: w_time={weights[0]}, w_transit={weights[1]}, "
              f"w_util={weights[2]}, w_fuel={weights[3]}")

        t0 = time_module.time()
        selected, yo_vals, yi_vals, ys_vals, metrics = solver.solve(
            routes, precomputed, outbound_demand, inbound_demand, shuttle_demand,
            weights=weights, norm=norm,
            time_limit=solver.TIME_LIMIT)
        t1 = time_module.time()
        if metrics:
            metrics['solve_time'] = t1 - t0

        if metrics is None or not selected:
            print(f"  无可行解!")
            all_results.append((scheme_name, weights, None))
            continue

        print(f"  结果: 时间={metrics['total_time']}min, "
              f"在途={metrics['total_transit']}min, "
              f"架次={metrics['total_flights']}, "
              f"燃油={fmt6(metrics['total_fuel'])}kg, "
              f"利用率={metrics['seat_util']*100:.2f}%")

        # 输出每组方案的结果
        scheme_dir = os.path.join(output_base, scheme_name)
        os.makedirs(scheme_dir, exist_ok=True)

        person_assignments = writer._assign_people(
            selected, yo_vals, yi_vals, ys_vals)
        flight_numbers = writer._assign_flight_numbers(selected)

        routes_path = os.path.join(scheme_dir, 'q2-routes.csv')
        writer._write_routes(selected, flight_numbers, routes_path)

        assign_path = os.path.join(scheme_dir, 'q2-assignments.csv')
        writer._write_assignments(person_assignments, flight_numbers, assign_path)

        metrics_path = os.path.join(scheme_dir, 'q2-metrics.txt')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write(f"方案: {desc} ({scheme_name})\n")
            f.write(f"权重: w_time={weights[0]}, w_transit={weights[1]}, "
                    f"w_util={weights[2]}, w_fuel={weights[3]}\n\n")
            f.write(f"总飞机使用时间: {metrics['total_time']} 分钟 "
                    f"({metrics['total_time']/60:.6f} 小时)\n")
            f.write(f"人员总在途时间: {metrics['total_transit']} 分钟 "
                    f"({metrics['total_transit']/60:.6f} 小时)\n")
            f.write(f"总架次数: {metrics['total_flights']}\n")
            f.write(f"总燃油消耗量: {fmt6(metrics['total_fuel'])} kg\n")
            f.write(f"座位利用率: {metrics['seat_util']*100:.6f}%\n")
            f.write(f"  客座公里: {fmt6(metrics['pass_km'])}\n")
            f.write(f"  可用座公里: {fmt6(metrics['seat_km'])}\n")
            f.write(f"  浪费座公里: {fmt6(metrics['wasted'])}\n")
            f.write(f"求解时间: {metrics['solve_time']:.1f}s\n")

        all_results.append((scheme_name, weights, metrics))

    # 6. 汇总输出
    print("\n[6] 汇总输出...")
    comparison_csv = os.path.join(output_base, 'q2-weight-comparison.csv')
    SensitivityAnalyzer.write_comparison_csv(all_results, comparison_csv, 'Q2')

    comparison_txt = os.path.join(output_base, 'q2-weight-comparison.txt')
    SensitivityAnalyzer.write_comparison_txt(all_results, comparison_txt, 'Q2', norm)

    # 7. 生成图表
    print("\n[7] 生成敏感性分析图表...")
    SensitivityAnalyzer.plot_sensitivity(all_results, FIGURES_DIR_Q2, 'Q2')

    print("\n" + "=" * 60)
    print("Q2加权单目标法完成!")
    print("=" * 60)
    return all_results


# ==================== 主入口 ====================
def main():
    """主函数：依次运行Q1和Q2的加权单目标法"""
    print("加权单目标优化法 — 问题1和问题2")
    print("将字典序多目标转化为加权单目标，研究不同权重比例的影响\n")

    # Q1
    q1_results = run_q1_weighted()

    print("\n\n")

    # Q2
    q2_results = run_q2_weighted()

    # 总结
    print("\n\n" + "=" * 60)
    print("全部完成!")
    print("=" * 60)
    print(f"\nQ1: {sum(1 for _,_,m in q1_results if m is not None)}/8 组方案成功")
    print(f"Q2: {sum(1 for _,_,m in q2_results if m is not None)}/8 组方案成功")
    print(f"\n结果文件:")
    print(f"  Q1: output/Problem1_weighted/")
    print(f"  Q2: output/Problem2_weighted/")


if __name__ == '__main__':
    main()
