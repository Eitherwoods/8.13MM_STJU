# -*- coding: utf-8 -*-
"""
文件: lagrangian_bound.py
功能: C9 拉格朗日松弛下界 — LP松弛下界 + 拉格朗日次梯度下界
项目: 2026策联杯 B题 - 海上油田人员直升机运载计划编排

方法说明:
  1. LP松弛下界: 将MIP中所有二元/整数变量松弛为连续变量，求解LP得到下界
  2. 拉格朗日下界: 松弛机队约束(按机场×机型×时间槽分解)，引入乘子λ≥0，
     用次梯度法优化乘子，L(λ)为原MIP的有效下界

拉格朗日松弛公式:
  原约束: 对每个(airport, atype, slot): sum_{r∈R^{a,t}_τ} x_r <= M_{a,t}
  松弛后: L(λ) = min { weighted_obj + sum λ_{a,t,τ} * (sum x_r - M_{a,t}) }
  次梯度: g_{a,t,τ} = sum_{r∈R^{a,t}_τ} x_r - M_{a,t}
  步长:   α_k = β * (UB - L(λ_k)) / ||g||^2, β从1.0递减
"""

import sys
import os
import time
from datetime import datetime, date
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional

# ==================== 模块路径 ====================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

import coptpy as cp
from config import AIRCRAFT, AIRCRAFT_TYPES, AIRPORTS, DATA_DIR
from base_models import DataLoader
from problem3_solution import (
    Q3DataLoader, Q3DemandDayAllocator, Q3RouteGenerator,
    Q3ScheduleHelper, Q3DailySolver,
    FLEET, PLANNING_DATES,
    OPERATION_START_HOUR, RETURN_DEADLINE_HOUR,
    TURNAROUND_MINUTES, TIME_SLOT_MINUTES,
)
from problem2_solution import Q2RoutePreprocessor


class LagrangianBoundAnalyzer:
    """
    拉格朗日松弛下界分析器

    对单日MIP模型计算两种下界:
    1. LP松弛下界 — 全变量连续化
    2. 拉格朗日下界 — 松弛机队约束，次梯度法优化乘子
    """

    # 加权目标权重（与 Q3DailySolver 保持一致）
    W_TIME = Q3DailySolver.W_TIME          # 总时间权重
    W_TRANSIT = Q3DailySolver.W_TRANSIT    # 在途时间权重
    W_UTIL = Q3DailySolver.W_UTIL          # 利用率项权重
    W_FUEL = Q3DailySolver.W_FUEL          # 燃油权重
    LAMBDA_UTIL = Q3DailySolver.LAMBDA_INIT  # Dinkelbach利用率乘子(固定)

    def __init__(self, routes_info: List[Dict], day_demands: List[Dict],
                 day: date, distances: Dict[str, Dict[str, float]]):
        """
        初始化下界分析器

        参数:
            routes_info: 候选路径信息列表（由Q3RouteGenerator生成）
            day_demands: 当日需求列表
            day: 目标日期
            distances: 距离矩阵
        """
        self.routes_info = routes_info
        self.day_demands = day_demands
        self.day = day
        self.distances = distances

        # 预处理所有路径
        for ri in routes_info:
            if 'precomputed' not in ri:
                ri['precomputed'] = Q2RoutePreprocessor.preprocess(ri['route'], distances)

        # 分类当日需求
        airport_set = set(AIRPORTS) | {'LAND'}
        self.outbound = [d for d in day_demands
                         if d['origin_id'] in airport_set and d['destination_id'].startswith('F')]
        self.inbound = [d for d in day_demands
                        if d['origin_id'].startswith('F') and d['destination_id'] in airport_set]
        self.shuttle = [d for d in day_demands
                        if d['origin_id'].startswith('F') and d['destination_id'].startswith('F')]

        # 按设施分组需求
        self.ob_by_fac = defaultdict(list)
        for d in self.outbound:
            self.ob_by_fac[d['destination_id']].append(d)
        self.ib_by_fac = defaultdict(list)
        for d in self.inbound:
            self.ib_by_fac[d['origin_id']].append(d)
        self.sh_by_od = defaultdict(list)
        for d in self.shuttle:
            self.sh_by_od[(d['origin_id'], d['destination_id'])].append(d)

        # 计算兼容性缓存
        self._compute_compatibility()

        # 构建机队约束结构
        self._build_fleet_structure()

        # 计算x变量的基础目标系数
        self._compute_base_x_coeffs()

    def _compute_compatibility(self):
        """计算每条路径与需求的兼容性计数"""
        airport_set = set(AIRPORTS) | {'LAND'}
        self.compat_ob = defaultdict(dict)  # compat_ob[ri_idx][fac] = count
        self.compat_ib = defaultdict(dict)
        self.compat_sh = defaultdict(dict)

        for ri_idx, ri in enumerate(self.routes_info):
            for d in ri['compatible_demands']:
                o, de = d['origin_id'], d['destination_id']
                if o in airport_set and de.startswith('F'):
                    self.compat_ob[ri_idx][de] = self.compat_ob[ri_idx].get(de, 0) + 1
                elif o.startswith('F') and de in airport_set:
                    self.compat_ib[ri_idx][o] = self.compat_ib[ri_idx].get(o, 0) + 1
                elif o.startswith('F') and de.startswith('F'):
                    self.compat_sh[ri_idx][(o, de)] = \
                        self.compat_sh[ri_idx].get((o, de), 0) + 1

    def _build_fleet_structure(self):
        """
        构建机队约束结构

        机队约束: 对每个(airport, atype, slot)，
        若覆盖该时间槽的路径数 > 机队规模M，则 sum(x_r) <= M
        """
        # 按机场+机型分组路径
        self.fleet_groups = defaultdict(list)
        for ri_idx, ri in enumerate(self.routes_info):
            r = ri['route']
            self.fleet_groups[(r.airport, r.aircraft_type)].append(ri_idx)

        # 构建机队约束列表: [(key, route_indices, max_count), ...]
        self.fleet_constraints = []
        for (airport, atype), ri_indices in self.fleet_groups.items():
            max_count = FLEET[airport][atype]
            slot_to_routes = defaultdict(list)
            for ri_idx in ri_indices:
                for slot in self.routes_info[ri_idx]['time_slots']:
                    slot_to_routes[slot].append(ri_idx)
            for slot, ri_list in slot_to_routes.items():
                # 仅当路径数超过机队规模时才需要约束
                if len(ri_list) > max_count:
                    self.fleet_constraints.append(
                        ((airport, atype, slot), ri_list, max_count)
                    )

        # 构建路径->约束映射（用于拉格朗日目标系数调整）
        self.route_to_fleet_constraints = defaultdict(list)
        for cidx, (key, ri_list, max_count) in enumerate(self.fleet_constraints):
            for ri_idx in ri_list:
                self.route_to_fleet_constraints[ri_idx].append(cidx)

        print(f"  机队约束: {len(self.fleet_constraints)}条 "
              f"(覆盖{len(set(k[0] for k, _, _ in self.fleet_constraints))}组机场×机型)")

    def _compute_base_x_coeffs(self):
        """
        计算x变量的基础目标系数（不含拉格朗日项）

        x的目标系数 = W_TIME * time + W_UTIL * λ_util * seat_km + W_FUEL * fuel
        """
        self.base_x_coeff = {}
        for idx, ri in enumerate(self.routes_info):
            r = ri['route']
            coeff = (
                self.W_TIME * r.time
                + self.W_UTIL * self.LAMBDA_UTIL * r.capacity * r.total_distance
                + self.W_FUEL * r.fuel
            )
            self.base_x_coeff[idx] = coeff

    def _get_y_coeff(self, ri_idx: int, var_type: str, f1: str, f2: str = None) -> float:
        """
        计算yo/yi/ys变量的目标系数

        参数:
            ri_idx: 路径索引
            var_type: 'yo'(出海) / 'yi'(海返) / 'ys'(穿梭)
            f1: 设施1
            f2: 设施2（仅穿梭）

        返回:
            目标系数 = W_TRANSIT * 在途时间 - W_UTIL * 客座公里
        """
        pre = self.routes_info[ri_idx]['precomputed']
        if var_type == 'yo':
            return (self.W_TRANSIT * pre.get('time_to', {}).get(f1, 0)
                    - self.W_UTIL * pre.get('dist_to', {}).get(f1, 0))
        elif var_type == 'yi':
            return (self.W_TRANSIT * pre.get('time_from', {}).get(f1, 0)
                    - self.W_UTIL * pre.get('dist_from', {}).get(f1, 0))
        elif var_type == 'ys':
            si = pre.get('shuttle_info', {}).get((f1, f2), {})
            return (self.W_TRANSIT * si.get('time', 0)
                    - self.W_UTIL * si.get('dist', 0))
        return 0.0

    def _build_model(self, relax_integrality: bool = False,
                     include_fleet: bool = True,
                     lam_multipliers: Optional[Dict] = None) -> Tuple:
        """
        构建COPT模型

        参数:
            relax_integrality: True=LP松弛(所有变量连续), False=保持整数
            include_fleet: True=包含机队约束, False=不含(用于拉格朗日子问题)
            lam_multipliers: 拉格朗日乘子 {(airport,atype,slot): λ}
                            若提供且include_fleet=False，则乘子项进入目标函数

        返回:
            (model, x_dict, yo_dict, yi_dict, ys_dict)
        """
        env = cp.Envr()
        model = env.createModel('Q3_Bound')

        # 变量类型
        vtype_x = cp.COPT.CONTINUOUS if relax_integrality else cp.COPT.BINARY
        vtype_y = cp.COPT.CONTINUOUS if relax_integrality else cp.COPT.INTEGER

        # --- 创建变量 ---
        x = {}
        yo, yi, ys = {}, {}, {}

        for ri_idx, ri in enumerate(self.routes_info):
            x[ri_idx] = model.addVar(vtype=vtype_x, name=f'x_{ri_idx}')
            pre = ri['precomputed']
            for f in pre['facility_pos']:
                if f in self.ob_by_fac and self.compat_ob[ri_idx].get(f, 0) > 0:
                    yo[(ri_idx, f)] = model.addVar(
                        vtype=vtype_y, lb=0, ub=self.compat_ob[ri_idx][f],
                        name=f'yo_{ri_idx}_{f}')
                if f in self.ib_by_fac and self.compat_ib[ri_idx].get(f, 0) > 0:
                    yi[(ri_idx, f)] = model.addVar(
                        vtype=vtype_y, lb=0, ub=self.compat_ib[ri_idx][f],
                        name=f'yi_{ri_idx}_{f}')
            for (f1, f2) in pre['shuttle_info']:
                if (f1, f2) in self.sh_by_od and self.compat_sh[ri_idx].get((f1, f2), 0) > 0:
                    ys[(ri_idx, f1, f2)] = model.addVar(
                        vtype=vtype_y, lb=0, ub=self.compat_sh[ri_idx][(f1, f2)],
                        name=f'ys_{ri_idx}_{f1}_{f2}')

        # --- 约束1: 需求覆盖 (等式) ---
        for f, dlist in self.ob_by_fac.items():
            terms = [yo[(ri, f)] for ri in range(len(self.routes_info)) if (ri, f) in yo]
            if terms:
                model.addConstr(cp.quicksum(terms) == len(dlist))
        for f, dlist in self.ib_by_fac.items():
            terms = [yi[(ri, f)] for ri in range(len(self.routes_info)) if (ri, f) in yi]
            if terms:
                model.addConstr(cp.quicksum(terms) == len(dlist))
        for (f1, f2), dlist in self.sh_by_od.items():
            terms = [ys[(ri, f1, f2)]
                     for ri in range(len(self.routes_info)) if (ri, f1, f2) in ys]
            if terms:
                model.addConstr(cp.quicksum(terms) == len(dlist))

        # --- 约束2: 起点感知 (不等式) ---
        for f, dlist in self.ob_by_fac.items():
            by_origin = defaultdict(int)
            for d in dlist:
                by_origin[d['origin_id']] += 1
            for origin, cnt in by_origin.items():
                if origin != 'LAND' and cnt > 0:
                    terms = [yo[(ri, f)] for ri in range(len(self.routes_info))
                             if (ri, f) in yo and self.routes_info[ri]['route'].airport == origin]
                    if terms:
                        model.addConstr(cp.quicksum(terms) >= cnt)
        for f, dlist in self.ib_by_fac.items():
            by_dest = defaultdict(int)
            for d in dlist:
                by_dest[d['destination_id']] += 1
            for dest, cnt in by_dest.items():
                if dest != 'LAND' and cnt > 0:
                    terms = [yi[(ri, f)] for ri in range(len(self.routes_info))
                             if (ri, f) in yi and self.routes_info[ri]['route'].airport == dest]
                    if terms:
                        model.addConstr(cp.quicksum(terms) >= cnt)

        # --- 约束3: 航段容量 (链接y与x) ---
        for ri_idx, ri in enumerate(self.routes_info):
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

        # --- 约束4: 机队约束 (可选) ---
        if include_fleet:
            for (key, ri_list, max_count) in self.fleet_constraints:
                model.addConstr(cp.quicksum(x[ri] for ri in ri_list) <= max_count)

        # --- 目标函数 ---
        # x部分: 基础系数 + 拉格朗日乘子项
        x_terms = []
        for idx in range(len(self.routes_info)):
            coeff = self.base_x_coeff[idx]
            if lam_multipliers is not None and not include_fleet:
                for cidx in self.route_to_fleet_constraints.get(idx, []):
                    key = self.fleet_constraints[cidx][0]
                    coeff += lam_multipliers.get(key, 0.0)
            x_terms.append(coeff * x[idx])

        # yo/yi/ys部分: 系数不随拉格朗日变化
        y_terms = []
        for (ri_idx, f), var in yo.items():
            y_terms.append(self._get_y_coeff(ri_idx, 'yo', f) * var)
        for (ri_idx, f), var in yi.items():
            y_terms.append(self._get_y_coeff(ri_idx, 'yi', f) * var)
        for (ri_idx, f1, f2), var in ys.items():
            y_terms.append(self._get_y_coeff(ri_idx, 'ys', f1, f2) * var)

        all_terms = x_terms + y_terms
        obj_expr = cp.quicksum(all_terms) if all_terms else 0
        model.setObjective(obj_expr, sense=cp.COPT.MINIMIZE)

        return model, x, yo, yi, ys

    def _get_model_bound(self, model) -> Optional[float]:
        """
        安全获取模型bestbound（COPT下界）

        COPT Python API通过 model.getAttr(cp.COPT.Attr.BestBnd) 获取下界，
        而非 model.bestbound 属性。
        """
        try:
            return model.getAttr(cp.COPT.Attr.BestBnd)
        except Exception:
            return None

    def solve_lp_relaxation(self, time_limit: int = 120) -> Optional[float]:
        """
        求解LP松弛下界

        将所有二元/整数变量松弛为连续变量，求解LP。
        LP最优值是MIP最优值的有效下界。

        参数:
            time_limit: 求解时间限制(秒)

        返回:
            LP松弛下界值，或None(求解失败)
        """
        print("\n===== LP松弛下界 =====")
        print("  构建LP模型(全变量连续化)...")

        model, x, yo, yi, ys = self._build_model(
            relax_integrality=True, include_fleet=True, lam_multipliers=None
        )
        model.setParam(cp.COPT.Param.LogToConsole, 0)
        model.setParam(cp.COPT.Param.TimeLimit, time_limit)

        t0 = time.time()
        model.solve()
        t1 = time.time()

        if model.status not in (cp.COPT.OPTIMAL, cp.COPT.TIMEOUT,
                                 cp.COPT.NODELIMIT, cp.COPT.UNBOUNDED):
            print(f"  LP求解失败: 状态={model.status}")
            return None

        try:
            lp_bound = model.objval
        except Exception:
            print(f"  无法获取LP目标值")
            return None

        print(f"  LP松弛下界: {lp_bound:.4f}")
        print(f"  求解耗时: {t1 - t0:.1f}s, 状态={model.status}")
        return lp_bound

    def solve_mip(self, time_limit: int = 120, rel_gap: float = 0.15) -> Tuple[Optional[float], Optional[float]]:
        """
        求解MIP（含机队约束，保持整数）

        参数:
            time_limit: 求解时间限制(秒)
            rel_gap: 相对间隙容忍度

        返回:
            (MIP目标值, COPT内置下界)，或(None, None)
        """
        print("\n===== MIP求解 =====")
        print("  构建MIP模型(保持整数约束)...")

        model, x, yo, yi, ys = self._build_model(
            relax_integrality=False, include_fleet=True, lam_multipliers=None
        )
        model.setParam(cp.COPT.Param.LogToConsole, 0)
        model.setParam(cp.COPT.Param.TimeLimit, time_limit)
        model.setParam(cp.COPT.Param.RelGap, rel_gap)

        t0 = time.time()
        model.solve()
        t1 = time.time()

        if not Q3DailySolver._has_solution(model):
            print(f"  MIP求解失败: 状态={model.status}")
            return None, None

        mip_obj = model.objval
        mip_bestbound = self._get_model_bound(model)

        print(f"  MIP目标值:   {mip_obj:.4f}")
        if mip_bestbound is not None:
            gap_copt = (mip_obj - mip_bestbound) / abs(mip_obj) * 100 if abs(mip_obj) > 1e-10 else 0
            print(f"  COPT下界:    {mip_bestbound:.4f}  (间隙={gap_copt:.2f}%)")
        print(f"  求解耗时: {t1 - t0:.1f}s, 状态={model.status}")

        return mip_obj, mip_bestbound

    def solve_lagrangian(self, ub: float, max_iters: int = 25,
                         time_limit_per_iter: int = 30,
                         beta_init: float = 1.0, beta_min: float = 0.1,
                         beta_decay: float = 0.95) -> Tuple[Optional[float], List[Dict]]:
        """
        拉格朗日松弛下界（次梯度法）

        松弛机队约束，引入乘子λ≥0，用次梯度法优化。
        构建模型一次，每次迭代仅更新目标函数中的x系数后重求解。

        参数:
            ub: 上界（MIP解值），用于计算Polyak步长
            max_iters: 最大迭代次数
            time_limit_per_iter: 每次子问题求解时间限制(秒)
            beta_init: 初始步长因子β
            beta_min: β最小值
            beta_decay: β衰减率

        返回:
            (最佳拉格朗日下界, 迭代历史列表)
        """
        print("\n===== 拉格朗日松弛下界 =====")
        print(f"  机队约束数: {len(self.fleet_constraints)}")
        print(f"  最大迭代: {max_iters}, 每次时限: {time_limit_per_iter}s")

        if not self.fleet_constraints:
            print("  无机队约束需要松弛，拉格朗日下界等于无约束MIP下界")
            return None, []

        # ===== 构建子问题模型一次（无机队约束）=====
        print("  构建拉格朗日子问题模型(无机队约束)...")
        t_build0 = time.time()
        model, x, yo, yi, ys = self._build_model(
            relax_integrality=False,
            include_fleet=False,
            lam_multipliers=None  # 初始无乘子
        )
        model.setParam(cp.COPT.Param.LogToConsole, 0)
        model.setParam(cp.COPT.Param.TimeLimit, time_limit_per_iter)
        model.setParam(cp.COPT.Param.RelGap, 0.01)
        t_build1 = time.time()
        print(f"  模型构建完成, 耗时 {t_build1 - t_build0:.1f}s")

        # 预计算yo/yi/ys目标项（不随迭代变化）
        y_terms = []
        for (ri_idx, f), var in yo.items():
            y_terms.append(self._get_y_coeff(ri_idx, 'yo', f) * var)
        for (ri_idx, f), var in yi.items():
            y_terms.append(self._get_y_coeff(ri_idx, 'yi', f) * var)
        for (ri_idx, f1, f2), var in ys.items():
            y_terms.append(self._get_y_coeff(ri_idx, 'ys', f1, f2) * var)
        y_obj_expr = cp.quicksum(y_terms) if y_terms else 0

        # 预计算x变量列表和基础系数
        n_routes = len(self.routes_info)
        x_vars = [x[idx] for idx in range(n_routes)]
        base_coeffs = [self.base_x_coeff[idx] for idx in range(n_routes)]

        # 预计算每条路径对应的机队约束key列表（用于乘子项）
        route_lam_keys = []
        for idx in range(n_routes):
            keys = []
            for cidx in self.route_to_fleet_constraints.get(idx, []):
                keys.append(self.fleet_constraints[cidx][0])
            route_lam_keys.append(keys)

        # 预计算机队约束数据（用于次梯度和常数项）
        fleet_data = []  # [(key, ri_list, max_count), ...]
        for (key, ri_list, max_count) in self.fleet_constraints:
            fleet_data.append((key, ri_list, max_count))

        # 初始化乘子 λ = 0
        lam = {key: 0.0 for (key, _, _) in fleet_data}

        best_bound = -float('inf')
        history = []
        beta = beta_init
        no_improve_count = 0

        for iteration in range(max_iters):
            # ===== 构建新目标函数 =====
            # x系数 = 基础系数 + sum(λ * 1) for each fleet constraint containing this route
            x_terms = []
            for idx in range(n_routes):
                coeff = base_coeffs[idx]
                for key in route_lam_keys[idx]:
                    coeff += lam[key]
                x_terms.append(coeff * x_vars[idx])

            obj_expr = cp.quicksum(x_terms) + y_obj_expr if x_terms else y_obj_expr
            model.setObjective(obj_expr, sense=cp.COPT.MINIMIZE)

            # ===== 求解 =====
            t0 = time.time()
            model.solve()
            t1 = time.time()

            if not Q3DailySolver._has_solution(model):
                print(f"  迭代{iteration + 1:>3}: 子问题无解(状态={model.status}), β减小")
                beta = max(beta_min, beta * 0.5)
                no_improve_count += 1
                if no_improve_count >= 5:
                    print("  连续5次无改善/无解，终止")
                    break
                continue

            # 子问题目标值（不含常数项）
            sub_obj = model.objval
            sub_bestbound = self._get_model_bound(model)

            # 常数项: -sum(λ * M)
            const_term = 0.0
            for (key, _, max_count) in fleet_data:
                const_term -= lam[key] * max_count

            # 拉格朗日函数值
            # L(λ) = 子问题最优 + 常数项
            # model.objval 是子问题上界（最小化），model.bestbound 是子问题下界
            # 有效拉格朗日下界 = bestbound + 常数项 (<= L(λ) <= MIP最优)
            # L值估计 = objval + 常数项 (>= L(λ), 用于步长计算)
            lag_val = sub_obj + const_term  # L(λ)的上界估计，用于步长
            if sub_bestbound is not None:
                lag_bound = sub_bestbound + const_term  # 有效下界
            else:
                # bestbound不可用时，无法提供有效下界
                lag_bound = None

            # 更新最佳下界（仅使用有效下界）
            if lag_bound is not None:
                if lag_bound > best_bound:
                    best_bound = lag_bound
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1

            # 提取x值用于次梯度计算
            x_vals = {}
            for idx in range(n_routes):
                try:
                    x_vals[idx] = x[idx].x
                except Exception:
                    x_vals[idx] = 0.0

            # 计算次梯度: g_{a,t,τ} = sum(x_r) - M_{a,t}
            subgrad = {}
            subgrad_norm_sq = 0.0
            n_violated = 0
            for (key, ri_list, max_count) in fleet_data:
                g = sum(x_vals.get(ri, 0.0) for ri in ri_list) - max_count
                subgrad[key] = g
                subgrad_norm_sq += g * g
                if g > 1e-6:
                    n_violated += 1

            subgrad_norm = subgrad_norm_sq ** 0.5

            history.append({
                'iter': iteration + 1,
                'lag_val': lag_val,
                'lag_bound': lag_bound,
                'best_bound': best_bound if best_bound > -float('inf') else None,
                'subgrad_norm': subgrad_norm,
                'n_violated': n_violated,
                'n_constraints': len(fleet_data),
                'beta': beta,
                'time': t1 - t0,
                'status': model.status,
            })

            print(f"  迭代{iteration + 1:>3}: L(λ)={lag_val:>12.4f}  "
                  f"下界={lag_bound if lag_bound is not None else float('nan'):>12.4f}  "
                  f"最佳={best_bound if best_bound > -float('inf') else float('nan'):>12.4f}  "
                  f"||g||={subgrad_norm:>8.4f}  违反={n_violated}/{len(fleet_data)}  "
                  f"β={beta:.3f}  {t1 - t0:.1f}s")

            # 收敛判断
            if subgrad_norm < 1e-8:
                print("  次梯度范数接近0，已收敛")
                break

            if no_improve_count >= 8:
                print("  连续8次无改善，终止")
                break

            # Polyak步长: α = β * (UB - L_best) / ||g||^2
            # 使用最佳有效下界（而非上界估计）计算步长，更符合Polyak步长定义
            step_ref = best_bound if best_bound > -float('inf') else lag_val
            gap = ub - step_ref
            if gap <= 0:
                # 下界已接近上界，减小步长
                gap = abs(ub) * 0.01
            step = beta * gap / subgrad_norm_sq

            # 更新乘子: λ = max(0, λ + α * g)
            for key in lam:
                lam[key] = max(0.0, lam[key] + step * subgrad[key])

            # 衰减β
            beta = max(beta_min, beta * beta_decay)

        if best_bound > -float('inf'):
            print(f"\n  拉格朗日最佳下界: {best_bound:.4f}")
        else:
            print(f"\n  拉格朗日下界: 未能获取有效下界")
            best_bound = None

        return best_bound, history

    def compute_gap(self, mip_obj: float, bound: float) -> float:
        """
        计算最优性间隙

        间隙 = (MIP解 - 下界) / |MIP解| * 100%
        """
        if abs(mip_obj) < 1e-10:
            return 0.0
        return (mip_obj - bound) / abs(mip_obj) * 100


def print_summary(mip_obj: float, lp_bound: Optional[float],
                  lag_bound: Optional[float], lag_history: List[Dict],
                  copt_bound: Optional[float]):
    """
    打印下界对比汇总

    参数:
        mip_obj: MIP目标值
        lp_bound: LP松弛下界
        lag_bound: 拉格朗日下界
        lag_history: 拉格朗日迭代历史
        copt_bound: COPT内置下界
    """
    print("\n" + "=" * 70)
    print("  下界对比汇总")
    print("=" * 70)

    print(f"\n  MIP当前解值:       {mip_obj:>14.4f}")

    if copt_bound is not None:
        gap_copt = (mip_obj - copt_bound) / abs(mip_obj) * 100 if abs(mip_obj) > 1e-10 else 0
        print(f"  COPT内置下界:      {copt_bound:>14.4f}  间隙={gap_copt:>8.2f}%")

    if lp_bound is not None:
        gap_lp = (mip_obj - lp_bound) / abs(mip_obj) * 100 if abs(mip_obj) > 1e-10 else 0
        print(f"  LP松弛下界:        {lp_bound:>14.4f}  间隙={gap_lp:>8.2f}%")

    if lag_bound is not None:
        gap_lag = (mip_obj - lag_bound) / abs(mip_obj) * 100 if abs(mip_obj) > 1e-10 else 0
        print(f"  拉格朗日下界:      {lag_bound:>14.4f}  间隙={gap_lag:>8.2f}%")

    # 拉格朗日迭代过程表
    if lag_history:
        print("\n  拉格朗日迭代过程:")
        print(f"  {'迭代':>4}  {'L(λ)':>12}  {'有效下界':>12}  {'最佳下界':>12}  "
              f"{'||g||':>10}  {'违反数':>6}  {'β':>8}  {'耗时(s)':>8}")
        print("  " + "-" * 90)
        for h in lag_history:
            lb_str = f"{h['lag_bound']:>12.4f}" if h['lag_bound'] is not None else f"{'N/A':>12}"
            bb_str = f"{h['best_bound']:>12.4f}" if h['best_bound'] is not None and h['best_bound'] > -float('inf') else f"{'N/A':>12}"
            print(f"  {h['iter']:>4}  {h['lag_val']:>12.4f}  {lb_str}  "
                  f"{bb_str}  {h['subgrad_norm']:>10.4f}  "
                  f"{h['n_violated']:>3}/{h['n_constraints']:<2}  "
                  f"{h['beta']:>8.4f}  {h['time']:>8.1f}")

    print("\n" + "=" * 70)


def main():
    """主函数: 构建单日问题并计算LP松弛下界与拉格朗日下界"""
    print("=" * 70)
    print("  C9: 拉格朗日松弛下界分析")
    print("  目标日期: 2026-08-03")
    print("=" * 70)

    # ========== 1. 加载数据 ==========
    print("\n[1] 加载数据...")
    loader = Q3DataLoader()
    distances = loader.load_distances()
    all_demands = loader.load_demands_q3()
    print(f"  总需求: {len(all_demands)}")

    # 分离临时和非临时需求
    non_temp = [d for d in all_demands if d['task_type'] != 'temporary']
    print(f"  非临时需求(阶段一): {len(non_temp)}")

    # ========== 2. 需求-日期分配 ==========
    print("\n[2] 需求-日期分配...")
    allocation, daily_load = Q3DemandDayAllocator.allocate(non_temp, PLANNING_DATES)
    for d in non_temp:
        d['_assigned_day'] = allocation[d['person_id']]
    for d in PLANNING_DATES:
        print(f"  {d}: {daily_load[d]}人")

    # ========== 3. 选取目标日期 ==========
    target_day = date(2026, 8, 3)
    day_demands = [d for d in non_temp if allocation[d['person_id']] == target_day]
    print(f"\n[3] 目标日期: {target_day} ({len(day_demands)}人)")

    # ========== 4. 生成候选路径 ==========
    print("\n[4] 生成候选路径...")
    generator = Q3RouteGenerator(distances)
    routes_info = generator.generate_daily_routes(day_demands, target_day, route_id_start=1)
    print(f"  候选路径: {len(routes_info)}")

    if not routes_info:
        print("  无候选路径，退出")
        return

    # ========== 5. 创建分析器 ==========
    print("\n[5] 构建下界分析器...")
    analyzer = LagrangianBoundAnalyzer(routes_info, day_demands, target_day, distances)

    # ========== 6. 求解MIP ==========
    print("\n[6] 求解MIP(上界)...")
    mip_obj, copt_bound = analyzer.solve_mip(time_limit=120, rel_gap=0.15)
    if mip_obj is None:
        print("  MIP求解失败，退出")
        return

    # ========== 7. LP松弛下界 ==========
    print("\n[7] 计算LP松弛下界...")
    lp_bound = analyzer.solve_lp_relaxation(time_limit=120)

    # ========== 8. 拉格朗日下界 ==========
    print("\n[8] 计算拉格朗日松弛下界...")
    lag_bound, lag_history = analyzer.solve_lagrangian(
        ub=mip_obj,
        max_iters=25,
        time_limit_per_iter=30,
    )

    # ========== 9. 汇总输出 ==========
    print_summary(mip_obj, lp_bound, lag_bound, lag_history, copt_bound)


if __name__ == '__main__':
    main()
