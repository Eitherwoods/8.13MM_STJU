# -*- coding: utf-8 -*-
"""
文件: two_stage_solver.py
功能: 问题1/问题2 两阶段组合优化求解器（ε-容差约束 + AHP加权）
项目: 2026策联杯 B题 - 海上油田人员直升机运载计划编排

方法:
  阶段一: 最小化总飞机使用时间 T，得到最佳已知值 T*（主导目标优先）
  阶段二: 加入容差约束 T <= (1+ε)·T*，在容差带内最小化 AHP 加权归一化的
          二级目标（人员总在途时间 τ、浪费座公里 W、总燃油消耗量 F）

设计动机:
  1. 题目只要求"在最小化总飞机使用时间的基础上尽可能优化其他因素"，
     未给出其他指标的重要程度排序，直接字典序分层缺乏依据；
  2. 完全加权单目标会稀释主导目标（时间）的地位；
  3. 两阶段 ε-容差法既保持时间的主导地位，又在允许的小幅退化范围内
     用加权单目标系统优化其余指标，权重由层次分析法（AHP）确定并通过一致性检验。

创新点:
  1. 两阶段 ε-容差组合优化：时间先优化、容差带内再优化其他指标
  2. AHP 层次分析法确定二级目标权重（特征向量法 + 一致性检验 CR<0.1）
  3. 以浪费座公里 W = seat_km - pass_km 作为座位利用率的线性代理，
     阶段二无需 Dinkelbach 迭代，保持线性 MIP
"""

import csv
import os
import sys
import time as time_module

import numpy as np
import coptpy as cp

# ==================== 模块路径 ====================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VIBECODING_DIR = os.path.dirname(_THIS_DIR)
if _VIBECODING_DIR not in sys.path:
    sys.path.insert(0, _VIBECODING_DIR)
# 同目录导入 problem1_solution/problem2_solution/weighted_solver
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from config import (
    PROJECT_DIR, OUTPUT_DIR, OUTPUT_DIR_Q2, fmt6, setup_matplotlib
)
from base_models import DataLoader

from problem1_solution import RouteGenerator, OutputWriter
from problem2_solution import (
    Q2DataLoader, Q2RoutePreprocessor, Q2RouteGenerator, Q2OutputWriter
)
from weighted_solver import (
    WeightedSolverQ1, WeightedSolverQ2,
    WEIGHT_SCHEMES, WEIGHT_DESCRIPTIONS
)


# ==================== 层次分析法 ====================
class AHPAnalyzer:
    """
    层次分析法（AHP）权重计算器

    采用特征向量法：求判断矩阵的最大特征值及其特征向量，
    归一化后作为权重；同时计算一致性指标 CI 与一致性比率 CR，
    CR < 0.1 时认为判断矩阵通过一致性检验。
    """

    # 随机一致性指标 RI（Saaty 表，n=1..10）
    RI_TABLE = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12,
                6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

    @staticmethod
    def compute(matrix):
        """
        计算判断矩阵的权重与一致性指标

        参数:
            matrix: n×n 判断矩阵（Saaty 1~9 标度）

        返回:
            dict: weights(归一化权重), lam_max, ci, ri, cr
        """
        a = np.array(matrix, dtype=float)
        n = a.shape[0]
        # 特征向量法：最大特征值对应的特征向量归一化即为权重
        eig_vals, eig_vecs = np.linalg.eig(a)
        idx = int(np.argmax(eig_vals.real))
        lam_max = eig_vals[idx].real
        weights = eig_vecs[:, idx].real
        weights = weights / weights.sum()
        # 一致性检验
        ci = (lam_max - n) / (n - 1)
        ri = AHPAnalyzer.RI_TABLE.get(n, 1.49)
        cr = ci / ri if ri > 0 else 0.0
        return {'weights': weights, 'lam_max': lam_max,
                'ci': ci, 'ri': ri, 'cr': cr}


# AHP 判断矩阵：二级目标（在途时间 τ、座位利用率 util、燃油 F）
# 依据: 题目称在途时间为"重要因素"；座位利用率与燃油均为"其他因素"，
#       其中座位利用率是题目明确要求报告的效率指标，略重于油耗。
# 1-9 Saaty 标度: 1=同等重要, 3=稍微重要, 4=明显重要, 2=介于两者之间
AHP_MATRIX = [
    [1,   3,   4],      # 在途时间 vs 座位利用率: 3，vs 燃油: 4
    [1/3, 1,   2],      # 座位利用率 vs 在途时间: 1/3，vs 燃油: 2
    [1/4, 1/2, 1],      # 燃油 vs 在途时间: 1/4，vs 座位利用率: 1/2
]

# 阶段二容差（相对总飞机使用时间最优值的允许退化比例）
# 实现说明: 显式添加 time <= (1+ε)·T* 硬约束会导致 COPT 启发式找不到可行解，
# 因此阶段二采用 COPT 原生分层目标 setObjectiveN，第一级时间带 reltol=ε，
# 语义与"先求 T*、再在 (1+ε)T* 容差带内优化其他指标"等价。
EPS_TOL = 0.01


class _BaseTwoStage:

    STAGE2_USE_WARM = True      # 阶段二是否使用阶段一解作为 MIP 起点

    # 求解参数默认值（子类可覆盖）
    STAGE1_TIME_LIMIT = 7200
    STAGE2_TIME_LIMIT = 7200
    STAGE1_REL_GAP = 0.01
    STAGE2_REL_GAP = 0.01

    def __init__(self, weights, eps=EPS_TOL):
        self.weights = weights          # (w_transit, w_util, w_fuel)
        self.eps = eps

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

    def _create_model(self, name, stage=1):
        """创建COPT模型并设置通用参数"""
        env = cp.Envr()
        model = env.createModel(name)
        if stage == 1:
            model.setParam(cp.COPT.Param.TimeLimit, self.STAGE1_TIME_LIMIT)
            model.setParam(cp.COPT.Param.RelGap, self.STAGE1_REL_GAP)
        else:
            model.setParam(cp.COPT.Param.TimeLimit, self.STAGE2_TIME_LIMIT)
            model.setParam(cp.COPT.Param.RelGap, self.STAGE2_REL_GAP)
        model.setParam(cp.COPT.Param.LogToConsole, 0)
        return model

    def _create_vars_q1(self, model, routes, demands_by_facility):
        """创建Q1问题的决策变量"""
        x = {}
        y = {}
        for r in routes:
            x[r.route_id] = model.addVar(vtype=cp.COPT.BINARY,
                                         name=f'x_{r.route_id}')
            for f in r.dest_facilities:
                if f in demands_by_facility:
                    y[(r.route_id, f)] = model.addVar(
                        vtype=cp.COPT.INTEGER, lb=0,
                        name=f'y_{r.route_id}_{f}')
        return x, y

    def _create_vars_q2(self, model, routes, precomputed, outbound_demand, 
                        inbound_demand, shuttle_demand):
        """创建Q2问题的决策变量"""
        x, yo, yi, ys = {}, {}, {}, {}
        for r in routes:
            pre = precomputed[r.route_id]
            x[r.route_id] = model.addVar(vtype=cp.COPT.BINARY,
                                         name=f'x_{r.route_id}')
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
        return x, yo, yi, ys

    def _add_demand_constraints_q1(self, model, routes, demands_by_facility, x, y):
        """添加Q1问题的需求约束"""
        # 约束1: 需求满足
        for f, info in demands_by_facility.items():
            terms = [y[(r.route_id, f)] for r in routes
                     if f in r.dest_facilities and (r.route_id, f) in y]
            if terms:
                model.addConstr(cp.quicksum(terms) == info['total'])

        # 约束2: 起点感知
        for f, info in demands_by_facility.items():
            for origin, cnt in info['by_origin'].items():
                if origin != 'LAND' and cnt > 0:
                    terms = [y[(r.route_id, f)] for r in routes
                             if f in r.dest_facilities and r.airport == origin
                             and (r.route_id, f) in y]
                    if terms:
                        model.addConstr(cp.quicksum(terms) >= cnt)

    def _add_demand_constraints_q2(self, model, routes, outbound_demand, 
                                   inbound_demand, shuttle_demand, x, yo, yi, ys):
        """添加Q2问题的需求约束"""
        # 约束1: 三类需求满足
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

        # 约束2: 起点感知
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

    def _add_capacity_constraints_q1(self, model, routes, demands_by_facility, x, y):
        """添加Q1问题的容量约束"""
        for r in routes:
            terms = [y[(r.route_id, f)] for f in r.dest_facilities
                     if (r.route_id, f) in y]
            if terms:
                model.addConstr(cp.quicksum(terms) <= r.capacity * x[r.route_id])

    def _add_capacity_constraints_q2(self, model, routes, precomputed, x, yo, yi, ys):
        """添加Q2问题的航段级容量约束"""
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

    def _extract_solution_q1(self, model, routes, demands_by_facility, x, y):
        """提取Q1问题的解"""
        selected_routes = [r for r in routes if x[r.route_id].x > 0.5]
        assignments = {}
        for r in routes:
            if x[r.route_id].x > 0.5:
                for f in r.dest_facilities:
                    if (r.route_id, f) in y:
                        val = int(round(y[(r.route_id, f)].x))
                        if val > 0:
                            assignments[(r.route_id, f)] = val
        return selected_routes, assignments

    def _extract_solution_q2(self, model, routes, precomputed, x, yo, yi, ys):
        """提取Q2问题的解"""
        selected_routes = [r for r in routes if x[r.route_id].x > 0.5]
        yo_vals, yi_vals, ys_vals = {}, {}, {}
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
        return selected_routes, yo_vals, yi_vals, ys_vals

    def _build_objective_q1(self, model, routes, x, y, norm=None, stage=1,
                            use_explicit_constraint=False, t_star=None):
        """构建Q1问题的目标函数
        
        参数:
            use_explicit_constraint: 是否使用显式时间约束（而非多目标优化）
            t_star: 阶段一最优时间（用于显式约束）
        """
        # 各目标表达式
        time_expr = cp.quicksum(r.time * x[r.route_id] for r in routes)
        transit_terms = []
        for r in routes:
            for f in r.dest_facilities:
                if (r.route_id, f) in y and f in r.transit_times:
                    transit_terms.append(r.transit_times[f] * y[(r.route_id, f)])
        transit_expr = cp.quicksum(transit_terms) if transit_terms else 0

        seatkm_expr = cp.quicksum(r.capacity * r.total_distance * x[r.route_id]
                                  for r in routes)
        passkm_terms = []
        for r in routes:
            for f in r.dest_facilities:
                if (r.route_id, f) in y and f in r.dist_to_facility:
                    passkm_terms.append(r.dist_to_facility[f] * y[(r.route_id, f)])
        passkm_expr = cp.quicksum(passkm_terms) if passkm_terms else 0
        wasted_expr = seatkm_expr - passkm_expr

        fuel_expr = cp.quicksum(r.fuel * x[r.route_id] for r in routes)

        if stage == 1:
            # 阶段一：仅最小化总飞机使用时间
            model.setObjective(time_expr, sense=cp.COPT.MINIMIZE)
            return time_expr, transit_expr, wasted_expr, fuel_expr
        else:
            # 阶段二：根据是否使用显式约束选择不同策略
            w_transit, w_util, w_fuel = self.weights
            
            if use_explicit_constraint and t_star is not None:
                # 显式添加时间约束: time <= (1+eps)*t_star
                max_time = (1 + self.eps) * t_star
                model.addConstr(time_expr <= max_time, name='time_tolerance')
                # 仅优化AHP加权二级目标
                obj = (w_transit * transit_expr / norm['transit']
                       + w_util * wasted_expr / norm['wasted']
                       + w_fuel * fuel_expr / norm['fuel'])
                model.setObjective(obj, sense=cp.COPT.MINIMIZE)
            else:
                # 使用COPT多目标优化
                obj = (w_transit * transit_expr / norm['transit']
                       + w_util * wasted_expr / norm['wasted']
                       + w_fuel * fuel_expr / norm['fuel'])
                model.setObjectiveN(0, time_expr, sense=cp.COPT.MINIMIZE,
                                    priority=10, reltol=self.eps)
                model.setObjectiveN(1, obj, sense=cp.COPT.MINIMIZE, priority=1)
            return time_expr, transit_expr, wasted_expr, fuel_expr
    def _build_objective_q2(self, model, routes, precomputed, x, yo, yi, ys,
                             norm=None, stage=1, use_explicit_constraint=False, t_star=None):
        """构建Q2问题的目标函数
        
        参数:
            use_explicit_constraint: 是否使用显式时间约束（而非多目标优化）
            t_star: 阶段一最优时间（用于显式约束）
        """
        # 各目标表达式
        time_expr = cp.quicksum(r.time * x[r.route_id] for r in routes)
        
        # 在途时间
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
                    transit_terms.append(
                        pre['shuttle_info'][(f1, f2)]['time'] * ys[(r.route_id, f1, f2)])
        transit_expr = cp.quicksum(transit_terms) if transit_terms else 0

        # 浪费座公里
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
                    passkm_terms.append(
                        pre['shuttle_info'][(f1, f2)]['dist'] * ys[(r.route_id, f1, f2)])
        passkm_expr = cp.quicksum(passkm_terms) if passkm_terms else 0
        wasted_expr = seatkm_expr - passkm_expr

        fuel_expr = cp.quicksum(r.fuel * x[r.route_id] for r in routes)

        if stage == 1:
            # 阶段一：仅最小化总飞机使用时间
            model.setObjective(time_expr, sense=cp.COPT.MINIMIZE)
            return time_expr, transit_expr, wasted_expr, fuel_expr
        else:
            # 阶段二：根据是否使用显式约束选择不同策略
            w_transit, w_util, w_fuel = self.weights
            
            if use_explicit_constraint and t_star is not None:
                # 显式添加时间约束: time <= (1+eps)*t_star
                max_time = (1 + self.eps) * t_star
                model.addConstr(time_expr <= max_time, name='time_tolerance')
                # 仅优化AHP加权二级目标
                obj = (w_transit * transit_expr / norm['transit']
                       + w_util * wasted_expr / norm['wasted']
                       + w_fuel * fuel_expr / norm['fuel'])
                model.setObjective(obj, sense=cp.COPT.MINIMIZE)
            else:
                # 使用COPT多目标优化
                obj = (w_transit * transit_expr / norm['transit']
                       + w_util * wasted_expr / norm['wasted']
                       + w_fuel * fuel_expr / norm['fuel'])
                model.setObjectiveN(0, time_expr, sense=cp.COPT.MINIMIZE,
                                    priority=10, reltol=self.eps)
                model.setObjectiveN(1, obj, sense=cp.COPT.MINIMIZE, priority=1)
            return time_expr, transit_expr, wasted_expr, fuel_expr
    def _set_warm_start_q1(self, model, routes, x, y, warm_start):
        """设置Q1问题的MIP热启动"""
        if warm_start is not None:
            s1_selected, s1_assign = warm_start
            selected_ids = {r.route_id for r in s1_selected}
            start_vars, start_vals = [], []
            for r in routes:
                start_vars.append(x[r.route_id])
                start_vals.append(1.0 if r.route_id in selected_ids else 0.0)
                for f in r.dest_facilities:
                    if (r.route_id, f) in y:
                        start_vars.append(y[(r.route_id, f)])
                        start_vals.append(float(s1_assign.get((r.route_id, f), 0)))
            model.setMipStart(start_vars, start_vals)

    def _set_warm_start_q2(self, model, routes, precomputed, x, yo, yi, ys, warm_start):
        """设置Q2问题的MIP热启动"""
        if warm_start is not None:
            s1_selected, s1_yo, s1_yi, s1_ys = warm_start
            selected_ids = {r.route_id for r in s1_selected}
            start_vars, start_vals = [], []
            for r in routes:
                pre = precomputed[r.route_id]
                start_vars.append(x[r.route_id])
                start_vals.append(1.0 if r.route_id in selected_ids else 0.0)
                for f in pre['facility_pos']:
                    if (r.route_id, f) in yo:
                        start_vars.append(yo[(r.route_id, f)])
                        start_vals.append(float(s1_yo.get((r.route_id, f), 0)))
                    if (r.route_id, f) in yi:
                        start_vars.append(yi[(r.route_id, f)])
                        start_vals.append(float(s1_yi.get((r.route_id, f), 0)))
                for (f1, f2) in pre['shuttle_info']:
                    if (r.route_id, f1, f2) in ys:
                        start_vars.append(ys[(r.route_id, f1, f2)])
                        start_vals.append(float(s1_ys.get((r.route_id, f1, f2), 0)))
            model.setMipStart(start_vars, start_vals)

    def _print_stage_result(self, stage_name, metrics):
        """打印阶段结果"""
        print(f"  {stage_name}结果: T = {metrics['total_time']} min, "
              f"τ = {metrics['total_transit']} min, "
              f"F = {fmt6(metrics['total_fuel'])} kg, "
              f"util = {metrics['seat_util']*100:.2f}%")



class TwoStageSolverQ1(_BaseTwoStage):
    """
    问题1 两阶段求解器

    阶段一: 仅以总飞机使用时间为目标求解 MIP，得到 T*
    阶段二: 添加约束 T <= (1+ε)·T*，目标为 AHP 加权归一化二级目标
            min w_tau·(τ/τ0) + w_util·(W/W0) + w_fuel·(F/F0)
    """

    STAGE1_TIME_LIMIT = 7200    # 阶段一求解时间限制(秒)，2小时
    STAGE2_TIME_LIMIT = 7200    # 阶段二求解时间限制(秒)，2小时
    STAGE1_REL_GAP = 0.01       # 阶段一相对间隙
    STAGE2_REL_GAP = 0.01       # 阶段二相对间隙

    def solve(self, routes, demands_by_facility, out_dir, use_explicit_constraint=False):
        """
        完整两阶段求解

        参数:
            routes: 候选路径列表
            demands_by_facility: 按设施分组的需求
            out_dir: 结果输出目录

        返回:
            (stage1_metrics, final_metrics, norm)
        """
        # ---- 阶段一：最小化总飞机使用时间 ----
        print("\n===== 阶段一：最小化总飞机使用时间 =====")
        stage1 = self._solve_stage1(routes, demands_by_facility)
        if stage1 is None:
            return None, None, None
        s1_selected, s1_assign, s1_metrics = stage1
        t_star = s1_metrics['total_time']
        self._print_stage_result("阶段一", s1_metrics)

        # 归一化常数取阶段一最优解处的二级指标值（锚点）
        norm = {
            'transit': s1_metrics['total_transit'],
            'wasted': s1_metrics['wasted'],
            'fuel': s1_metrics['total_fuel'],
        }

        # ---- 阶段二：容差带内 AHP 加权优化 ----
        print("\n===== 阶段二：ε-容差带内 AHP 加权优化 =====")
        warm_start = ((s1_selected, s1_assign)
                      if self.STAGE2_USE_WARM else None)
        stage2 = self._solve_stage2(routes, demands_by_facility,
                                    t_star, norm, out_dir, warm_start)
        if stage2 is None:
            return s1_metrics, None, norm
        final_metrics = stage2
        self._print_stage_result("阶段二", final_metrics)
        return s1_metrics, final_metrics, norm

    def _solve_stage1(self, routes, demands_by_facility):
        """阶段一：仅最小化总飞机使用时间"""
        model = self._create_model('Q1_Stage1_MinTime', stage=1)
        x, y = self._create_vars_q1(model, routes, demands_by_facility)
        
        # 添加约束
        self._add_demand_constraints_q1(model, routes, demands_by_facility, x, y)
        self._add_capacity_constraints_q1(model, routes, demands_by_facility, x, y)
        
        # 构建目标函数
        self._build_objective_q1(model, routes, x, y, stage=1)

        t0 = time_module.time()
        model.solve()
        elapsed = time_module.time() - t0
        if not self._has_solution(model):
            print(f"  错误: 阶段一无可行解，状态={model.status}")
            return None

        selected_routes, assignments = self._extract_solution_q1(
            model, routes, demands_by_facility, x, y)
        metrics = WeightedSolverQ1._calc_metrics(selected_routes, assignments)
        metrics['solve_time'] = elapsed
        return selected_routes, assignments, metrics

    def _solve_stage2(self, routes, demands_by_facility, t_star, norm, out_dir,
                      warm_start=None):
        """
        阶段二：容差带内 AHP 加权优化二级目标

        参数:
            t_star: 阶段一最优时间（用于结果记录）
            norm: 归一化锚点（阶段一最优解处的二级指标值）
        """
        model = self._create_model('Q1_Stage2_AHP', stage=2)
        x, y = self._create_vars_q1(model, routes, demands_by_facility)
        
        # 添加约束
        self._add_demand_constraints_q1(model, routes, demands_by_facility, x, y)
        self._add_capacity_constraints_q1(model, routes, demands_by_facility, x, y)
        
        # 构建目标函数（阶段二）
        self._build_objective_q1(model, routes, x, y, norm, stage=2)

        # 设置热启动
        self._set_warm_start_q1(model, routes, x, y, warm_start)

        t0 = time_module.time()
        model.solve()
        elapsed = time_module.time() - t0
        if not self._has_solution(model):
            print(f"  错误: 阶段二无可行解，状态={model.status}")
            return None

        selected_routes, assignments = self._extract_solution_q1(
            model, routes, demands_by_facility, x, y)

        # 官方输出（含具体人员分配与指标复算）
        writer = OutputWriter(
            self._distances, self._demands, demands_by_facility)
        metrics = writer.write_all(selected_routes, assignments, out_dir)
        metrics['solve_time'] = elapsed
        return metrics

    def set_context(self, distances, demands):
        """注入数据上下文（供官方输出器使用）"""
        self._distances = distances
        self._demands = demands



class TwoStageSolverQ2(_BaseTwoStage):
    """
    问题2 两阶段求解器

    与 Q1 相同结构，但需处理三类需求与航段级容量约束；
    阶段二目标中的在途时间与浪费座公里覆盖出海/海返/穿梭三类需求。
    """

    STAGE1_TIME_LIMIT = 7200    # 阶段一求解时间限制(秒)，2小时
    STAGE2_TIME_LIMIT = 7200    # 阶段二求解时间限制(秒)，2小时
    STAGE1_REL_GAP = 0.01       # 阶段一相对间隙
    STAGE2_REL_GAP = 0.01       # 阶段二相对间隙
    STAGE2_USE_WARM = False     # Q2 阶段二不使用热启动（经验上热启动反而降低搜索效果）

    def solve(self, routes, precomputed, outbound_demand, inbound_demand,
              shuttle_demand, out_dir):
        """
        完整两阶段求解（Q2）

        返回:
            (stage1_metrics, final_metrics, norm)
        """
        print("\n===== 阶段一：最小化总飞机使用时间 =====")
        stage1 = self._solve_stage1(routes, precomputed, outbound_demand,
                                    inbound_demand, shuttle_demand)
        if stage1 is None:
            return None, None, None
        s1_selected, s1_yo, s1_yi, s1_ys, s1_metrics = stage1
        t_star = s1_metrics['total_time']
        self._print_stage_result("阶段一", s1_metrics)

        norm = {
            'transit': s1_metrics['total_transit'],
            'wasted': s1_metrics['wasted'],
            'fuel': s1_metrics['total_fuel'],
        }

        print("\n===== 阶段二：ε-容差带内 AHP 加权优化 =====")
        warm_start = ((s1_selected, s1_yo, s1_yi, s1_ys)
                      if self.STAGE2_USE_WARM else None)
        stage2 = self._solve_stage2(routes, precomputed, outbound_demand,
                                    inbound_demand, shuttle_demand,
                                    t_star, norm, out_dir, warm_start)
        if stage2 is None:
            return s1_metrics, None, norm
        final_metrics = stage2
        self._print_stage_result("阶段二", final_metrics)
        return s1_metrics, final_metrics, norm

    def _solve_stage1(self, routes, precomputed, outbound_demand,
                      inbound_demand, shuttle_demand):
        """阶段一：仅最小化总飞机使用时间（Q2）"""
        model = self._create_model('Q2_Stage1_MinTime', stage=1)
        x, yo, yi, ys = self._create_vars_q2(
            model, routes, precomputed, outbound_demand, inbound_demand, shuttle_demand)
        
        # 添加约束
        self._add_demand_constraints_q2(
            model, routes, outbound_demand, inbound_demand, shuttle_demand, x, yo, yi, ys)
        self._add_capacity_constraints_q2(model, routes, precomputed, x, yo, yi, ys)
        
        # 构建目标函数
        self._build_objective_q2(model, routes, precomputed, x, yo, yi, ys, stage=1)

        t0 = time_module.time()
        model.solve()
        elapsed = time_module.time() - t0
        if not self._has_solution(model):
            print(f"  错误: 阶段一无可行解，状态={model.status}")
            return None

        selected_routes, yo_vals, yi_vals, ys_vals = self._extract_solution_q2(
            model, routes, precomputed, x, yo, yi, ys)

        metrics = WeightedSolverQ2._calc_metrics(
            selected_routes, yo_vals, yi_vals, ys_vals, precomputed)
        metrics['solve_time'] = elapsed
        return selected_routes, yo_vals, yi_vals, ys_vals, metrics

    def _solve_stage2(self, routes, precomputed, outbound_demand,
                      inbound_demand, shuttle_demand, t_star, norm, out_dir,
                      warm_start=None, use_explicit_constraint=False):
        """
        阶段二：容差带内 AHP 加权优化二级目标（Q2）

        参数:
            t_star: 阶段一最优时间（用于结果记录）
            norm: 归一化锚点（阶段一最优解处的二级指标值）
            use_explicit_constraint: 是否使用显式时间约束
        """
        model = self._create_model('Q2_Stage2_AHP', stage=2)
        x, yo, yi, ys = self._create_vars_q2(
            model, routes, precomputed, outbound_demand, inbound_demand, shuttle_demand)
        
        # 添加约束
        self._add_demand_constraints_q2(
            model, routes, outbound_demand, inbound_demand, shuttle_demand, x, yo, yi, ys)
        self._add_capacity_constraints_q2(model, routes, precomputed, x, yo, yi, ys)
        
        # 构建目标函数（阶段二）
        self._build_objective_q2(model, routes, precomputed, x, yo, yi, ys, norm, stage=2,
                                 use_explicit_constraint=use_explicit_constraint,
                                 t_star=t_star)

        # 设置热启动
        self._set_warm_start_q2(model, routes, precomputed, x, yo, yi, ys, warm_start)

        t0 = time_module.time()
        model.solve()
        elapsed = time_module.time() - t0
        if not self._has_solution(model):
            print(f"  错误: 阶段二无可行解，状态={model.status}")
            return None

        selected_routes, yo_vals, yi_vals, ys_vals = self._extract_solution_q2(
            model, routes, precomputed, x, yo, yi, ys)

        # 官方输出
        writer = Q2OutputWriter(
            self._distances, self._demands, precomputed,
            self._outbound, self._inbound, self._shuttle)
        metrics = writer.write_all(
            selected_routes, yo_vals, yi_vals, ys_vals, out_dir)
        metrics['solve_time'] = elapsed
        return metrics
    def set_context(self, distances, demands, outbound, inbound, shuttle):
        """注入数据上下文（供官方输出器使用）"""
        self._distances = distances
        self._demands = demands
        self._outbound = outbound
        self._inbound = inbound
        self._shuttle = shuttle



# ==================== 方法对比与优越性量化 ====================
class MethodComparer:
    """
    候选方法对比器

    将两阶段组合法（本方法）与之前的尝试（字典序四目标法、8组权重
    加权单目标法）汇总对比，并给出量化优越性指标：
      1. AHP 综合得分 Z2 = w_tau·(τ/τ0) + w_util·(W/W0) + w_fuel·(F/F0)
         （以阶段一最优解为锚点归一化，越低越好）
      2. 等权重四目标归一化得分（稳健性检验，不依赖 AHP 权重）
      3. 距离理想点指标 D2（归一化空间中的欧氏距离）
      4. 帕累托支配：统计本方法支配的候选方案数
      5. 时间退化率 (T_final - T*)/T*
    """

    @staticmethod
    def parse_metrics_txt(path):
        """解析官方指标 txt 文件（字典序法结果）"""
        m = {}
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('总飞机使用时间'):
                    m['total_time'] = int(line.split('分钟')[0].split(':')[1].strip())
                elif line.startswith('人员总在途时间'):
                    m['total_transit'] = int(line.split('分钟')[0].split(':')[1].strip())
                elif line.startswith('总架次数'):
                    m['total_flights'] = int(line.split(':')[1].strip())
                elif line.startswith('总燃油消耗量'):
                    m['total_fuel'] = float(line.split('kg')[0].split(':')[1].strip())
                elif line.startswith('座位利用率'):
                    m['seat_util'] = float(line.split('%')[0].split(':')[1].strip()) / 100.0
                elif line.startswith('客座公里'):
                    m['pass_km'] = float(line.split(':')[1].strip())
                elif line.startswith('可用座公里'):
                    m['seat_km'] = float(line.split(':')[1].strip())
        m['wasted'] = m['seat_km'] - m['pass_km']
        return m

    @staticmethod
    def load_weighted_csv(path):
        """读取 8 组权重方案的指标 CSV"""
        rows = []
        with open(path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    'name': WEIGHT_DESCRIPTIONS.get(row['scheme'], row['scheme']),
                    'total_time': int(float(row['total_time_min'])),
                    'total_transit': int(float(row['total_transit_min'])),
                    'total_flights': int(row['total_flights']),
                    'total_fuel': float(row['total_fuel_kg']),
                    'seat_util': float(row['seat_util_pct']) / 100.0,
                    'wasted': float(row['wasted_seatkm']),
                })
        return rows

    @staticmethod
    def build_candidates(stage1, final, lexico, weighted_rows):
        """
        汇总所有候选方案指标

        返回:
            (candidates, anchor)
            candidates: [{'name', 'total_time', 'total_transit',
                          'total_flights', 'total_fuel', 'seat_util',
                          'wasted', 'kind'}]
            anchor: 阶段一最优解二级指标 {transit, wasted, fuel, time}
        """
        candidates = []
        # 本方法（两阶段组合）
        if final is not None:
            candidates.append({
                'name': '两阶段组合法（本方法）', 'kind': 'final',
                'total_time': final['total_time'],
                'total_transit': final['total_transit'],
                'total_flights': final['total_flights'],
                'total_fuel': final['total_fuel'],
                'seat_util': final['seat_util'],
                'wasted': final['seat_km'] - final['pass_km'],
            })
        # 阶段一基准解
        if stage1 is not None:
            candidates.append({
                'name': '阶段一（仅最小化时间）', 'kind': 'stage1',
                'total_time': stage1['total_time'],
                'total_transit': stage1['total_transit'],
                'total_flights': stage1['total_flights'],
                'total_fuel': stage1['total_fuel'],
                'seat_util': stage1['seat_util'],
                'wasted': stage1['wasted'],
            })
        # 字典序四目标法
        candidates.append({
            'name': '字典序四目标法', 'kind': 'lexico',
            'total_time': lexico['total_time'],
            'total_transit': lexico['total_transit'],
            'total_flights': lexico['total_flights'],
            'total_fuel': lexico['total_fuel'],
            'seat_util': lexico['seat_util'],
            'wasted': lexico['wasted'],
        })
        # 8 组权重加权单目标法
        for row in weighted_rows:
            candidates.append({'name': row['name'], 'kind': 'weighted',
                               **row})

        anchor = {
            'time': stage1['total_time'] if stage1 is not None else final['total_time'],
            'transit': stage1['total_transit'] if stage1 is not None else final['total_transit'],
            'wasted': stage1['wasted'] if stage1 is not None else final['wasted'],
            'fuel': stage1['total_fuel'] if stage1 is not None else final['total_fuel'],
        }
        return candidates, anchor

    @staticmethod
    def compute_scores(candidates, anchor, weights):
        """
        计算各候选方案的量化指标

        返回:
            list[dict]: 含 ahp_score, score_equal, dist_ideal, dominated,
                        in_time_band, best_t_rank
        """
        w_transit, w_util, w_fuel = weights
        # 等权重四目标归一化：以各目标在候选集中的最优值归一
        best = {
            'total_time': min(c['total_time'] for c in candidates),
            'total_transit': min(c['total_transit'] for c in candidates),
            'wasted': min(c['wasted'] for c in candidates),
            'total_fuel': min(c['total_fuel'] for c in candidates),
        }

        final = next((c for c in candidates if c['kind'] == 'final'), None)
        best_time = min(c['total_time'] for c in candidates)
        scored = []
        for c in candidates:
            # AHP 综合得分（阶段二目标，锚点为阶段一解）
            ahp_score = (w_transit * c['total_transit'] / anchor['transit']
                         + w_util * c['wasted'] / anchor['wasted']
                         + w_fuel * c['total_fuel'] / anchor['fuel'])
            # 等权重四目标归一化得分（不依赖 AHP 权重，稳健性检验）
            score_equal = 0.25 * (
                c['total_time'] / best['total_time']
                + c['total_transit'] / best['total_transit']
                + c['wasted'] / best['wasted']
                + c['total_fuel'] / best['total_fuel'])
            # 距离理想点（归一化空间欧氏距离）
            dist_ideal = np.sqrt(
                (c['total_time'] / best['total_time'] - 1.0) ** 2
                + (c['total_transit'] / best['total_transit'] - 1.0) ** 2
                + (c['wasted'] / best['wasted'] - 1.0) ** 2
                + (c['total_fuel'] / best['total_fuel'] - 1.0) ** 2)
            # 帕累托支配计数
            dominated = 0
            for c2 in candidates:
                if c2 is c:
                    continue
                dominated += int(
                    c2['total_time'] <= c['total_time']
                    and c2['total_transit'] <= c['total_transit']
                    and c2['wasted'] <= c['wasted']
                    and c2['total_fuel'] <= c['total_fuel']
                    and (c2['total_time'] < c['total_time']
                         or c2['total_transit'] < c['total_transit']
                         or c2['wasted'] < c['wasted']
                         or c2['total_fuel'] < c['total_fuel']))
            # 时间退化率
            in_time_band = (c['total_time'] - best_time) / best_time
            best_t_rank = sorted(set(c2['total_time'] for c2 in candidates)).index(c['total_time']) + 1
            scored.append({**c, 'ahp_score': ahp_score, 'score_equal': score_equal,
                           'dist_ideal': dist_ideal, 'dominated': dominated,
                           'in_time_band': in_time_band, 'best_t_rank': best_t_rank})
        return scored

    @staticmethod
    def write_comparison(candidates, anchor, weights, csv_path, txt_path,
                         problem_label, t_star, final_metrics):
        """写入方法对比结果（CSV + TXT）"""
        w_transit, w_util, w_fuel = weights
        # 排序：按 AHP 综合得分升序
        ranked = sorted(candidates, key=lambda c: c['ahp_score'])

        # CSV
        fieldnames = ['name', 'total_time', 'total_transit', 'total_flights',
                      'total_fuel', 'seat_util', 'wasted', 'ahp_score',
                      'score_equal', 'dist_ideal', 'dominated', 'in_time_band',
                      'best_t_rank']
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for c in ranked:
                writer.writerow({k: c.get(k, '') for k in fieldnames})

        # TXT
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"{problem_label} 方法对比与优越性量化\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"阶段一最优时间 T* = {t_star} min\n")
            f.write(f"AHP 权重: w_tau={w_transit:.4f}, w_util={w_util:.4f}, "
                    f"w_fuel={w_fuel:.4f}\n")
            f.write(f"容差 ε = {EPS_TOL}\n\n")
            f.write("排名 | 方法 | T(min) | τ(min) | F(kg) | util(%) | "
                    "AHP得分 | 等权得分 | 理想距离 | 支配数 | 时间退化率\n")
            f.write("-" * 120 + "\n")
            for i, c in enumerate(ranked, 1):
                f.write(f"{i:2d} | {c['name']:<20s} | {c['total_time']:>6d} | "
                        f"{c['total_transit']:>6d} | {c['total_fuel']:>8.1f} | "
                        f"{c['seat_util']*100:>6.2f} | {c['ahp_score']:>8.4f} | "
                        f"{c['score_equal']:>8.4f} | {c['dist_ideal']:>8.4f} | "
                        f"{c['dominated']:>5d} | {c['in_time_band']:>8.4f}\n")
            # 本方法的优越性
            final = next((c for c in ranked if c['kind'] == 'final'), None)
            if final is not None:
                f.write("\n" + "=" * 60 + "\n")
                f.write("本方法（两阶段组合法）优越性:\n")
                f.write(f"  AHP 综合得分排名: 第{ranked.index(final)+1}名\n")
                f.write(f"  时间退化率: {final['in_time_band']*100:.2f}%\n")
                f.write(f"  帕累托支配数: {final['dominated']}\n")

    @staticmethod
    def plot_comparison(candidates, anchor, weights, save_path, problem_label):
        """绘制方法对比图"""
        plt = setup_matplotlib()
        w_transit, w_util, w_fuel = weights
        ranked = sorted(candidates, key=lambda c: c['ahp_score'])
        names = [c['name'] for c in ranked]
        ahp_scores = [c['ahp_score'] for c in ranked]
        dist_ideals = [c['dist_ideal'] for c in ranked]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        # AHP 综合得分
        bars1 = ax1.barh(range(len(names)), ahp_scores, color='steelblue', alpha=0.8)
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names, fontsize=9)
        ax1.set_xlabel('AHP 综合得分（越低越好）')
        ax1.set_title(f'{problem_label} 方法对比 - AHP 综合得分')
        ax1.invert_yaxis()
        # 标注排名
        for i, (bar, score) in enumerate(zip(bars1, ahp_scores)):
            ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                     f'#{i+1}', va='center', fontsize=8)

        # 距离理想点
        bars2 = ax2.barh(range(len(names)), dist_ideals, color='coral', alpha=0.8)
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names, fontsize=9)
        ax2.set_xlabel('距离理想点（越低越好）')
        ax2.set_title(f'{problem_label} 方法对比 - 理想点距离')
        ax2.invert_yaxis()
        for i, (bar, dist) in enumerate(zip(bars2, dist_ideals)):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                     f'#{i+1}', va='center', fontsize=8)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def load_lexico_metrics(problem):
    """读取字典序法（之前的方法）官方指标"""
    if problem == 1:
        path = os.path.join(PROJECT_DIR, 'output', 'Problem1_lexicographic',
                            'q1-metrics.txt')
    else:
        path = os.path.join(PROJECT_DIR, 'output', 'Problem2_lexicographic',
                            'q2-metrics.txt')
    return MethodComparer.parse_metrics_txt(path)


def solve_problem1(out_dir):
    """
    问题1 两阶段求解主流程

    参数:
        out_dir: 结果输出目录

    返回:
        (stage1_metrics, final_metrics, norm, candidates)
    """
    print("=" * 60)
    print("问题1: 单向出海运输 — 两阶段组合优化")
    print("=" * 60)

    # 1. 数据加载
    print("\n[1] 数据加载...")
    loader = DataLoader()
    distances = loader.load_distances()
    demands = loader.load_demands_q1()
    demands_by_facility = loader.count_demands(demands)
    print(f"  地点: {len(distances)}, 需求: {len(demands)}人, "
          f"设施: {len(demands_by_facility)}个")

    # 2. 路径生成
    print("\n[2] 路径生成...")
    gen = RouteGenerator(distances)
    routes = gen.generate_all()
    print(f"  候选路径: {len(routes)} 条")

    # 3. 两阶段求解
    print("\n[3] 两阶段求解...")
    weights = (ahp_result['weights'][0], ahp_result['weights'][1],
               ahp_result['weights'][2])
    solver = TwoStageSolverQ1(weights, eps=EPS_TOL)
    solver.set_context(distances, demands)
    stage1, final, norm = solver.solve(routes, demands_by_facility, out_dir, use_explicit_constraint=True)
    if final is None:
        print("错误: 两阶段求解失败!")
        return None, None, None, None

    # 4. 方法对比与优越性量化
    print("\n[4] 方法对比与优越性量化...")
    lexico = load_lexico_metrics(1)
    weighted_rows = MethodComparer.load_weighted_csv(os.path.join(
        PROJECT_DIR, 'output', 'Problem1_weighted', 'q1-weight-comparison.csv'))
    candidates, anchor = MethodComparer.build_candidates(
        stage1, final, lexico, weighted_rows)
    candidates = MethodComparer.compute_scores(candidates, anchor, weights)

    comp_dir = out_dir
    MethodComparer.write_comparison(
        candidates, anchor, weights,
        os.path.join(comp_dir, 'q1-method-comparison.csv'),
        os.path.join(comp_dir, 'q1-method-comparison.txt'),
        '问题1', stage1['total_time'], final)
    MethodComparer.plot_comparison(
        candidates, anchor, weights,
        os.path.join(PROJECT_DIR, 'figures', 'Problem1',
                     'q1_method_compare.png'), '问题1')
    return stage1, final, norm, candidates


def solve_problem2(out_dir):
    """
    问题2 两阶段求解主流程

    参数:
        out_dir: 结果输出目录

    返回:
        (stage1_metrics, final_metrics, norm, candidates)
    """
    print("=" * 60)
    print("问题2: 出海、海返与穿梭联合运输 — 两阶段组合优化")
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
    print(f"  总需求: {len(demands)}人, 出海: {len(outbound)}人, "
          f"海返: {len(inbound)}人, 穿梭: {len(shuttle)}人")

    # 2. 路径生成
    print("\n[2] 路径生成...")
    shuttle_pairs = list(shuttle_demand.keys())
    gen = Q2RouteGenerator(distances)
    routes = gen.generate_all_q2(shuttle_pairs)
    print(f"  候选路径: {len(routes)} 条")

    # 3. 路由预处理
    print("\n[3] 路由预处理...")
    precomputed = {}
    for r in routes:
        precomputed[r.route_id] = Q2RoutePreprocessor.preprocess(r, distances)

    # 4. 两阶段求解
    print("\n[4] 两阶段求解...")
    weights = (ahp_result['weights'][0], ahp_result['weights'][1],
               ahp_result['weights'][2])
    solver = TwoStageSolverQ2(weights, eps=EPS_TOL)
    solver.set_context(distances, demands, outbound, inbound, shuttle)
    stage1, final, norm = solver.solve(
        routes, precomputed, outbound_demand, inbound_demand,
        shuttle_demand, out_dir)
    if final is None:
        print("错误: 两阶段求解失败!")
        return None, None, None, None

    # 5. 方法对比与优越性量化
    print("\n[5] 方法对比与优越性量化...")
    lexico = load_lexico_metrics(2)
    weighted_rows = MethodComparer.load_weighted_csv(os.path.join(
        PROJECT_DIR, 'output', 'Problem2_weighted', 'q2-weight-comparison.csv'))
    candidates, anchor = MethodComparer.build_candidates(
        stage1, final, lexico, weighted_rows)
    candidates = MethodComparer.compute_scores(candidates, anchor, weights)

    comp_dir = out_dir
    MethodComparer.write_comparison(
        candidates, anchor, weights,
        os.path.join(comp_dir, 'q2-method-comparison.csv'),
        os.path.join(comp_dir, 'q2-method-comparison.txt'),
        '问题2', stage1['total_time'], final)
    MethodComparer.plot_comparison(
        candidates, anchor, weights,
        os.path.join(PROJECT_DIR, 'figures', 'Problem2',
                     'q2_method_compare.png'), '问题2')
    return stage1, final, norm, candidates




if __name__ == '__main__':
    # AHP 权重计算与一致性检验
    ahp_result = AHPAnalyzer.compute(AHP_MATRIX)
    w = ahp_result['weights']
    print("=" * 60)
    print("层次分析法（AHP）权重计算")
    print("=" * 60)
    print(f"  判断矩阵: 在途时间 / 座位利用率 / 燃油")
    for row in AHP_MATRIX:
        print(f"    {row}")
    print(f"  权重: w_tau = {w[0]:.4f}, w_util = {w[1]:.4f}, "
          f"w_fuel = {w[2]:.4f}")
    print(f"  λmax = {ahp_result['lam_max']:.4f}, "
          f"CI = {ahp_result['ci']:.4f}, "
          f"CR = {ahp_result['cr']:.4f}")
    print(f"  一致性检验: {'通过 (CR < 0.1)' if ahp_result['cr'] < 0.1 else '未通过'}")

    # 运行指定问题（支持求解参数覆盖，便于多次试验取最优）
    args = sys.argv[1:]
    which = args[0] if args and not args[0].startswith('--') else 'all'
    if '--no-warm' in args:
        # 阶段二不使用阶段一 MIP 起点（某些模型上热启动反而降低搜索效果）
        TwoStageSolverQ1.STAGE2_USE_WARM = False
        TwoStageSolverQ2.STAGE2_USE_WARM = False
    for i, a in enumerate(args):
        if a == '--gap1' and i + 1 < len(args):
            v = float(args[i + 1])
            TwoStageSolverQ1.STAGE1_REL_GAP = v
            TwoStageSolverQ2.STAGE1_REL_GAP = v
        if a == '--gap2' and i + 1 < len(args):
            v = float(args[i + 1])
            TwoStageSolverQ1.STAGE2_REL_GAP = v
            TwoStageSolverQ2.STAGE2_REL_GAP = v
        if a == '--tl2' and i + 1 < len(args):
            v = int(args[i + 1])
            TwoStageSolverQ1.STAGE2_TIME_LIMIT = v
            TwoStageSolverQ2.STAGE2_TIME_LIMIT = v
        if a == '--eps' and i + 1 < len(args):
            v = float(args[i + 1])
            EPS_TOL = v
    if which in ('1', 'q1', 'all'):
        solve_problem1(os.path.join(PROJECT_DIR, 'output', 'Problem1'))
    if which in ('2', 'q2', 'all'):
        solve_problem2(os.path.join(PROJECT_DIR, 'output', 'Problem2'))

