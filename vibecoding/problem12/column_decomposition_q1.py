# -*- coding: utf-8 -*-
"""
文件: column_decomposition_q1.py
功能: 问题1 列分解（列生成 / Dantzig-Wolfe 分解）实验求解器
项目: 2026策联杯 B题 - 海上油田人员直升机运载计划编排

方法（双阶段思考流程）:
  ▸ 外层：列分解（Column Decomposition）
      受限主问题 RMP 采用"航线-装载模式"列（一架次 = 一条航线 + 一个装载向量），
      通过求解 RMP 的对偶变量（需求约束 π、起点约束 μ、时间预算约束 θ）驱动
      定价子问题，为对偶意义下最"划算"的航线生成新列并加入路径池，迭代至
      无负检验数（或达到迭代上限）。路径池由静态启发式池出发并做超集扩展，
      因此最终两阶段 MIP 的解质量理论上不劣于静态池。
  ▸ 内层：两阶段 ε-容差 + AHP 加权（与 two_stage_solver 一致）
      阶段一：在列生成收敛后的最终路径池上求解 min T，得到 T* 与归一化锚点；
      阶段二：在 T <= (1+ε)·T* 的容差带内最小化 AHP 加权二级目标
              min w_tau·(τ/τ0) + w_util·(W/W0) + w_fuel·(F/F0)。

设计要点:
  1. 路径池保证为静态池的超集 → 列分解解不劣于静态启发式池两阶段解；
  2. 定价子问题对每条候选航线做容量背包（按对偶价值排序装满座位），
     一次性给出该航线的最优装载模式，保证"若该航线可改善主问题，必然被发现"；
  3. 阶段一、阶段二各做一轮列生成：阶段一以总时间为目标，阶段二在时间
     预算约束下以 AHP 加权二级目标定价，两个阶段共用同一对偶定价框架；
  4. 最终两阶段 MIP 复用 TwoStageSolverQ1 与 OutputWriter，保证结果口径
     与官方方案完全一致。

运行（Powershell）:
  python column_decomposition_q1.py
  python column_decomposition_q1.py --smoke          # 快速自检
  python column_decomposition_q1.py --no-stage2-cg   # 跳过阶段二定价
  python column_decomposition_q1.py --lagrangian      # 启用乘子驱动列生成（允许长时间运行）
"""

import csv
import os
import sys
import time as time_module
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# ==================== 模块路径 ====================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VIBECODING_DIR = os.path.dirname(_THIS_DIR)
if _VIBECODING_DIR not in sys.path:
    sys.path.insert(0, _VIBECODING_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import numpy as np
import coptpy as cp

# 控制台 UTF-8 输出适配（Windows PowerShell 默认代码页可能为 GBK）
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

from config import (
    AIRCRAFT, AIRCRAFT_TYPES, AIRPORTS, PROJECT_DIR, fmt6, setup_matplotlib
)
from base_models import DataLoader, Route, TSPSolver, make_route
from problem1_solution import RouteGenerator
from two_stage_solver import (
    AHPAnalyzer, AHP_MATRIX, EPS_TOL, TwoStageSolverQ1
)


# ==================== 实验参数 ====================
RC_TOL = 1e-6                  # 检验数阈值（负数即认为存在改善空间）
MAX_ITERS = 30                 # 列生成最大迭代次数
ADD_TOP_K = 80                 # 每轮最多加入的新列数（控制主问题规模）

# 定价宇宙（候选航线超集）规模上限：静态池之外的扩展
UNIVERSE_CAPS = {
    'pairs': 700,              # 每机场：按节省值取前 N 个设施对
    'triples': 1500,           # 每机场：三设施组合上限
    'quads': 1200,             # 每机场：四设施组合上限
    'quints': 800,             # 每机场：五设施组合上限
}
# 距离阈值（km）：扩展组合时新设施与现有设施的最小距离上限
TRIPLE_DIST = 120.0
QUAD_DIST = 100.0
QUINT_DIST = 80.0

# 冒烟测试参数（快速验证流程，不追求解质量）
SMOKE_CAPS = {'pairs': 60, 'triples': 150, 'quads': 100, 'quints': 50}


@dataclass
class Column:
    """列生成主问题中的一列：一条航线 + 一个装载模式"""
    route: Route
    pattern: Dict[str, int] = field(default_factory=dict)

    def pattern_key(self) -> Tuple:
        """装载模式的去重键（按设施排序的二元组序列）"""
        return tuple(sorted(self.pattern.items()))

    def p(self, facility: str) -> int:
        """该列运往指定设施的人数"""
        return self.pattern.get(facility, 0)

    @property
    def t(self) -> int:
        """该列对应的航线总时间"""
        return self.route.time


class PricingUniverse:
    """
    定价子问题的候选航线宇宙

    以静态启发式路径池为基底，按 Clarke-Wright 节省值与距离阈值扩展
    2~5 设施组合，形成静态池的严格超集。宇宙只构建一次并缓存 Route 对象，
    每轮列生成仅做 O(|宇宙|) 的背包定价，计算开销可忽略。
    """

    def __init__(self, distances):
        self.distances = distances
        self.facilities = sorted([f for f in distances if f.startswith('F')])

    def build(self, caps: Optional[Dict[str, int]] = None) -> Tuple[List[Route], List[Route]]:
        """
        构建定价宇宙（静态池 ∪ 扩展组合，按路径去重）

        参数:
            caps: 组合规模上限（None 使用默认上限）

        返回:
            (宇宙航线列表, 静态池航线列表)
        """
        caps = caps or UNIVERSE_CAPS
        universe = []
        static_routes = []
        seen_paths = set()
        counter = [0]

        def add_route(airport, atype, fac_list):
            """构建一条航线并去重加入宇宙"""
            best_order, _ = TSPSolver.best_order(airport, fac_list, self.distances)
            r = make_route(counter[0], airport, atype, best_order, self.distances)
            if r is None:
                return
            key = (airport, tuple(r.path), atype)
            if key in seen_paths:
                return
            seen_paths.add(key)
            universe.append(r)
            counter[0] += 1

        # 1) 静态启发式池（基底，保证最终路径池为其超集）
        print("  [1/5] 静态启发式路径池...")
        gen = RouteGenerator(self.distances)
        static = gen.generate_all()
        for r in static:
            key = (r.airport, tuple(r.path), r.aircraft_type)
            if key not in seen_paths:
                seen_paths.add(key)
                universe.append(r)
            static_routes.append(r)
        counter[0] = max([r.route_id for r in universe], default=0) + 1
        print(f"        静态池: {len(static)} 条，宇宙累计: {len(universe)} 条")

        # 2) 双设施组合：全部正节省值设施对（静态池仅取前 300）
        print("  [2/5] 双设施扩展组合...")
        for airport in AIRPORTS:
            savings = []
            for i in range(len(self.facilities)):
                for j in range(i + 1, len(self.facilities)):
                    f1, f2 = self.facilities[i], self.facilities[j]
                    s = (self.distances[f1][airport] +
                         self.distances[airport][f2] -
                         self.distances[f1][f2])
                    if s > 0:
                        savings.append((s, f1, f2))
            savings.sort(reverse=True)
            for s, f1, f2 in savings[:caps['pairs']]:
                for atype in AIRCRAFT_TYPES:
                    add_route(airport, atype, [f1, f2])
        print(f"        宇宙累计: {len(universe)} 条")

        # 3) 三设施组合：双设施组合扩展 120km 内设施
        print("  [3/5] 三设施扩展组合...")
        for airport in AIRPORTS:
            pair_sets = sorted({
                frozenset(r.dest_facilities)
                for r in universe
                if r.airport == airport and len(r.dest_facilities) == 2
            })
            triples = set()
            for pair in pair_sets:
                if len(triples) >= caps['triples']:
                    break
                members = list(pair)
                for f3 in self.facilities:
                    if f3 in pair:
                        continue
                    min_d = min(self.distances[f3][f] for f in members)
                    if min_d <= TRIPLE_DIST:
                        triples.add(frozenset(members + [f3]))
            for fac_set in list(triples)[:caps['triples']]:
                for atype in AIRCRAFT_TYPES:
                    add_route(airport, atype, list(fac_set))
        print(f"        宇宙累计: {len(universe)} 条")

        # 4) 四设施组合：三设施组合扩展 100km 内设施
        print("  [4/5] 四设施扩展组合...")
        for airport in AIRPORTS:
            quads = set()
            triple_sets = sorted({
                frozenset(r.dest_facilities)
                for r in universe
                if r.airport == airport and len(r.dest_facilities) == 3
            })
            for triple in triple_sets:
                if len(quads) >= caps['quads']:
                    break
                members = list(triple)
                for f4 in self.facilities:
                    if f4 in triple:
                        continue
                    min_d = min(self.distances[f4][f] for f in members)
                    if min_d <= QUAD_DIST:
                        quads.add(frozenset(members + [f4]))
            for fac_set in list(quads)[:caps['quads']]:
                for atype in AIRCRAFT_TYPES:
                    add_route(airport, atype, list(fac_set))
        print(f"        宇宙累计: {len(universe)} 条")

        # 5) 五设施组合：四设施组合扩展 80km 内设施
        print("  [5/5] 五设施扩展组合...")
        for airport in AIRPORTS:
            quints = set()
            quad_sets = sorted({
                frozenset(r.dest_facilities)
                for r in universe
                if r.airport == airport and len(r.dest_facilities) == 4
            })
            for quad in quad_sets:
                if len(quints) >= caps['quints']:
                    break
                members = list(quad)
                for f5 in self.facilities:
                    if f5 in quad:
                        continue
                    min_d = min(self.distances[f5][f] for f in members)
                    if min_d <= QUINT_DIST:
                        quints.add(frozenset(members + [f5]))
            for fac_set in list(quints)[:caps['quints']]:
                for atype in AIRCRAFT_TYPES:
                    add_route(airport, atype, list(fac_set))
        print(f"        宇宙累计: {len(universe)} 条")

        return universe, static_routes


class MasterLP:
    """
    受限主问题（Restricted Master Problem）LP

    决策变量 λ_k ≥ 0：列 k（航线-装载模式）被使用的次数（LP 松弛允许分数）。
    约束:
      C1_f: Σ_k p_{k,f}·λ_k = n_f                     （需求满足，对偶 π_f 自由）
      C2_{f,a}: Σ_{k:a_k=a} p_{k,f}·λ_k ≥ n_{f,a}     （起点感知，对偶 μ ≥ 0）
      C3: Σ_k t_k·λ_k ≤ (1+ε)·T*                      （阶段二时间预算，对偶 θ ≥ 0）
    目标:
      阶段一: min Σ_k t_k·λ_k
      阶段二: min Σ_k [g_r + Σ_f u_{r,f}·p_{k,f}]·λ_k
    """

    def __init__(self, demands_by_facility, stage=1, weights=None,
                 norm=None, t_star=None, eps=EPS_TOL):
        self.demands_by_facility = demands_by_facility
        self.stage = stage
        self.weights = weights          # (w_transit, w_util, w_fuel)
        self.norm = norm                # 归一化锚点 {transit, wasted, fuel}
        self.t_star = t_star            # 阶段一最优时间（阶段二时间预算）
        self.eps = eps

    def _column_cost(self, col: Column) -> float:
        """列在阶段二目标中的固定项 + 人均项系数（阶段一直接返回航线时间）"""
        if self.stage == 1:
            return float(col.t)
        w_tau, w_util, w_fuel = self.weights
        n_tau = self.norm['transit']
        n_w = self.norm['wasted']
        n_f = self.norm['fuel']
        r = col.route
        # 固定项: w_util·(c·ℓ)/W0 + w_fuel·F/F0
        g = w_util * r.capacity * r.total_distance / n_w + w_fuel * r.fuel / n_f
        # 人均项: Σ_f [w_tau·τ/W0τ - w_util·δ/W0]·p_f
        for f, p in col.pattern.items():
            g += (w_tau * r.transit_times.get(f, 0) / n_tau
                  - w_util * r.dist_to_facility.get(f, 0) / n_w) * p
        return g

    def solve(self, columns: List[Column]) -> Dict:
        """
        构建并求解受限主问题 LP，返回对偶变量

        参数:
            columns: 当前列池

        返回:
            dict: obj(LP 目标值), pi({设施: π}), mu({(设施,机场): μ}),
                  theta(阶段二时间预算对偶), status
        """
        env = cp.Envr()
        model = env.createModel('Q1_RMP')
        # 关闭预求解，保证对偶变量与原约束一一对应
        model.setParam(cp.COPT.Param.Presolve, 0)
        model.setParam(cp.COPT.Param.LogToConsole, 0)

        lam = [model.addVar(lb=0, ub=cp.COPT.INFINITY,
                            name=f'lam_{i}') for i in range(len(columns))]

        cons_c1 = []
        for f in sorted(self.demands_by_facility.keys()):
            n_f = self.demands_by_facility[f]['total']
            expr = cp.quicksum([col.p(f) * lam[i]
                                for i, col in enumerate(columns)])
            cons_c1.append(model.addConstr(expr == n_f, name=f'demand_{f}'))

        cons_c2 = []
        origin_keys = sorted({
            (f, a)
            for f, info in self.demands_by_facility.items()
            for a, cnt in info['by_origin'].items()
            if a != 'LAND' and cnt > 0
        })
        for (f, a) in origin_keys:
            n_fa = self.demands_by_facility[f]['by_origin'][a]
            expr = cp.quicksum([
                col.p(f) * lam[i]
                for i, col in enumerate(columns) if col.route.airport == a])
            cons_c2.append(model.addConstr(expr >= n_fa, name=f'origin_{f}_{a}'))

        cons_c3 = []
        if self.stage == 2:
            max_time = (1 + self.eps) * self.t_star
            expr = cp.quicksum([col.t * lam[i] for i, col in enumerate(columns)])
            cons_c3.append(model.addConstr(expr <= max_time, name='time_budget'))

        obj_expr = cp.quicksum([self._column_cost(col) * lam[i]
                                for i, col in enumerate(columns)])
        model.setObjective(obj_expr, sense=cp.COPT.MINIMIZE)
        model.solve()

        if model.status not in (cp.COPT.OPTIMAL, cp.COPT.TIMEOUT):
            return {'status': model.status}

        duals = model.getDuals()
        pi = {f: duals[i] for i, f in enumerate(sorted(self.demands_by_facility.keys()))}
        mu = {(f, a): duals[len(self.demands_by_facility) + i]
              for i, (f, a) in enumerate(origin_keys)}
        theta = duals[-1] if cons_c3 else 0.0

        return {
            'status': model.status,
            'obj': model.objval,
            'pi': pi,
            'mu': mu,
            'theta': theta,
        }


class PricingSolver:
    """
    定价子问题：给定对偶变量，为每条候选航线求解最优装载模式与检验数

    对一条航线 r，阶段一检验数:
        rc1(r) = t_r - max_p Σ_f (π_f + μ_{f,a_r})·p_f
    阶段二检验数:
        rc2(r) = g_r + θ·t_r - max_p Σ_f (π_f + μ_{f,a_r} - u_{r,f})·p_f
    其中装载模式 p 由容量背包给出（按对偶价值排序装满座位）。
    """

    def __init__(self, universe: List[Route], demands_by_facility,
                 weights=None, norm=None):
        self.universe = universe
        self.demands_by_facility = demands_by_facility
        self.weights = weights
        self.norm = norm
        # 各设施需求总数（作为装载模式上界）
        self.n_f = {f: info['total'] for f, info in demands_by_facility.items()}

    @staticmethod
    def _knapsack_pattern(route: Route, values: Dict[str, float],
                          n_f: Dict[str, int]) -> Tuple[Dict[str, int], float]:
        """
        容量背包：按对偶价值降序装满座位，得到最优装载模式

        参数:
            route: 候选航线
            values: 每个设施的单位人数对偶价值
            n_f: 每个设施的剩余需求上界

        返回:
            (装载模式, 总价值)
        """
        cap_left = route.capacity
        pattern = {}
        value = 0.0
        for f, v in sorted(values.items(), key=lambda kv: -kv[1]):
            if v <= 0 or cap_left <= 0:
                break
            take = min(cap_left, n_f.get(f, 0))
            if take <= 0:
                continue
            pattern[f] = take
            value += v * take
            cap_left -= take
        return pattern, value

    def price(self, pi: Dict[str, float], mu: Dict[Tuple[str, str], float],
              stage: int, theta: float = 0.0,
              facility_multiplier: Optional[Dict[str, float]] = None) -> List[Column]:
        """
        定价：扫描整个宇宙，返回检验数为负的候选列（按检验数升序）

        参数:
            pi: 需求约束对偶
            mu: 起点感知约束对偶
            stage: 1（总时间目标）或 2（AHP 加权二级目标）
            theta: 阶段二时间预算约束对偶

        返回:
            负检验数候选列列表（已按检验数升序）
        """
        w_tau, w_util, w_fuel = self.weights if self.weights else (0.0, 0.0, 0.0)
        n_tau = self.norm['transit'] if self.norm else 1.0
        n_w = self.norm['wasted'] if self.norm else 1.0
        n_fuel = self.norm['fuel'] if self.norm else 1.0

        candidates = []
        for r in self.universe:
            facs = sorted(r.dest_facilities)
            if stage == 1:
                values = {
                    f: pi.get(f, 0.0) + mu.get((f, r.airport), 0.0)
                       - (facility_multiplier.get(f, 0.0) if facility_multiplier else 0.0)
                    for f in facs
                }
                pattern, value = self._knapsack_pattern(r, values, self.n_f)
                if not pattern:
                    continue
                rc = float(r.time) - value
            else:
                values = {}
                for f in facs:
                    # 对偶价值扣除二级目标的人均成本（在途贡献 + 浪费座公里贡献）
                    u_f = (w_tau * r.transit_times.get(f, 0) / n_tau
                           - w_util * r.dist_to_facility.get(f, 0) / n_w)
                    values[f] = pi.get(f, 0.0) + mu.get((f, r.airport), 0.0) - u_f - (facility_multiplier.get(f, 0.0) if facility_multiplier else 0.0)
                pattern, value = self._knapsack_pattern(r, values, self.n_f)
                if not pattern:
                    continue
                g = (w_util * r.capacity * r.total_distance / n_w
                     + w_fuel * r.fuel / n_fuel)
                rc = g + theta * r.time - value

            if rc < -RC_TOL:
                candidates.append((rc, Column(r, pattern)))

        candidates.sort(key=lambda x: x[0])
        return [col for _, col in candidates]


class ColumnDecompositionSolverQ1:
    """
    Q1 列分解 + 两阶段组合优化主流程

    流程:
      1. 构建定价宇宙（静态池超集）
      2. 以静态池单设施装载列初始化 RMP，做阶段一列生成（min T）
      3. 在收敛后的路径池上求解阶段一 MIP，得到 T* 与归一化锚点
      4. 阶段二列生成（时间预算约束下定价 AHP 加权二级目标）
      5. 最终路径池上运行两阶段 ε-容差 + AHP 求解（复用 TwoStageSolverQ1）
    """

    def __init__(self, out_dir, smoke=False, do_stage2_cg=True,
                 max_iters=MAX_ITERS, add_top_k=ADD_TOP_K,
                 limt_time_limit=7200, limt_rel_gap=0.005,
                 limt2_time_limit=7200, limt2_rel_gap=0.01,
                 final_name='列分解+两阶段（本实验）',
                 lagrangian_rounds: int = 0):
        self.out_dir = out_dir
        self.smoke = smoke
        self.do_stage2_cg = do_stage2_cg
        self.max_iters = max_iters
        self.add_top_k = add_top_k
        # 极限 T 模式参数：阶段一 MIP 的时限与相对间隙
        self.limt_time_limit = limt_time_limit
        self.limt_rel_gap = limt_rel_gap
        self.limt2_time_limit = limt2_time_limit
        self.limt2_rel_gap = limt2_rel_gap
        # 对比表中本实验方案的显示名称
        self.final_name = final_name
        self.cg_stats = {'stage1': [], 'stage2': [], 'lagrangian': []}
        self.lagrangian_rounds = lagrangian_rounds
        self.weights = None
        self.eps = EPS_TOL

    # ---------- 列池管理 ----------
    @staticmethod
    def _seed_columns(routes: List[Route],
                      demands_by_facility: Dict) -> List[Column]:
        """
        初始化列池：静态池每条航线为其每个目标设施生成单设施装载列

        保证 RMP 初始可行：任一设施的需求可由其单设施航线列（允许分数
        重复使用）完全覆盖。
        """
        cols = []
        seen = set()
        for r in routes:
            for f in sorted(r.dest_facilities):
                if f not in demands_by_facility:
                    continue
                pax = min(r.capacity, demands_by_facility[f]['total'])
                if pax <= 0:
                    continue
                key = (r.route_id, f, pax)
                if key in seen:
                    continue
                seen.add(key)
                cols.append(Column(r, {f: pax}))
        return cols

    @staticmethod
    def _dedup_columns(columns: List[Column], known_keys: Set[Tuple]) -> List[Column]:
        """过滤掉与已知列重复的候选列"""
        result = []
        for col in columns:
            key = (col.route.route_id, col.pattern_key())
            if key in known_keys:
                continue
            known_keys.add(key)
            result.append(col)
        return result

    # ---------- 列生成主循环 ----------
    def _cg_loop(self, columns: List[Column], universe: List[Route],
                 demands_by_facility, stage, known_keys, master_lp,
                 facility_multiplier=None):
        """
        单阶段列生成迭代循环

        参数:
            columns: 当前列池（会被就地扩充）
            universe: 定价宇宙
            demands_by_facility: 按设施分组的需求
            stage: 1 或 2
            known_keys: 已存在列的去重键集合（会被就地更新）
            master_lp: MasterLP 实例

        返回:
            (迭代次数, 首次 LP 目标, 末次 LP 目标, 新增列数)
        """
        pricing = PricingSolver(universe, demands_by_facility,
                                weights=master_lp.weights, norm=master_lp.norm)
        n_added_total = 0
        first_obj = None
        last_obj = None
        n_iters = 0
        objs = []

        for it in range(self.max_iters):
            res = master_lp.solve(columns)
            if res.get('status') not in (cp.COPT.OPTIMAL, cp.COPT.TIMEOUT):
                print(f"  [迭代 {it + 1}] RMP 状态异常: {res.get('status')}，终止列生成")
                break
            obj = res['obj']
            if first_obj is None:
                first_obj = obj
            last_obj = obj
            objs.append(obj)

            candidates = pricing.price(res['pi'], res['mu'], stage,
                                       theta=res.get('theta', 0.0),
                                       facility_multiplier=facility_multiplier)
            candidates = self._dedup_columns(candidates, known_keys)
            if not candidates:
                print(f"  [迭代 {it + 1}] 无负检验数列，LP 最优收敛 (obj={obj:.2f})")
                n_iters = it + 1
                break

            n_add = min(len(candidates), self.add_top_k)
            for col in candidates[:n_add]:
                columns.append(col)
            n_added_total += n_add
            n_iters = it + 1
            print(f"  [迭代 {it + 1}] LP obj={obj:.2f}，"
                  f"候选列 {len(candidates)} 条，新增 {n_add} 条 "
                  f"(列池 {len(columns)} 条)")

        self.cg_stats[f'stage{stage}'].append({
            'iters': n_iters, 'first_obj': first_obj, 'last_obj': last_obj,
            'added': n_added_total, 'pool': len(columns), 'objs': objs,
        })
        return n_iters, first_obj, last_obj, n_added_total

    # ---------- 主流程 ----------
    def run(self, distances, demands, demands_by_facility):
        """
        执行完整列分解 + 两阶段流程

        参数:
            distances: 距离矩阵
            demands: 需求列表
            demands_by_facility: 按设施分组的需求

        返回:
            dict: 包含阶段一/最终指标、列生成统计、路径池信息
        """
        t_start = time_module.time()
        print("=" * 66)
        print("问题1 列分解（列生成）+ 两阶段组合优化 实验")
        print("=" * 66)

        # AHP 权重
        ahp = AHPAnalyzer.compute(AHP_MATRIX)
        self.weights = tuple(ahp['weights'][i] for i in range(3))
        print(f"AHP 权重: w_tau={self.weights[0]:.4f}, "
              f"w_util={self.weights[1]:.4f}, w_fuel={self.weights[2]:.4f}")

        # 1) 构建定价宇宙
        print("\n[1] 构建定价宇宙（静态池超集）...")
        caps = SMOKE_CAPS if self.smoke else None
        universe, static_routes = PricingUniverse(distances).build(caps)
        print(f"  宇宙规模: {len(universe)} 条候选航线")
        print(f"  静态池规模: {len(static_routes)} 条")

        # 2) 阶段一列生成
        print("\n[2] 阶段一列生成（min T）...")
        columns = self._seed_columns(static_routes, demands_by_facility)
        known_keys = {(c.route.route_id, c.pattern_key()) for c in columns}

        master1 = MasterLP(demands_by_facility, stage=1,
                           eps=self.eps)
        it1, obj1_0, obj1_end, add1 = self._cg_loop(
            columns, universe, demands_by_facility, 1, known_keys, master1)
        pool1_routes = self._routes_from_columns(columns)
        print(f"  阶段一列生成完成: {it1} 轮, LP {obj1_0:.2f} -> {obj1_end:.2f}, "
              f"新增列 {add1} 条, 路径池 {len(pool1_routes)} 条")

        # 3) 阶段一 MIP（收敛后路径池）
        print("\n[3] 阶段一 MIP（收敛路径池）...")
        solver = TwoStageSolverQ1(self.weights, eps=self.eps)
        solver.set_context(distances, demands)
        stage1_mip = solver._solve_stage1(pool1_routes, demands_by_facility)
        if stage1_mip is None:
            print("  错误: 阶段一 MIP 无可行解")
            return None
        s1_routes, s1_assign, s1_metrics = stage1_mip
        t_star = s1_metrics['total_time']
        norm = {
            'transit': s1_metrics['total_transit'],
            'wasted': s1_metrics['wasted'],
            'fuel': s1_metrics['total_fuel'],
        }
        print(f"  T* = {t_star} min, τ0={norm['transit']}, "
              f"W0={norm['wasted']:.1f}, F0={norm['fuel']:.1f}, "
              f"架次={s1_metrics['total_flights']}")

        # 4) 阶段二列生成（可选）
        columns2 = list(columns)
        if self.do_stage2_cg:
            print("\n[4] 阶段二列生成（AHP 加权二级目标, T ≤ (1+ε)T*）...")
            master2 = MasterLP(demands_by_facility, stage=2,
                               weights=self.weights, norm=norm,
                               t_star=t_star, eps=self.eps)
            it2, obj2_0, obj2_end, add2 = self._cg_loop(
                columns2, universe, demands_by_facility, 2, known_keys, master2)
            print(f"  阶段二列生成完成: {it2} 轮, LP {obj2_0:.6f} -> {obj2_end:.6f}, "
                  f"新增列 {add2} 条")
        else:
            print("\n[4] 跳过阶段二列生成（--no-stage2-cg）")

        pool2_routes = self._routes_from_columns(columns2)
        print(f"  最终路径池: {len(pool2_routes)} 条航线, "
              f"{len(columns2)} 条装载列")

        # 5) 最终两阶段 ε-容差 + AHP 求解
        print("\n[5] 最终两阶段 ε-容差 + AHP 求解...")
        os.makedirs(self.out_dir, exist_ok=True)
        stage1_final, final_metrics, norm_final = solver.solve(
            pool2_routes, demands_by_facility, self.out_dir)
        if final_metrics is None:
            print("  错误: 最终两阶段求解失败")
            return None

        elapsed = time_module.time() - t_start
        result = {
            'stage1_mip': s1_metrics,
            'stage1_final': stage1_final,
            'final': final_metrics,
            'norm': norm,
            't_star': t_star,
            'pool1': len(pool1_routes),
            'pool2': len(pool2_routes),
            'columns1': len(columns),
            'columns2': len(columns2),
            'universe': len(universe),
            'cg_stats': self.cg_stats,
            'elapsed': elapsed,
        }
        print(f"\n[6] 列分解实验完成，总耗时 {elapsed:.1f}s")
        return result

    # ---------- 极限 T 模式 ----------
    def run_limT(self, distances, demands, demands_by_facility):
        """
        极限 T 模式主流程：列分解只服务 T 最小，锁定 T* 后再优化二级目标

        与 run() 的区别:
          1. 阶段一 MIP 使用更长时限、更紧间隙，把 T 压到极限（T*_lim）；
          2. 阶段二采用 COPT 分层目标（group-0: min T reltol=ε 宽松间隙
             快速通过，group-1: min Z2 长时间收敛），在极限 T 基础上
             优化 AHP 加权二级目标。

        返回:
            同 run() 的结果字典
        """
        t_start = time_module.time()
        print("=" * 66)
        print("问题1 列分解 + 极限 T + 两阶段 实验")
        print(f"  阶段一 MIP: 时限 {self.limt_time_limit}s, "
              f"间隙 {self.limt_rel_gap}")
        print("=" * 66)

        ahp = AHPAnalyzer.compute(AHP_MATRIX)
        self.weights = tuple(ahp['weights'][i] for i in range(3))
        print(f"AHP 权重: w_tau={self.weights[0]:.4f}, "
              f"w_util={self.weights[1]:.4f}, w_fuel={self.weights[2]:.4f}")

        # 1) 构建定价宇宙（静态池超集）
        print("\n[1] 构建定价宇宙（静态池超集）...")
        caps = SMOKE_CAPS if self.smoke else None
        universe, static_routes = PricingUniverse(distances).build(caps)
        print(f"  宇宙规模: {len(universe)} 条候选航线, "
              f"静态池: {len(static_routes)} 条")

        # 2) 阶段一列生成（只服务 min T）
        print("\n[2] 阶段一列生成（min T）...")
        columns = self._seed_columns(static_routes, demands_by_facility)
        known_keys = {(c.route.route_id, c.pattern_key()) for c in columns}
        master1 = MasterLP(demands_by_facility, stage=1, eps=self.eps)
        it1, obj1_0, obj1_end, add1 = self._cg_loop(
            columns, universe, demands_by_facility, 1, known_keys, master1)
        pool1_routes = self._routes_from_columns(columns)
        print(f"  阶段一列生成完成: {it1} 轮, LP {obj1_0:.2f} -> {obj1_end:.2f}, "
              f"新增列 {add1} 条, 路径池 {len(pool1_routes)} 条")

        # 3) 极限 T 阶段一 MIP：长时限 + 紧间隙，把 T 压到极限
        #    说明: 不用硬约束/热启动（实测本版 COPT 的 setMipStart 不生效，
        #    且硬约束会卡死启发式），单次求解 + 紧间隙即可稳定收敛到
        #    低于静态池 T* 的水平（LP 下界 ~14720，0.5% 间隙对应 <14800）。
        print("\n[3] 极限 T 阶段一 MIP（min T，长时限 + 紧间隙）...")
        solver = TwoStageSolverQ1(self.weights, eps=self.eps)
        solver.set_context(distances, demands)
        solver.STAGE1_TIME_LIMIT = self.limt_time_limit
        solver.STAGE1_REL_GAP = self.limt_rel_gap
        stage1_limt = solver._solve_stage1(pool1_routes, demands_by_facility)
        if stage1_limt is None:
            print("  错误: 极限 T 阶段一无可行解")
            return None
        s1_routes, s1_assign, s1_metrics = stage1_limt
        t_star = s1_metrics['total_time']
        norm = {
            'transit': s1_metrics['total_transit'],
            'wasted': s1_metrics['wasted'],
            'fuel': s1_metrics['total_fuel'],
        }
        print(f"  T*_lim = {t_star} min, "
              f"τ0={norm['transit']}, W0={norm['wasted']:.1f}, "
              f"F0={norm['fuel']:.1f}, 架次={s1_metrics['total_flights']}, "
              f"求解 {s1_metrics['solve_time']:.0f}s")

        # 4) 阶段二列生成（在极限 T 预算下定价 Z2）
        columns2 = list(columns)
        if self.do_stage2_cg:
            print(f"\n[4] 阶段二列生成（min Z2, T ≤ (1+{self.eps})·{t_star}）...")
            master2 = MasterLP(demands_by_facility, stage=2,
                               weights=self.weights, norm=norm,
                               t_star=t_star, eps=self.eps)
            it2, obj2_0, obj2_end, add2 = self._cg_loop(
                columns2, universe, demands_by_facility, 2, known_keys, master2)
            print(f"  阶段二列生成完成: {it2} 轮, LP {obj2_0:.6f} -> "
                  f"{obj2_end:.6f}, 新增列 {add2} 条")
        else:
            print("\n[4] 跳过阶段二列生成（--no-stage2-cg）")

        pool2_routes = self._routes_from_columns(columns2)
        print(f"  最终路径池: {len(pool2_routes)} 条航线, "
              f"{len(columns2)} 条装载列")

        # 5) 阶段二 MIP：COPT 分层目标（group-0: min T reltol=ε 宽松间隙，
        #    group-1: min Z2 长时间收敛）。
        #    注意: 曾尝试 slack 可行性目标（group-0 应秒停），但 COPT 启发式
        #    找不到 slack=0 的解（120s 卡在 slack=354），因此保留原分层方案：
        #    group-0 用 min T（3% 间隙约 200s 停），group-1 获得剩余全部时间。
        print(f"\n[5] 阶段二 MIP（分层目标: min T → min Z2, "
              f"RelGap={self.limt2_rel_gap}, TL={self.limt2_time_limit}s）...")
        os.makedirs(self.out_dir, exist_ok=True)
        solver.STAGE2_TIME_LIMIT = self.limt2_time_limit
        solver.STAGE2_REL_GAP = self.limt2_rel_gap
        final_metrics = solver._solve_stage2(
            pool2_routes, demands_by_facility, t_star, norm, self.out_dir)
        if final_metrics is None:
            print("  错误: 阶段二求解失败")
            return None

        elapsed = time_module.time() - t_start
        result = {
            'stage1_mip': s1_metrics,
            'stage1_final': s1_metrics,
            'final': final_metrics,
            'norm': norm,
            't_star': t_star,
            'pool1': len(pool1_routes),
            'pool2': len(pool2_routes),
            'columns1': len(columns),
            'columns2': len(columns2),
            'universe': len(universe),
            'cg_stats': self.cg_stats,
            'elapsed': elapsed,
        }
        print(f"\n[6] 极限 T 实验完成，总耗时 {elapsed:.1f}s")
        return result


    @staticmethod
    def _facility_multiplier_from_columns(columns: List[Column], demands_by_facility: Dict[str, Dict],
                                          penalty_scale: float = 0.10) -> Dict[str, float]:
        """
        基于当前列池计算设施级乘子（惩罚项）。
        思路: 统计列池里每个设施的总运力占比，超过需求 1.0 时给正乘子，
        让定价阶段更倾向选择未过饱和设施的列。

        参数:
            columns: 当前列池
            demands_by_facility: 设施需求信息
            penalty_scale: 乘子缩放系数

        返回:
            facility_multiplier: {设施: 乘子值}
        """
        fac_load = {}
        for col in columns:
            for f, p in col.pattern.items():
                fac_load[f] = fac_load.get(f, 0) + p

        facility_multiplier = {}
        for f, info in demands_by_facility.items():
            demand = info['total']
            if demand <= 0:
                facility_multiplier[f] = 0.0
                continue
            overload_ratio = (fac_load.get(f, 0) - demand) / demand
            facility_multiplier[f] = penalty_scale * max(0.0, overload_ratio)
        return facility_multiplier

    def run_lagrangian(self, distances, demands, demands_by_facility):
        """
        拉格朗日乘子驱动列生成（针对第一问）。
        在阶段一/阶段二列生成的基础上，额外引入 facility_multiplier 进行定价修正，
        多轮迭代后收敛路径池，再做最终两阶段 MIP。
        """
        t_start = time_module.time()
        print("=" * 66)
        print("问题1 列分解 + 乘子驱动列生成（Lagrangian）实验")
        print("=" * 66)

        ahp = AHPAnalyzer.compute(AHP_MATRIX)
        self.weights = tuple(ahp['weights'][i] for i in range(3))
        print(f"AHP 权重: w_tau={self.weights[0]:.4f}, "
              f"w_util={self.weights[1]:.4f}, w_fuel={self.weights[2]:.4f}")

        # 1) 构建定价宇宙
        print("\n[1] 构建定价宇宙（静态池超集）...")
        caps = SMOKE_CAPS if self.smoke else None
        universe, static_routes = PricingUniverse(distances).build(caps)
        print(f"  宇宙规模: {len(universe)} 条候选航线, 静态池: {len(static_routes)} 条")

        # 2) 初始化列池
        columns = self._seed_columns(static_routes, demands_by_facility)
        known_keys = {(c.route.route_id, c.pattern_key()) for c in columns}

        # 3) 阶段一基准列生成
        print("\n[2] 阶段一列生成（min T）...")
        master1 = MasterLP(demands_by_facility, stage=1, eps=self.eps)
        it1, obj1_0, obj1_end, add1 = self._cg_loop(
            columns, universe, demands_by_facility, 1, known_keys, master1,
            facility_multiplier=None)
        print(f"  阶段一列生成: {it1} 轮, LP {obj1_0:.2f} -> {obj1_end:.2f}, 新增 {add1} 条")

        # 4) 乘子驱动列生成（多轮）
        lag_stats = []
        for rd in range(1, self.lagrangian_rounds + 1):
            print(f"\n===== 乘子驱动列生成 第 {rd}/{self.lagrangian_rounds} 轮 =====")
            fac_mult = self._facility_multiplier_from_columns(columns, demands_by_facility)
            n_pos = sum(1 for v in fac_mult.values() if v > 0)
            print(f"  计算 facility multiplier 完成: 正乘子设施数={n_pos}/{len(fac_mult)}")

            master_tmp = MasterLP(demands_by_facility, stage=1, eps=self.eps)
            it_r, obj_r0, obj_r1, add_r = self._cg_loop(
                columns, universe, demands_by_facility, 1, known_keys, master_tmp,
                facility_multiplier=fac_mult)
            lag_stats.append({
                'round': rd, 'iters': it_r, 'obj0': obj_r0, 'obj1': obj_r1,
                'added': add_r, 'pool': len(columns), 'n_pos_mult': n_pos,
            })
            print(f"  本轮: {it_r} 轮, LP {obj_r0:.2f} -> {obj_r1:.2f}, 新增 {add_r} 条, 列池 {len(columns)}")

        self.cg_stats['lagrangian'] = lag_stats

        pool1_routes = self._routes_from_columns(columns)

        # 5) 阶段一 MIP
        print("\n[3] 阶段一 MIP（收敛路径池）...")
        solver = TwoStageSolverQ1(self.weights, eps=self.eps)
        solver.set_context(distances, demands)
        stage1_mip = solver._solve_stage1(pool1_routes, demands_by_facility)
        if stage1_mip is None:
            print("  错误: 阶段一 MIP 无可行解")
            return None
        s1_routes, s1_assign, s1_metrics = stage1_mip
        t_star = s1_metrics['total_time']
        norm = {
            'transit': s1_metrics['total_transit'],
            'wasted': s1_metrics['wasted'],
            'fuel': s1_metrics['total_fuel'],
        }
        print(f"  T* = {t_star} min, τ0={norm['transit']}, "
              f"W0={norm['wasted']:.1f}, F0={norm['fuel']:.1f}, "
              f"架次={s1_metrics['total_flights']}")

        # 6) 阶段二列生成（可选，仍保留乘子修正）
        columns2 = list(columns)
        if self.do_stage2_cg:
            print("\n[4] 阶段二列生成（AHP 加权二级目标, T ≤ (1+ε)T*）...")
            master2 = MasterLP(demands_by_facility, stage=2,
                               weights=self.weights, norm=norm,
                               t_star=t_star, eps=self.eps)
            fac_mult2 = self._facility_multiplier_from_columns(columns2, demands_by_facility)
            it2, obj2_0, obj2_end, add2 = self._cg_loop(
                columns2, universe, demands_by_facility, 2, known_keys, master2,
                facility_multiplier=fac_mult2)
            print(f"  阶段二列生成: {it2} 轮, LP {obj2_0:.6f} -> {obj2_end:.6f}, 新增 {add2} 条")
        else:
            print("\n[4] 跳过阶段二列生成（--no-stage2-cg）")

        pool2_routes = self._routes_from_columns(columns2)
        print(f"  最终路径池: {len(pool2_routes)} 条航线, {len(columns2)} 条装载列")

        # 7) 最终两阶段 MIP
        print("\n[5] 最终两阶段 ε-容差 + AHP 求解...")
        os.makedirs(self.out_dir, exist_ok=True)
        stage1_final, final_metrics, norm_final = solver.solve(
            pool2_routes, demands_by_facility, self.out_dir)
        if final_metrics is None:
            print("  错误: 最终两阶段求解失败")
            return None

        elapsed = time_module.time() - t_start
        result = {
            'stage1_mip': s1_metrics,
            'stage1_final': stage1_final,
            'final': final_metrics,
            'norm': norm,
            't_star': t_star,
            'pool1': len(pool1_routes),
            'pool2': len(pool2_routes),
            'columns1': len(columns),
            'columns2': len(columns2),
            'universe': len(universe),
            'cg_stats': self.cg_stats,
            'elapsed': elapsed,
        }
        print(f"\n[6] 乘子驱动列生成实验完成，总耗时 {elapsed:.1f}s")
        return result

    @staticmethod
    def _routes_from_columns(columns: List[Column]) -> List[Route]:
        """从列池提取去重后的航线列表（保持列池中首次出现顺序）"""
        routes = []
        seen = set()
        for col in columns:
            if col.route.route_id not in seen:
                seen.add(col.route.route_id)
                routes.append(col.route)
        return routes


def parse_metrics_txt(path):
    """解析 q1-metrics.txt 指标文件"""
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


def load_static_baseline():
    """
    读取静态池两阶段基线（output/Problem1 官方输出）

    返回:
        (baseline_final, baseline_stage1)
        baseline_final: 静态池两阶段最终指标
        baseline_stage1: 静态池阶段一锚点指标（无则用最终指标近似）
    """
    base_dir = os.path.join(PROJECT_DIR, 'output', 'Problem1')
    final_path = os.path.join(base_dir, 'q1-metrics.txt')
    if not os.path.exists(final_path):
        return None, None
    baseline_final = parse_metrics_txt(final_path)
    # 阶段一锚点优先取方法对比表中的"阶段一（仅最小化时间）"
    comp_path = os.path.join(base_dir, 'q1-method-comparison.csv')
    baseline_stage1 = None
    if os.path.exists(comp_path):
        with open(comp_path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row.get('name') == '阶段一（仅最小化时间）':
                    baseline_stage1 = {
                        'total_time': int(float(row['total_time'])),
                        'total_transit': int(float(row['total_transit'])),
                        'total_flights': int(row['total_flights']),
                        'total_fuel': float(row['total_fuel']),
                        'seat_util': float(row['seat_util']),
                        'wasted': float(row['wasted']),
                    }
                    break
    return baseline_final, baseline_stage1


def compute_ahp_score(metrics, anchor, weights):
    """按 AHP 权重计算归一化二级目标综合得分 Z2"""
    w_tau, w_util, w_fuel = weights
    # 指标字典中 wasted 可能缺省（OutputWriter 只给 seat_km/pass_km）
    wasted = metrics.get('wasted', metrics['seat_km'] - metrics['pass_km'])
    return (w_tau * metrics['total_transit'] / anchor['transit']
            + w_util * wasted / anchor['wasted']
            + w_fuel * metrics['total_fuel'] / anchor['fuel'])


def write_comparison(result, baseline_final, baseline_stage1, weights,
                     out_dir, final_name='列分解+两阶段（本实验）'):
    """
    写出列分解 vs 静态池两阶段对比表

    参数:
        result: 列分解实验结果字典
        baseline_final: 静态池两阶段最终指标
        baseline_stage1: 静态池阶段一锚点指标
        weights: AHP 权重
        out_dir: 输出目录
    """
    if baseline_stage1 is None:
        # 无静态基线锚点时用列分解自身阶段一作为统一锚点
        baseline_stage1 = result['stage1_final']
    anchor = {
        'transit': baseline_stage1['total_transit'],
        'wasted': baseline_stage1['wasted'],
        'fuel': baseline_stage1['total_fuel'],
    }

    rows = []
    if baseline_final is not None:
        rows.append({
            'name': '静态池两阶段（基线）',
            'total_time': baseline_final['total_time'],
            'total_transit': baseline_final['total_transit'],
            'total_flights': baseline_final['total_flights'],
            'total_fuel': baseline_final['total_fuel'],
            'seat_util': baseline_final['seat_util'],
            'wasted': baseline_final['wasted'],
            'z2': compute_ahp_score(baseline_final, anchor, weights),
        })
    final = result['final']
    rows.append({
        'name': final_name,
        'total_time': final['total_time'],
        'total_transit': final['total_transit'],
        'total_flights': final['total_flights'],
        'total_fuel': final['total_fuel'],
        'seat_util': final['seat_util'],
        'wasted': final['seat_km'] - final['pass_km'],
        'z2': compute_ahp_score(final, anchor, weights),
    })
    rows.append({
        'name': '列分解阶段一（仅 min T）',
        'total_time': result['stage1_final']['total_time'],
        'total_transit': result['stage1_final']['total_transit'],
        'total_flights': result['stage1_final']['total_flights'],
        'total_fuel': result['stage1_final']['total_fuel'],
        'seat_util': result['stage1_final']['seat_util'],
        'wasted': result['stage1_final']['wasted'],
        'z2': compute_ahp_score(result['stage1_final'], anchor, weights),
    })

    # CSV
    csv_path = os.path.join(out_dir, 'q1-column-comparison.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['name', 'total_time', 'total_transit', 'total_flights',
                    'total_fuel', 'seat_util', 'wasted', 'z2'])
        for row in rows:
            w.writerow([row['name'], row['total_time'], row['total_transit'],
                        row['total_flights'], row['total_fuel'],
                        row['seat_util'], row['wasted'], row['z2']])

    # TXT
    txt_path = os.path.join(out_dir, 'q1-column-comparison.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("问题1 列分解（列生成）+ 两阶段 实验对比\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"AHP 权重: w_tau={weights[0]:.4f}, w_util={weights[1]:.4f}, "
                f"w_fuel={weights[2]:.4f}, ε={EPS_TOL}\n")
        f.write(f"统一锚点(阶段一): T*={baseline_stage1['total_time']} min, "
                f"τ0={anchor['transit']}, W0={anchor['wasted']:.1f}, "
                f"F0={anchor['fuel']:.1f}\n\n")
        f.write("列生成统计:\n")
        for stage_name in ('stage1', 'stage2'):
            for s in result['cg_stats'][stage_name]:
                f.write(f"  {stage_name}: {s['iters']} 轮, "
                        f"LP obj {s['first_obj']:.2f} -> {s['last_obj']:.2f}, "
                        f"新增列 {s['added']} 条\n")
        f.write(f"  定价宇宙: {result['universe']} 条航线\n")
        f.write(f"  阶段一后路径池: {result['pool1']} 条\n")
        f.write(f"  最终路径池: {result['pool2']} 条 "
                f"(装载列 {result['columns2']} 条)\n\n")
        f.write("方法对比:\n")
        f.write(f"  {'方法':<24}{'T(min)':>8}{'τ(min)':>9}{'架次':>5}"
                f"{'燃油(kg)':>11}{'利用率':>8}{'Z2':>9}\n")
        f.write("  " + "-" * 76 + "\n")
        for row in rows:
            f.write(f"  {row['name']:<22}{row['total_time']:>8}"
                    f"{row['total_transit']:>9}{row['total_flights']:>5}"
                    f"{row['total_fuel']:>11.1f}"
                    f"{row['seat_util'] * 100:>7.2f}%"
                    f"{row['z2']:>9.4f}\n")
        f.write("\n说明: Z2 以静态池阶段一解为锚点归一化，越低越好；\n")
        f.write("      列分解路径池为静态池超集，故理论上最终 MIP 目标不劣于基线。\n")

    print(f"  对比文件: {csv_path}")
    print(f"  对比文件: {txt_path}")
    return rows


def plot_cg_convergence(result, save_path):
    """
    绘制列生成收敛曲线（LP 目标值随迭代变化）

    参数:
        result: 列分解实验结果字典
        save_path: 图片保存路径
    """
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {'stage1': 'steelblue', 'stage2': 'coral'}
    labels = {'stage1': '阶段一（min T）', 'stage2': '阶段二（AHP 加权）'}
    for stage_name in ('stage1', 'stage2'):
        for s in result['cg_stats'][stage_name]:
            iters = s['iters']
            objs = s.get('objs', [])
            if iters <= 0 or not objs:
                continue
            # 绘制每轮 LP 目标值的变化曲线
            ax.plot(range(1, len(objs) + 1), objs, marker='o',
                    color=colors[stage_name],
                    label=labels[stage_name])
            ax.annotate(f"{objs[0]:.2f} → {objs[-1]:.2f}",
                        xy=(len(objs), objs[-1]),
                        xytext=(6, 6), textcoords='offset points', fontsize=8)
    ax.set_xlabel('列生成迭代轮次')
    ax.set_ylabel('受限主问题 LP 目标值')
    ax.set_title('问题1 列分解收敛曲线')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  收敛图: {save_path}")


def main():
    """主函数：编排 Q1 列分解实验"""
    args = sys.argv[1:]
    smoke = '--smoke' in args
    limt = '--limT' in args
    lagrangian = '--lagrangian' in args
    do_stage2_cg = '--no-stage2-cg' not in args
    eps = EPS_TOL
    lagrangian_rounds = 3
    tl1, tl2 = 7200, 7200
    gap1, gap2 = 0.005, 0.01
    out_override = None
    for i, a in enumerate(args):
        if a == '--eps' and i + 1 < len(args):
            eps = float(args[i + 1])
        if a == '--tl1' and i + 1 < len(args):
            tl1 = int(args[i + 1])
        if a == '--tl2' and i + 1 < len(args):
            tl2 = int(args[i + 1])
        if a == '--gap1' and i + 1 < len(args):
            gap1 = float(args[i + 1])
        if a == '--gap2' and i + 1 < len(args):
            gap2 = float(args[i + 1])
        if a == '--out' and i + 1 < len(args):
            out_override = args[i + 1]
        if a == '--lagrangian-rounds' and i + 1 < len(args):
            lagrangian_rounds = int(args[i + 1])
    out_dir = os.path.join(PROJECT_DIR, 'output', 'Problem1_column_lagrangian' if lagrangian else 'Problem1_column')
    if smoke:
        out_dir = os.path.join(PROJECT_DIR, 'output', 'Problem1_column_smoke')
    if limt:
        out_dir = os.path.join(PROJECT_DIR, 'output', 'Problem1_column_limT')
    if out_override:
        out_dir = out_override

    print("=" * 66)
    print("问题1 列分解 + 两阶段实验" + ("（极限 T 模式）" if limt else ""))
    print(f"  冒烟模式: {smoke}, 极限T模式: {limt}, 拉格朗日模式: {lagrangian}, "
          f"阶段二列生成: {do_stage2_cg}, ε={eps}")
    print(f"  输出目录: {out_dir}")
    print("=" * 66)

    # 冒烟模式下使用 2h 时限
    if smoke:
        solver = ColumnDecompositionSolverQ1(
            out_dir, smoke=True, do_stage2_cg=do_stage2_cg, max_iters=5,
            limt_time_limit=7200, limt_rel_gap=0.01,
            limt2_time_limit=7200, limt2_rel_gap=0.01,
            final_name='列分解+极限T+两阶段（本实验）' if limt
            else '列分解+两阶段（本实验）',
            lagrangian_rounds=lagrangian_rounds)
    else:
        solver = ColumnDecompositionSolverQ1(
            out_dir, smoke=False, do_stage2_cg=do_stage2_cg,
            limt_time_limit=tl1, limt_rel_gap=gap1,
            limt2_time_limit=tl2, limt2_rel_gap=gap2,
            final_name='列分解+极限T+两阶段（本实验）' if limt
            else '列分解+两阶段（本实验）',
            lagrangian_rounds=lagrangian_rounds)
    solver.eps = eps

    # 1) 数据加载
    print("\n[0] 数据加载...")
    loader = DataLoader()
    distances = loader.load_distances()
    demands = loader.load_demands_q1()
    demands_by_facility = loader.count_demands(demands)
    print(f"  地点: {len(distances)}, 需求: {len(demands)}人, "
          f"设施: {len(demands_by_facility)}个")

    # 2) 运行列分解 + 两阶段（极限 T 模式：长时限+紧间隙压 T）
    baseline_final, baseline_stage1 = load_static_baseline()
    if lagrangian:
        result = solver.run_lagrangian(distances, demands, demands_by_facility)
    elif limt:
        result = solver.run_limT(distances, demands, demands_by_facility)
    else:
        result = solver.run(distances, demands, demands_by_facility)
    if result is None:
        print("错误: 实验失败")
        return 1

    # 3) 对比与输出
    print("\n[7] 对比与输出...")
    if baseline_final is not None:
        print(f"  静态池基线(官方): T={baseline_final['total_time']} min, "
              f"τ={baseline_final['total_transit']} min, "
              f"架次={baseline_final['total_flights']}, "
              f"燃油={baseline_final['total_fuel']:.1f} kg")
    write_comparison(result, baseline_final, baseline_stage1,
                     solver.weights, out_dir,
                     final_name='列分解+乘子驱动列生成（本实验）' if lagrangian else solver.final_name)

    # 4) 收敛图
    fig_path = os.path.join(PROJECT_DIR, 'figures', 'Problem1',
                            'q1_cg_convergence.png')
    plot_cg_convergence(result, fig_path)

    # 5) 摘要
    print("\n" + "=" * 66)
    print("列分解 + 两阶段 最终结果摘要" + ("（极限 T 模式）" if limt else ""))
    print("=" * 66)
    m = result['final']
    print(f"  总飞机使用时间: {m['total_time']} 分钟 ({m['total_time']/60:.2f}h)")
    print(f"  人员总在途时间: {m['total_transit']} 分钟")
    print(f"  总架次数: {m['total_flights']}")
    print(f"  总燃油消耗量: {fmt6(m['total_fuel'])} kg")
    print(f"  座位利用率: {m['seat_util']*100:.2f}%")
    print(f"  总耗时: {result['elapsed']:.1f}s")
    return 0


if __name__ == '__main__':
    sys.exit(main())
