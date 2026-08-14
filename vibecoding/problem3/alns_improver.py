# -*- coding: utf-8 -*-
"""
文件: alns_improver.py
功能: 自适应大邻域搜索(ALNS)改进器 — 为Q3单日MIP提供高质量初始解（热启动）
项目: 2026策联杯 B题 - 海上油田人员直升机运载计划编排

方法:
  在受限路径池上执行 ALNS，解的表示与 Q3DailySolver 的 MIP 变量对齐：
    - x[ri_idx]: 路径是否选中
    - yo[(ri_idx, f)] / yi[(ri_idx, f)] / ys[(ri_idx, f1, f2)]: 设施级人员分配
  破坏算子: 随机移除 / 低利用率移除 / 高时间成本移除
  修复算子: 贪心插入 / regret-2 插入
  接受准则: 模拟退火
  自适应层: 算子权重轮盘赌 + 按改进得分更新

设计动机:
  Q3 单日 MIP 在 60~150s 时限内最优性间隙高达 60%~90%，求解器难以自行找到好解。
  ALNS 在同样的路径池上快速搜索出一个高质量可行解，注入 COPT 作为 MipStart，
  为分支定界提供初始上界，从而压缩间隙、加速收敛。
"""

import math
import random
import time as time_module
import os
import sys
from collections import defaultdict
from datetime import datetime

# ==================== 模块路径 ====================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VIBECODING_DIR = os.path.dirname(_THIS_DIR)
if _VIBECODING_DIR not in sys.path:
    sys.path.insert(0, _VIBECODING_DIR)
_PROBLEM12_DIR = os.path.join(_VIBECODING_DIR, 'problem12')
if _PROBLEM12_DIR not in sys.path:
    sys.path.insert(0, _PROBLEM12_DIR)

from problem12.problem2_solution import Q2RoutePreprocessor
from problem3_solution import AIRPORTS


class AlnsSolution:
    """ALNS 解：选中路径子集 + 设施级人员分配"""

    def __init__(self, selected, yo, yi, ys):
        """
        参数:
            selected: 选中路径索引集合 {ri_idx}
            yo: 出海分配 {(ri_idx, facility): count}
            yi: 海返分配 {(ri_idx, facility): count}
            ys: 穿梭分配 {(ri_idx, f1, f2): count}
        """
        self.selected = set(selected)
        self.yo = dict(yo)
        self.yi = dict(yi)
        self.ys = dict(ys)

    def copy(self):
        """返回解的一个浅拷贝（集合与字典复制，便于破坏/修复）"""
        return AlnsSolution(set(self.selected), dict(self.yo), dict(self.yi), dict(self.ys))


class AlnsImprover:
    """ALNS 主类：构造、破坏、修复、自适应权重与模拟退火"""

    # 权重（与 Q3DailySolver 的加权目标保持一致，λ取 MIP 首轮 Dinkelbach 初值）
    W_TIME = 1.0
    W_TRANSIT = 0.01
    W_UTIL = 0.15
    W_FUEL = 0.005
    LAM = 0.45

    def __init__(self, routes_info, day_demands, distances, day,
                 random_seed=20260814):
        """
        参数:
            routes_info: 候选路径列表（与 Q3DailySolver 传入的同一对象）
            day_demands: 当日个人级需求列表
            distances: 距离矩阵
            day: 排班日期（date 对象）
        """
        self.routes_info = routes_info
        self.day_demands = day_demands
        self.distances = distances
        self.day = day
        self.rng = random.Random(random_seed)

        # 预计算所有路径的 precomputed（供分配与评估复用）
        self._ensure_precomputed()

        # 设施级需求聚合
        self._build_demand_aggregates()

        # 路径能力索引：每条路径能服务的出海/海返设施与穿梭 OD
        self._build_route_capability()

        # 机队分组（与 MIP 约束4 一致）
        self._build_fleet_groups()

        # 算子与权重
        self.destroy_ops = {
            'random': self._destroy_random,
            'low_util': self._destroy_low_util,
            'expensive': self._destroy_expensive,
        }
        self.repair_ops = {
            'greedy': self._repair_greedy,
            'regret2': self._repair_regret2,
        }
        self.destroy_weights = {k: 1.0 for k in self.destroy_ops}
        self.repair_weights = {k: 1.0 for k in self.repair_ops}
        self.destroy_scores = {k: 0.0 for k in self.destroy_ops}
        self.repair_scores = {k: 0.0 for k in self.repair_ops}
        self.destroy_calls = {k: 0 for k in self.destroy_ops}
        self.repair_calls = {k: 0 for k in self.repair_ops}

    # ==================== 预处理 ====================
    def _ensure_precomputed(self):
        """为每条路径补齐 precomputed（航段/在途/距离信息）"""
        for ri in self.routes_info:
            if 'precomputed' not in ri or ri['precomputed'] is None:
                ri['precomputed'] = Q2RoutePreprocessor.preprocess(
                    ri['route'], self.distances)

    def _build_demand_aggregates(self):
        """把个人级需求聚合成设施级需求（含起点感知信息）"""
        airport_set = set(AIRPORTS) | {'LAND'}
        self.ob_by_fac = defaultdict(lambda: {'total': 0, 'by_origin': defaultdict(int)})
        self.ib_by_fac = defaultdict(lambda: {'total': 0, 'by_dest': defaultdict(int)})
        self.sh_by_od = defaultdict(int)

        for d in self.day_demands:
            o, de = d['origin_id'], d['destination_id']
            if o in airport_set and de.startswith('F'):
                self.ob_by_fac[de]['total'] += 1
                self.ob_by_fac[de]['by_origin'][o] += 1
            elif o.startswith('F') and de in airport_set:
                self.ib_by_fac[o]['total'] += 1
                self.ib_by_fac[o]['by_dest'][de] += 1
            elif o.startswith('F') and de.startswith('F'):
                self.sh_by_od[(o, de)] += 1

        self.ob_facilities = sorted(self.ob_by_fac)
        self.ib_facilities = sorted(self.ib_by_fac)
        self.sh_ods = sorted(self.sh_by_od)

    def _build_route_capability(self):
        """建立路径能力索引：路径->可服务的设施/OD 集合"""
        airport_set = set(AIRPORTS) | {'LAND'}
        self.cap_ob = defaultdict(set)      # ri_idx -> {facility}
        self.cap_ib = defaultdict(set)      # ri_idx -> {facility}
        self.cap_sh = defaultdict(set)      # ri_idx -> {(f1, f2)}
        self.airport_of = {}                # ri_idx -> airport

        for idx, ri in enumerate(self.routes_info):
            r = ri['route']
            self.airport_of[idx] = r.airport
            for d in ri['compatible_demands']:
                o, de = d['origin_id'], d['destination_id']
                if o in airport_set and de.startswith('F'):
                    self.cap_ob[idx].add(de)
                elif o.startswith('F') and de in airport_set:
                    self.cap_ib[idx].add(o)
                elif o.startswith('F') and de.startswith('F'):
                    self.cap_sh[idx].add((o, de))

        # 反索引：设施/OD -> 能服务的路径集合（用于修复阶段快速查找）
        self.routes_by_ob = defaultdict(list)
        self.routes_by_ib = defaultdict(list)
        self.routes_by_sh = defaultdict(list)
        for idx, facs in self.cap_ob.items():
            for f in facs:
                self.routes_by_ob[f].append(idx)
        for idx, facs in self.cap_ib.items():
            for f in facs:
                self.routes_by_ib[f].append(idx)
        for idx, ods in self.cap_sh.items():
            for od in ods:
                self.routes_by_sh[od].append(idx)

    def _build_fleet_groups(self):
        """建立机场-机型分组与时间槽占用（与 MIP 约束4 一致）"""
        from problem3_solution import FLEET
        self.fleet_max = FLEET
        self.fleet_groups = defaultdict(list)
        for idx, ri in enumerate(self.routes_info):
            r = ri['route']
            self.fleet_groups[(r.airport, r.aircraft_type)].append(idx)

    # ==================== 路径与分配的基本量 ====================
    def _route_time(self, idx):
        """路径总飞机使用时间（分钟）"""
        return self.routes_info[idx]['route'].time

    def _route_fuel(self, idx):
        """路径总燃油（kg）"""
        return self.routes_info[idx]['route'].fuel

    def _route_seat_km(self, idx):
        """路径可用座公里"""
        r = self.routes_info[idx]['route']
        return r.capacity * r.total_distance

    def _segment_loads(self, solution):
        """计算每条选中路径每个航段的在机人数，返回 {(idx): [load0, load1, ...]}"""
        loads = {}
        for idx in solution.selected:
            pre = self.routes_info[idx]['precomputed']
            n_seg = len(pre['segments'])
            seg_load = [0] * n_seg
            for (ri, f), v in solution.yo.items():
                if ri != idx:
                    continue
                pos = pre['facility_pos'][f]
                # 出海：航段 0..pos-1 在机
                for k in range(pos):
                    seg_load[k] += v
            for (ri, f), v in solution.yi.items():
                if ri != idx:
                    continue
                pos = pre['facility_pos'][f]
                # 海返：航段 pos..n_seg-1 在机
                for k in range(pos, n_seg):
                    seg_load[k] += v
            for (ri, f1, f2), v in solution.ys.items():
                if ri != idx:
                    continue
                pos1 = pre['facility_pos'][f1]
                pos2 = pre['facility_pos'][f2]
                # 穿梭：航段 pos1..pos2-1 在机
                for k in range(pos1, pos2):
                    seg_load[k] += v
            loads[idx] = seg_load
        return loads

    def _check_capacity(self, solution):
        """检查航段容量，返回违规总量（0表示可行）"""
        loads = self._segment_loads(solution)
        violation = 0
        for idx, seg_load in loads.items():
            cap = self.routes_info[idx]['route'].capacity
            for v in seg_load:
                if v > cap:
                    violation += (v - cap)
        return violation

    def _check_fleet(self, solution):
        """检查机队约束，返回违规总量"""
        violation = 0
        for (airport, atype), indices in self.fleet_groups.items():
            max_count = self.fleet_max[airport][atype]
            slot_to_routes = defaultdict(list)
            for idx in indices:
                if idx in solution.selected:
                    for slot in self.routes_info[idx]['time_slots']:
                        slot_to_routes[slot].append(idx)
            for slot, ri_list in slot_to_routes.items():
                if len(ri_list) > max_count:
                    violation += len(ri_list) - max_count
        return violation

    def _check_coverage(self, solution):
        """检查需求覆盖，返回未覆盖的需求总量（0表示全覆盖）"""
        uncovered = 0
        ob_served = defaultdict(int)
        ib_served = defaultdict(int)
        sh_served = defaultdict(int)
        for (ri, f), v in solution.yo.items():
            ob_served[f] += v
        for (ri, f), v in solution.yi.items():
            ib_served[f] += v
        for (ri, f1, f2), v in solution.ys.items():
            sh_served[(f1, f2)] += v
        for f, info in self.ob_by_fac.items():
            uncovered += max(0, info['total'] - ob_served.get(f, 0))
        for f, info in self.ib_by_fac.items():
            uncovered += max(0, info['total'] - ib_served.get(f, 0))
        for od, n in self.sh_by_od.items():
            uncovered += max(0, n - sh_served.get(od, 0))
        return uncovered

    # ==================== 目标函数 ====================
    def evaluate(self, solution, penalty_cap=1e6, penalty_cov=1e6, penalty_fleet=1e6):
        """
        计算加权目标值（与 Q3DailySolver 一致）+ 约束惩罚。

        返回:
            float: 目标值（越小越好）
        """
        time_total = sum(self._route_time(i) for i in solution.selected)
        fuel_total = sum(self._route_fuel(i) for i in solution.selected)
        seat_km = sum(self._route_seat_km(i) for i in solution.selected)

        transit = 0.0
        pass_km = 0.0
        for (ri, f), v in solution.yo.items():
            pre = self.routes_info[ri]['precomputed']
            transit += pre['time_to'].get(f, 0) * v
            pass_km += pre['dist_to'].get(f, 0) * v
        for (ri, f), v in solution.yi.items():
            pre = self.routes_info[ri]['precomputed']
            transit += pre['time_from'].get(f, 0) * v
            pass_km += pre['dist_from'].get(f, 0) * v
        for (ri, f1, f2), v in solution.ys.items():
            pre = self.routes_info[ri]['precomputed']
            si = pre['shuttle_info'].get((f1, f2), {})
            transit += si.get('time', 0) * v
            pass_km += si.get('dist', 0) * v

        obj = (self.W_TIME * time_total
               + self.W_TRANSIT * transit
               + self.W_UTIL * (self.LAM * seat_km - pass_km)
               + self.W_FUEL * fuel_total)

        # 约束惩罚
        obj += penalty_cap * self._check_capacity(solution)
        obj += penalty_cov * self._check_coverage(solution)
        obj += penalty_fleet * self._check_fleet(solution)
        return obj

    # ==================== 贪心分配 ====================
    def _assign_greedy(self, solution):
        """
        给定选中路径集合，贪心把设施级需求分配到路径，满足航段容量。

        返回:
            (yo, yi, ys): 分配结果字典
        """
        yo, yi, ys = {}, {}, {}
        loads = {idx: [0] * len(self.routes_info[idx]['precomputed']['segments'])
                 for idx in solution.selected}

        # 出海：优先指定机场需求
        for f in self.ob_facilities:
            info = self.ob_by_fac[f]
            remaining = info['total']
            # 先处理指定机场（非 LAND）需求
            for origin, cnt in sorted(info['by_origin'].items(),
                                      key=lambda kv: kv[0] == 'LAND'):
                if origin == 'LAND' or cnt == 0:
                    continue
                for idx in self.routes_by_ob.get(f, []):
                    if idx not in solution.selected:
                        continue
                    if self.airport_of[idx] != origin:
                        continue
                    n_take = min(cnt, self._free_capacity(idx, f, 'ob', loads))
                    if n_take > 0:
                        self._apply_load(idx, f, 'ob', n_take, loads)
                        yo[(idx, f)] = yo.get((idx, f), 0) + n_take
                        cnt -= n_take
                        remaining -= n_take
                    if cnt == 0:
                        break
            # 剩余需求（含 LAND 与指定机场未满足的）由任意经过路径服务
            if remaining > 0:
                for idx in self.routes_by_ob.get(f, []):
                    if idx not in solution.selected:
                        continue
                    n_take = min(remaining, self._free_capacity(idx, f, 'ob', loads))
                    if n_take > 0:
                        self._apply_load(idx, f, 'ob', n_take, loads)
                        yo[(idx, f)] = yo.get((idx, f), 0) + n_take
                        remaining -= n_take
                    if remaining == 0:
                        break

        # 海返
        for f in self.ib_facilities:
            info = self.ib_by_fac[f]
            remaining = info['total']
            for dest, cnt in sorted(info['by_dest'].items(),
                                    key=lambda kv: kv[0] == 'LAND'):
                if dest == 'LAND' or cnt == 0:
                    continue
                for idx in self.routes_by_ib.get(f, []):
                    if idx not in solution.selected:
                        continue
                    if self.airport_of[idx] != dest:
                        continue
                    n_take = min(cnt, self._free_capacity(idx, f, 'ib', loads))
                    if n_take > 0:
                        self._apply_load(idx, f, 'ib', n_take, loads)
                        yi[(idx, f)] = yi.get((idx, f), 0) + n_take
                        cnt -= n_take
                        remaining -= n_take
                    if cnt == 0:
                        break
            if remaining > 0:
                for idx in self.routes_by_ib.get(f, []):
                    if idx not in solution.selected:
                        continue
                    n_take = min(remaining, self._free_capacity(idx, f, 'ib', loads))
                    if n_take > 0:
                        self._apply_load(idx, f, 'ib', n_take, loads)
                        yi[(idx, f)] = yi.get((idx, f), 0) + n_take
                        remaining -= n_take
                    if remaining == 0:
                        break

        # 穿梭
        for od in self.sh_ods:
            f1, f2 = od
            remaining = self.sh_by_od[od]
            for idx in self.routes_by_sh.get(od, []):
                if idx not in solution.selected:
                    continue
                pre = self.routes_info[idx]['precomputed']
                pos1 = pre['facility_pos'][f1]
                pos2 = pre['facility_pos'][f2]
                if pos1 >= pos2:
                    continue
                # 穿梭占用航段 pos1..pos2-1，取最小剩余容量
                cap = self.routes_info[idx]['route'].capacity
                free = min(cap - loads[idx][k] for k in range(pos1, pos2))
                n_take = min(remaining, free)
                if n_take > 0:
                    for k in range(pos1, pos2):
                        loads[idx][k] += n_take
                    ys[(idx, f1, f2)] = ys.get((idx, f1, f2), 0) + n_take
                    remaining -= n_take
                if remaining == 0:
                    break

        return yo, yi, ys

    def _free_capacity(self, idx, f, kind, loads):
        """计算路径 idx 对指定设施/类型需求的剩余容量"""
        pre = self.routes_info[idx]['precomputed']
        cap = self.routes_info[idx]['route'].capacity
        pos = pre['facility_pos'][f]
        if kind == 'ob':
            segs = range(pos)
        elif kind == 'ib':
            segs = range(pos, len(pre['segments']))
        else:
            segs = range(len(pre['segments']))
        return min(cap - loads[idx][k] for k in segs)

    def _apply_load(self, idx, f, kind, n, loads):
        """在 loads 上累加指定需求的在机人数"""
        pre = self.routes_info[idx]['precomputed']
        pos = pre['facility_pos'][f]
        if kind == 'ob':
            segs = range(pos)
        else:  # ib
            segs = range(pos, len(pre['segments']))
        for k in segs:
            loads[idx][k] += n

    # ==================== 初始解构造 ====================
    def build_initial_solution(self):
        """贪心构造初始可行解：逐设施/OD 选边际成本最低的路径"""
        selected = set()
        covered_ob = set()
        covered_ib = set()
        covered_sh = set()

        # 出海
        for f in self.ob_facilities:
            candidates = self.routes_by_ob.get(f, [])
            if not candidates:
                continue
            best_idx = min(candidates, key=lambda i: self._route_time(i))
            selected.add(best_idx)
            covered_ob.add(f)

        # 海返
        for f in self.ib_facilities:
            candidates = self.routes_by_ib.get(f, [])
            if not candidates:
                continue
            best_idx = min(candidates, key=lambda i: self._route_time(i))
            selected.add(best_idx)
            covered_ib.add(f)

        # 穿梭
        for od in self.sh_ods:
            candidates = self.routes_by_sh.get(od, [])
            if not candidates:
                continue
            best_idx = min(candidates, key=lambda i: self._route_time(i))
            selected.add(best_idx)
            covered_sh.add(od)

        yo, yi, ys = self._assign_greedy(AlnsSolution(selected, {}, {}, {}))
        sol = AlnsSolution(selected, yo, yi, ys)
        return sol

    # ==================== 破坏算子 ====================
    def _destroy_random(self, solution, q):
        """随机移除 q 条路径"""
        partial = solution.copy()
        to_remove = self.rng.sample(sorted(partial.selected), min(q, len(partial.selected)))
        for idx in to_remove:
            partial.selected.discard(idx)
        self._drop_assignments(partial, to_remove)
        return partial

    def _destroy_low_util(self, solution, q):
        """移除座位利用率最低的 q 条路径"""
        partial = solution.copy()
        scores = []
        for idx in partial.selected:
            pass_km = self._route_pass_km(idx, solution)
            seat_km = self._route_seat_km(idx)
            util = pass_km / seat_km if seat_km > 0 else 0.0
            scores.append((util, idx))
        scores.sort()
        to_remove = [idx for _, idx in scores[:q]]
        for idx in to_remove:
            partial.selected.discard(idx)
        self._drop_assignments(partial, to_remove)
        return partial

    def _destroy_expensive(self, solution, q):
        """移除单位服务人数时间成本最高的 q 条路径"""
        partial = solution.copy()
        scores = []
        for idx in partial.selected:
            persons = self._route_persons(idx, solution)
            cost = self._route_time(idx) / max(persons, 1)
            scores.append((cost, idx))
        scores.sort(reverse=True)
        to_remove = [idx for _, idx in scores[:q]]
        for idx in to_remove:
            partial.selected.discard(idx)
        self._drop_assignments(partial, to_remove)
        return partial

    def _route_pass_km(self, idx, solution):
        """路径的客座公里"""
        pre = self.routes_info[idx]['precomputed']
        pass_km = 0.0
        for f in pre['facility_pos']:
            pass_km += solution.yo.get((idx, f), 0) * pre['dist_to'].get(f, 0)
            pass_km += solution.yi.get((idx, f), 0) * pre['dist_from'].get(f, 0)
        for od, si in pre['shuttle_info'].items():
            pass_km += solution.ys.get((idx, od[0], od[1]), 0) * si['dist']
        return pass_km

    def _route_persons(self, idx, solution):
        """路径服务的总人数"""
        pre = self.routes_info[idx]['precomputed']
        n = 0
        for f in pre['facility_pos']:
            n += solution.yo.get((idx, f), 0) + solution.yi.get((idx, f), 0)
        for od in pre['shuttle_info']:
            n += solution.ys.get((idx, od[0], od[1]), 0)
        return n

    def _drop_assignments(self, solution, removed_indices):
        """移除与指定路径相关的所有人员分配"""
        removed = set(removed_indices)
        solution.yo = {k: v for k, v in solution.yo.items() if k[0] not in removed}
        solution.yi = {k: v for k, v in solution.yi.items() if k[0] not in removed}
        solution.ys = {k: v for k, v in solution.ys.items() if k[0] not in removed}

    # ==================== 修复算子 ====================
    def _repair_greedy(self, partial):
        """贪心修复：对未覆盖需求，逐个选边际时间成本最低的路径"""
        solution = partial.copy()
        # 重新分配（已有路径可能空出容量）
        yo, yi, ys = self._assign_greedy(solution)
        solution.yo, solution.yi, solution.ys = yo, yi, ys

        # 找未覆盖需求
        uncovered = self._uncovered_demands(solution)
        for kind, key in uncovered:
            best_idx, best_cost = None, None
            candidates = self._candidate_routes(kind, key)
            for idx in candidates:
                if idx in solution.selected:
                    continue
                cost = self._route_time(idx)
                if best_cost is None or cost < best_cost:
                    best_cost, best_idx = cost, idx
            if best_idx is not None:
                solution.selected.add(best_idx)

        # 加入路径后重新分配
        yo, yi, ys = self._assign_greedy(solution)
        solution.yo, solution.yi, solution.ys = yo, yi, ys
        return solution

    def _repair_regret2(self, partial):
        """regret-2 修复：优先处理"最优与次优成本差距最大"的未覆盖需求"""
        solution = partial.copy()
        yo, yi, ys = self._assign_greedy(solution)
        solution.yo, solution.yi, solution.ys = yo, yi, ys

        for _ in range(200):
            uncovered = self._uncovered_demands(solution)
            if not uncovered:
                break
            regrets = []
            for kind, key in uncovered:
                candidates = self._candidate_routes(kind, key)
                cands_out = [i for i in candidates if i not in solution.selected]
                if not cands_out:
                    continue
                costs = sorted(self._route_time(i) for i in cands_out)
                if len(costs) >= 2:
                    regret = costs[1] - costs[0]
                else:
                    regret = 1e9  # 只有一条可行路径，优先级最高
                regrets.append((regret, kind, key))
            if not regrets:
                break
            regrets.sort(reverse=True)
            _, kind, key = regrets[0]
            cands_out = [i for i in self._candidate_routes(kind, key)
                         if i not in solution.selected]
            if cands_out:
                best_idx = min(cands_out, key=lambda i: self._route_time(i))
                solution.selected.add(best_idx)

        yo, yi, ys = self._assign_greedy(solution)
        solution.yo, solution.yi, solution.ys = yo, yi, ys
        return solution

    def _uncovered_demands(self, solution):
        """返回未覆盖需求的 (kind, key) 列表"""
        ob_served = defaultdict(int)
        ib_served = defaultdict(int)
        sh_served = defaultdict(int)
        for (ri, f), v in solution.yo.items():
            ob_served[f] += v
        for (ri, f), v in solution.yi.items():
            ib_served[f] += v
        for (ri, f1, f2), v in solution.ys.items():
            sh_served[(f1, f2)] += v
        uncovered = []
        for f, info in self.ob_by_fac.items():
            if ob_served.get(f, 0) < info['total']:
                uncovered.append(('ob', f))
        for f, info in self.ib_by_fac.items():
            if ib_served.get(f, 0) < info['total']:
                uncovered.append(('ib', f))
        for od, n in self.sh_by_od.items():
            if sh_served.get(od, 0) < n:
                uncovered.append(('sh', od))
        return uncovered

    def _candidate_routes(self, kind, key):
        """返回能服务指定需求类型的候选路径索引"""
        if kind == 'ob':
            return self.routes_by_ob.get(key, [])
        if kind == 'ib':
            return self.routes_by_ib.get(key, [])
        return self.routes_by_sh.get(key, [])

    # ==================== 自适应与主循环 ====================
    def _pick_destroy(self):
        """按轮盘赌选择破坏算子"""
        return self._roulette(self.destroy_ops, self.destroy_weights)

    def _pick_repair(self):
        """按轮盘赌选择修复算子"""
        return self._roulette(self.repair_ops, self.repair_weights)

    @staticmethod
    def _roulette(ops, weights):
        """轮盘赌选择算子"""
        total = sum(weights.values())
        r = random.uniform(0, total)
        acc = 0.0
        for name in ops:
            acc += weights[name]
            if r <= acc:
                return name
        return list(ops)[-1]

    def _update_weights(self, op_name, is_destroy, score):
        """按改进得分更新算子权重（自适应层）"""
        if is_destroy:
            self.destroy_scores[op_name] += score
            self.destroy_calls[op_name] += 1
        else:
            self.repair_scores[op_name] += score
            self.repair_calls[op_name] += 1

    def _refresh_weights(self, reaction=0.4):
        """周期性用累计得分刷新权重（指数平滑）"""
        for name in self.destroy_ops:
            n = self.destroy_calls[name]
            if n == 0:
                continue
            avg = self.destroy_scores[name] / n
            self.destroy_weights[name] = max(0.05,
                (1 - reaction) * self.destroy_weights[name] + reaction * avg)
        for name in self.repair_ops:
            n = self.repair_calls[name]
            if n == 0:
                continue
            avg = self.repair_scores[name] / n
            self.repair_weights[name] = max(0.05,
                (1 - reaction) * self.repair_weights[name] + reaction * avg)
        # 重置累计
        self.destroy_scores = {k: 0.0 for k in self.destroy_ops}
        self.repair_scores = {k: 0.0 for k in self.repair_ops}
        self.destroy_calls = {k: 0 for k in self.destroy_ops}
        self.repair_calls = {k: 0 for k in self.repair_ops}

    def run(self, iterations=300, q_min=2, q_max=8,
            t0=5000.0, cooling=0.99, segment=30):
        """
        ALNS 主循环。

        参数:
            iterations: 最大迭代次数
            q_min/q_max: 破坏规模区间
            t0: 模拟退火初始温度
            cooling: 降温系数
            segment: 权重刷新周期（迭代数）
        """
        best = self.build_initial_solution()
        current = best.copy()
        best_obj = self.evaluate(best)
        current_obj = best_obj
        t = t0

        print(f"  ALNS 初始解: 路径={len(best.selected)}, 目标={best_obj:.2f}")

        for it in range(iterations):
            q = self.rng.randint(q_min, q_max)

            # 破坏 + 修复
            dname = self._pick_destroy()
            rname = self._pick_repair()
            partial = self.destroy_ops[dname](current, q)
            candidate = self.repair_ops[rname](partial)
            cand_obj = self.evaluate(candidate)

            # 模拟退火接受准则
            delta = cand_obj - current_obj
            # 模拟退火接受准则（delta<0 必接受，否则按温度概率接受）
            if delta < 0:
                accept = True
            elif t > 0:
                accept = self.rng.random() < math.exp(-delta / t)
            else:
                accept = False

            if accept:
                current = candidate
                current_obj = cand_obj
                if cand_obj < best_obj:
                    best = candidate.copy()
                    best_obj = cand_obj
                # 算子评分：全局改进 3 分，局部接受 1 分
                score = 3.0 if cand_obj < best_obj else 1.0
            else:
                score = 0.0

            self._update_weights(dname, True, score)
            self._update_weights(rname, False, score)

            # 降温
            t *= cooling

            # 周期性刷新权重
            if (it + 1) % segment == 0:
                self._refresh_weights()

            if (it + 1) % 50 == 0:
                print(f"  iter {it+1}/{iterations}: 当前={current_obj:.2f}, "
                      f"最优={best_obj:.2f}, 路径={len(best.selected)}")

        print(f"  ALNS 最终: 路径={len(best.selected)}, 目标={best_obj:.2f}")
        return best, best_obj

    # ==================== 热启动映射 ====================
    def to_mip_start(self, solution):
        """
        把 ALNS 解映射为索引级热启动数据，供 Q3DailySolver 注入 COPT MipStart。

        返回:
            dict: {
                'selected': {ri_idx},
                'yo': {(ri_idx, f): count},
                'yi': {(ri_idx, f): count},
                'ys': {(ri_idx, f1, f2): count},
            }
        """
        return {
            'selected': set(solution.selected),
            'yo': dict(solution.yo),
            'yi': dict(solution.yi),
            'ys': dict(solution.ys),
        }
