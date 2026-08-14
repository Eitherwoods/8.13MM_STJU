# -*- coding: utf-8 -*-
"""
文件: problem2_solution.py
功能: 问题2求解 — 出海、海返与穿梭联合运输
方法: 两阶段法（启发式路径生成 + COPT集合划分MIP）
创新点:
  1. 加油中转点嵌入：当路径燃油不可行时，自动插入加油设施作为中转站
  2. 节省算法路径生成：基于Clarke-Wright节省值排序生成多设施路径
  3. 字典序多目标优化：按题目优先级 time→transit→seat_util→fuel 四层优化
  4. 起点感知需求约束：指定机场人员必须匹配，LAND人员可任意分配
  5. Dinkelbach迭代：将分数目标 max(pass_km/seat_km) 线性化为 max(pass_km-λ×seat_km)
  6. 航段级容量约束：出海/海返/穿梭三类需求在同一航段上的总人数不超过座位数（NEW）
  7. 三类需求统一建模：outbound + inbound + shuttle 在同一集合划分框架内求解（NEW）
  8. 穿梭OD对专用路径生成：确保所有穿梭需求对至少有一条覆盖路径（NEW）
项目: 2026策联杯 B题
"""

import csv
import os
import sys
import math
import time as time_module
from collections import defaultdict
from itertools import permutations

# ==================== 模块路径 ====================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VIBECODING_DIR = os.path.dirname(_THIS_DIR)
if _VIBECODING_DIR not in sys.path:
    sys.path.insert(0, _VIBECODING_DIR)
# problem12 子包路径（在同一文件夹内导入 problem1_solution）
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import coptpy as cp
from config import (
    AIRCRAFT, AIRCRAFT_TYPES, AIRPORTS, REFUEL_FACILITIES,
    MAX_LANDINGS, OUTPUT_DIR_Q2, DATA_DIR, fmt6,
    STOP_TIME_NO_REFUEL, STOP_TIME_REFUEL
)
from base_models import (
    DataLoader, FuelChecker, TSPSolver, Route,
    calculate_time, make_route
)

from problem1_solution import RouteGenerator


# ==================== Q2数据加载器 ====================
class Q2DataLoader(DataLoader):
    """
    Q2数据加载器：加载4000条需求并分类为出海、海返、穿梭
    继承DataLoader以复用load_distances等方法
    出海: origin=机场/LAND, destination=海上设施
    海返: origin=海上设施, destination=机场/LAND
    穿梭: origin=海上设施, destination=海上设施
    """

    def load_demands_q2(self):
        """加载Q2需求数据，返回 [{person_id, origin_id, destination_id}, ...]"""
        demands = []
        path = os.path.join(DATA_DIR, 'peopleQ2.csv')
        with open(path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                demands.append({
                    'person_id': row['person_id'],
                    'origin_id': row['origin_id'],
                    'destination_id': row['destination_id']
                })
        return demands

    @staticmethod
    def classify_demands(demands):
        """
        将需求分类为出海、海返、穿梭

        参数:
            demands: 原始需求列表

        返回:
            (outbound, inbound, shuttle) 三个列表
        """
        airport_set = set(AIRPORTS) | {'LAND'}
        outbound = []
        inbound = []
        shuttle = []

        for d in demands:
            o = d['origin_id']
            de = d['destination_id']
            if o in airport_set and de.startswith('F'):
                outbound.append(d)
            elif o.startswith('F') and de in airport_set:
                inbound.append(d)
            elif o.startswith('F') and de.startswith('F'):
                shuttle.append(d)

        return outbound, inbound, shuttle

    @staticmethod
    def group_outbound(outbound):
        """
        按目的地设施分组出海需求

        返回: {facility: {'total': n, 'by_origin': {origin: count}}}
        """
        result = {}
        for d in outbound:
            f = d['destination_id']
            if f not in result:
                result[f] = {'total': 0, 'by_origin': {}}
            result[f]['total'] += 1
            origin = d['origin_id']
            result[f]['by_origin'][origin] = result[f]['by_origin'].get(origin, 0) + 1
        return result

    @staticmethod
    def group_inbound(inbound):
        """
        按起点设施分组海返需求

        返回: {facility: {'total': n, 'by_dest': {dest: count}}}
        """
        result = {}
        for d in inbound:
            f = d['origin_id']
            if f not in result:
                result[f] = {'total': 0, 'by_dest': {}}
            result[f]['total'] += 1
            dest = d['destination_id']
            result[f]['by_dest'][dest] = result[f]['by_dest'].get(dest, 0) + 1
        return result

    @staticmethod
    def group_shuttle(shuttle):
        """
        按起点-终点对分组穿梭需求

        返回: {(f1, f2): {'total': n}}
        """
        result = {}
        for d in shuttle:
            key = (d['origin_id'], d['destination_id'])
            if key not in result:
                result[key] = {'total': 0}
            result[key]['total'] += 1
        return result


# ==================== Q2路由预处理器 ====================
class Q2RoutePreprocessor:
    """
    预计算Q2路由属性：航段级容量、在途时间、客座公里

    创新点6: 航段级容量约束
      同一架次中，出海/海返/穿梭人员在各航段上的人数之和不超过座位数。
      对路径 [A, S1, ..., SK, A] 的第k航段（path[k]→path[k+1]）：
        机上人数 = (航向后仍未下机的出海人员) + (已上机的海返人员) + (已上机未下机的穿梭人员)

    创新点7: 三类需求统一建模
      出海: 机场→设施，在途时间=到设施为止的累计时间
      海返: 设施→机场，在途时间=离设施后的剩余时间
      穿梭: 设施→设施，在途时间=两设施之间的时间
    """

    @staticmethod
    def preprocess(route, distances):
        """
        为路由计算Q2特有属性

        返回:
            dict with keys:
            - 'facility_pos': {facility: position_in_path}
            - 'time_to': {facility: 出海在途时间(分钟)}
            - 'time_from': {facility: 海返在途时间(分钟)}
            - 'dist_to': {facility: 出海沿程距离(km)}
            - 'dist_from': {facility: 海返沿程距离(km)}
            - 'shuttle_info': {(f1, f2): {'time': 穿梭在途时间, 'dist': 穿梭距离}}
            - 'segments': [(outbound_after, inbound_before, shuttle_crossing)] 每航段
        """
        path = route.path
        params = AIRCRAFT[route.aircraft_type]
        speed = params['speed']

        # 停靠时间（每个中间位置）
        stop_time_at = {}
        for i in range(1, len(path) - 1):
            if i in route.refuel_at:
                stop_time_at[i] = STOP_TIME_REFUEL
            else:
                stop_time_at[i] = STOP_TIME_NO_REFUEL

        # 累计距离（浮点，用于客座公里）
        dist_to_pos = [0.0] * len(path)
        cum_dist = 0.0
        for i in range(1, len(path)):
            cum_dist += distances[path[i - 1]][path[i]]
            dist_to_pos[i] = cum_dist

        # 航段飞行时间（分钟）：逐段累加后统一向上取整，
        # 避免“累计时间大数相减”引入浮点误差导致多取整1分钟
        def segment_minutes(i):
            return distances[path[i]][path[i + 1]] / speed * 60.0

        # 设施位置：路径中所有海上设施（含加油中转点中也是设施的）
        facility_pos = {}
        for i in range(1, len(path) - 1):
            f = path[i]
            if f.startswith('F'):
                facility_pos[f] = i

        # 出海/海返的在途时间和距离
        time_to = {}   # 出海: 机场→设施
        time_from = {}  # 海返: 设施→机场
        dist_to = {}
        dist_from = {}

        for f, pos in facility_pos.items():
            flight_to = sum(segment_minutes(i) for i in range(pos))
            stops_to = sum(stop_time_at.get(i, 0) for i in range(1, pos))
            time_to[f] = math.ceil(flight_to + stops_to)

            flight_from = sum(segment_minutes(i)
                              for i in range(pos, len(path) - 1))
            stops_from = sum(stop_time_at.get(i, 0)
                             for i in range(pos + 1, len(path) - 1))
            time_from[f] = math.ceil(flight_from + stops_from)

            dist_to[f] = dist_to_pos[pos]
            dist_from[f] = route.total_distance - dist_to_pos[pos]

        # 穿梭信息：{(f1, f2): {'time': t, 'dist': d}}
        shuttle_info = {}
        fac_list = list(facility_pos.keys())
        for i in range(len(fac_list)):
            for j in range(i + 1, len(fac_list)):
                f1 = fac_list[i]
                f2 = fac_list[j]
                pos1 = facility_pos[f1]
                pos2 = facility_pos[f2]
                if pos1 < pos2:
                    flight_between = sum(segment_minutes(i)
                                         for i in range(pos1, pos2))
                    stops_between = sum(stop_time_at.get(i, 0)
                                        for i in range(pos1 + 1, pos2))
                    t_between = math.ceil(flight_between + stops_between)
                    d_between = dist_to_pos[pos2] - dist_to_pos[pos1]
                    shuttle_info[(f1, f2)] = {'time': t_between, 'dist': d_between}

        # 航段容量信息
        segments = []
        for k in range(len(path) - 1):
            out_after = [f for f, p in facility_pos.items() if p > k]
            in_before = [f for f, p in facility_pos.items() if p <= k]
            shut_cross = [(f1, f2) for f1, p1 in facility_pos.items()
                          for f2, p2 in facility_pos.items()
                          if p1 <= k and p2 > k and p1 < p2]
            segments.append((out_after, in_before, shut_cross))

        return {
            'facility_pos': facility_pos,
            'time_to': time_to,
            'time_from': time_from,
            'dist_to': dist_to,
            'dist_from': dist_from,
            'shuttle_info': shuttle_info,
            'segments': segments
        }


# ==================== Q2路径生成器 ====================
class Q2RouteGenerator(RouteGenerator):
    """
    Q2路径生成器：继承Q1的节省算法路径生成，并添加穿梭OD对专用路径

    创新点8: 穿梭OD对专用路径生成
      确保所有有穿梭需求的设施对至少有一条覆盖路径
    """

    def generate_shuttle_routes(self, shuttle_pairs, start_id):
        """
        为穿梭OD对生成专用路径

        参数:
            shuttle_pairs: 穿梭OD对列表 [(f1, f2), ...]
            start_id: 起始路径编号

        返回:
            新生成的路径列表
        """
        routes = []
        counter = [start_id]

        def next_id():
            counter[0] += 1
            return counter[0]

        for f1, f2 in shuttle_pairs:
            for airport in AIRPORTS:
                best_order, _ = TSPSolver.best_order(airport, [f1, f2], self.distances)
                for atype in AIRCRAFT_TYPES:
                    r = make_route(next_id(), airport, atype, best_order, self.distances)
                    if r is not None:
                        routes.append(r)

        return routes

    def generate_all_q2(self, shuttle_pairs):
        """
        生成Q2全部候选路径（Q1路径 + 穿梭专用路径）

        参数:
            shuttle_pairs: 穿梭OD对列表

        返回:
            去重后的路径列表
        """
        # 先生成Q1路径
        routes = self.generate_all()
        max_id = max(r.route_id for r in routes) if routes else 0

        # 添加穿梭专用路径
        print("  [5/5] 生成穿梭OD对专用路径...")
        shuttle_routes = self.generate_shuttle_routes(shuttle_pairs, max_id)
        routes.extend(shuttle_routes)
        print(f"        穿梭路径: {len(shuttle_routes)} 条")

        # 去重
        routes = self._deduplicate(routes)
        print(f"  去重后总计: {len(routes)} 条")
        return routes


# ==================== Q2 COPT求解器 ====================
class Q2COPTSolver:
    """
    COPT MIP求解器：集合划分 + 三类需求 + 航段级容量 + 字典序四目标优化

    创新点6: 航段级容量约束
      对每个航段k，约束 机上人数 <= 座位数 × x[r]
      机上人数 = Σ出海(航向后设施) + Σ海返(航向前设施) + Σ穿梭(跨越该航段)

    创新点7: 三类需求统一建模
      yo[(r,f)]  — 出海: 到设施f的人数
      yi[(r,f)]  — 海返: 从设施f登机的人数
      ys[(r,f1,f2)] — 穿梭: 从f1到f2的人数

    创新点5: Dinkelbach迭代
      座位利用率 = pass_km / seat_km，线性化为 max(pass_km - λ×seat_km)
  """

    # Dinkelbach参数
    DINKELBACH_MAX_ITERS = 2        # 最大迭代次数
    DINKELBACH_TOL = 0.005          # 收敛阈值
    LAMBDA_INIT = 0.45              # 初始利用率估计
    TIME_LIMITS = [120, 180]        # 每次迭代的求解时间限制(秒)

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

    def solve(self, routes, precomputed, outbound_demand, inbound_demand, shuttle_demand):
        """
        求解集合划分MIP（Dinkelbach迭代 + COPT字典序多目标优化）

        参数:
            routes: 路径列表
            precomputed: {route_id: 预处理信息}
            outbound_demand: {facility: {'total', 'by_origin'}}
            inbound_demand: {facility: {'total', 'by_dest'}}
            shuttle_demand: {(f1,f2): {'total'}}

        返回:
            (selected_routes, yo_vals, yi_vals, ys_vals) 或 ([], {}, {}, {})
        """
        print("\n===== COPT MIP 求解 (Dinkelbach迭代) =====")

        lam = self.LAMBDA_INIT
        best_result = None
        best_util = -1.0

        for iteration in range(self.DINKELBACH_MAX_ITERS):
            tl = self.TIME_LIMITS[min(iteration, len(self.TIME_LIMITS) - 1)]
            print(f"\n--- Dinkelbach迭代 {iteration + 1}/{self.DINKELBACH_MAX_ITERS} "
                  f"(λ={lam:.4f}, 时间限制={tl}s) ---")

            result = self._solve_with_lambda(
                routes, precomputed, outbound_demand,
                inbound_demand, shuttle_demand, lam, tl)

            if result is None:
                print("  无可行解，终止迭代")
                break

            selected_routes, yo_vals, yi_vals, ys_vals, pass_km, seat_km = result
            actual_util = pass_km / seat_km if seat_km > 0 else 0.0

            print(f"  实际座位利用率: {actual_util:.4f} (λ={lam:.4f})")
            print(f"  pass_km={pass_km:.0f}, seat_km={seat_km:.0f}")

            # 记录最优解
            if actual_util > best_util:
                best_util = actual_util
                best_result = result

            # 检查收敛
            delta = abs(actual_util - lam)
            if delta < self.DINKELBACH_TOL:
                print(f"  Dinkelbach收敛! (|Δλ|={delta:.6f})")
                break

            # 更新λ
            lam = actual_util

        if best_result is None:
            return [], {}, {}, {}

        selected_routes, yo_vals, yi_vals, ys_vals, _, _ = best_result
        print(f"\n选中路径: {len(selected_routes)}")
        total_people = sum(yo_vals.values()) + sum(yi_vals.values()) + sum(ys_vals.values())
        print(f"分配人数: {total_people} (出海:{sum(yo_vals.values())}, "
              f"海返:{sum(yi_vals.values())}, 穿梭:{sum(ys_vals.values())})")
        print(f"最终座位利用率: {best_util:.4f}")
        return selected_routes, yo_vals, yi_vals, ys_vals

    def _solve_with_lambda(self, routes, precomputed, outbound_demand,
                           inbound_demand, shuttle_demand, lam, time_limit):
        """
        使用给定的λ构建并求解COPT模型

        返回:
            (selected_routes, yo_vals, yi_vals, ys_vals, pass_km, seat_km) 或 None
        """
        env = cp.Envr()
        model = env.createModel('Q2_SetPartitioning')
        model.setParam(cp.COPT.Param.TimeLimit, time_limit)
        model.setParam(cp.COPT.Param.RelGap, 0.10)
        model.setParam(cp.COPT.Param.LogToConsole, 0)

        # --- 创建变量 ---
        print("  创建变量...")
        x = {}    # x[route_id] = binary: 是否启用该路径
        yo = {}   # yo[(r, f)] = 出海到f的人数
        yi = {}   # yi[(r, f)] = 从f海返的人数
        ys = {}   # ys[(r, f1, f2)] = 从f1到f2穿梭的人数

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

        n_vars = len(x) + len(yo) + len(yi) + len(ys)
        print(f"  变量: {len(x)} binary + {len(yo)}+{len(yi)}+{len(ys)} integer = {n_vars}")

        # --- 约束1: 需求满足（等式） ---
        print("  添加需求约束...")
        for f, info in outbound_demand.items():
            terms = [yo[(r.route_id, f)] for r in routes
                     if (r.route_id, f) in yo]
            if terms:
                model.addConstr(cp.quicksum(terms) == info['total'])
            else:
                print(f"  警告: 设施 {f} 无出海覆盖路径!")

        for f, info in inbound_demand.items():
            terms = [yi[(r.route_id, f)] for r in routes
                     if (r.route_id, f) in yi]
            if terms:
                model.addConstr(cp.quicksum(terms) == info['total'])
            else:
                print(f"  警告: 设施 {f} 无海返覆盖路径!")

        for (f1, f2), info in shuttle_demand.items():
            terms = [ys[(r.route_id, f1, f2)] for r in routes
                     if (r.route_id, f1, f2) in ys]
            if terms:
                model.addConstr(cp.quicksum(terms) == info['total'])
            else:
                print(f"  警告: 穿梭对 ({f1},{f2}) 无覆盖路径!")

        # --- 约束2: 起点感知（不等式，保证指定机场需求由该机场路径服务） ---
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
        print("  添加航段容量约束...")
        n_cap_constrs = 0
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
                    n_cap_constrs += 1
        print(f"  容量约束: {n_cap_constrs} 条")

        # --- 设置字典序多目标 ---
        print("  设置字典序多目标优化...")

        # 目标0(最高): 最小化总飞机使用时间
        time_expr = cp.quicksum(r.time * x[r.route_id] for r in routes)
        model.setObjectiveN(0, time_expr, sense=cp.COPT.MINIMIZE,
                            priority=10, reltol=0.05)
        print("    [优先级10] 最小化总飞机使用时间")

        # 目标1: 最小化人员总在途时间
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
        if transit_terms:
            transit_expr = cp.quicksum(transit_terms)
            model.setObjectiveN(1, transit_expr, sense=cp.COPT.MINIMIZE,
                                priority=7, reltol=0.05)
            print("    [优先级7]  最小化人员总在途时间")

        # 目标2: 最大化座位利用率（Dinkelbach线性化）
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
        seatkm_expr = cp.quicksum(r.capacity * r.total_distance * x[r.route_id]
                                  for r in routes)
        util_expr = lam * seatkm_expr - passkm_expr
        model.setObjectiveN(2, util_expr, sense=cp.COPT.MINIMIZE,
                            priority=4, reltol=0.05)
        print(f"    [优先级4]  最大化座位利用率 (Dinkelbach: λ={lam:.4f})")

        # 目标3(最低): 最小化总燃油消耗量
        fuel_expr = cp.quicksum(r.fuel * x[r.route_id] for r in routes)
        model.setObjectiveN(3, fuel_expr, sense=cp.COPT.MINIMIZE,
                            priority=1, reltol=0.10)
        print("    [优先级1]  最小化总燃油消耗量")

        # --- 求解 ---
        print("  求解中（COPT自动按优先级分层优化）...")
        t0 = time_module.time()
        model.solve()
        t1 = time_module.time()

        if not self._has_solution(model):
            print(f"  错误: 求解状态={model.status}, 无可行解")
            return None

        status_map = {cp.COPT.OPTIMAL: "最优", cp.COPT.TIMEOUT: "时间限制内近似"}
        status_str = status_map.get(model.status, f"状态{model.status}")
        print(f"  求解完成: {status_str}, 耗时 {t1 - t0:.1f}s")

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

        # --- 计算pass_km和seat_km用于Dinkelbach更新 ---
        pass_km = 0.0
        seat_km = 0.0
        for r in selected_routes:
            pre = precomputed[r.route_id]
            seat_km += r.capacity * r.total_distance
            for f in pre['facility_pos']:
                n_out = yo_vals.get((r.route_id, f), 0)
                pass_km += n_out * pre['dist_to'].get(f, 0)
                n_in = yi_vals.get((r.route_id, f), 0)
                pass_km += n_in * pre['dist_from'].get(f, 0)
            for (f1, f2) in pre['shuttle_info']:
                n_shut = ys_vals.get((r.route_id, f1, f2), 0)
                pass_km += n_shut * pre['shuttle_info'][(f1, f2)]['dist']

        return selected_routes, yo_vals, yi_vals, ys_vals, pass_km, seat_km


# ==================== Q2结果输出器 ====================
class Q2OutputWriter:
    """结果输出器：生成CSV文件和计算指标"""

    def __init__(self, distances, demands, precomputed,
                 outbound, inbound, shuttle):
        self.distances = distances
        self.demands = demands
        self.precomputed = precomputed
        self.outbound = outbound
        self.inbound = inbound
        self.shuttle = shuttle
        # 人员查找表
        self.person_lookup = {p['person_id']: p for p in demands}

    def write_all(self, selected_routes, yo_vals, yi_vals, ys_vals, output_dir):
        """输出所有结果文件并返回指标"""
        os.makedirs(output_dir, exist_ok=True)

        # 分配具体人员
        person_assignments = self._assign_people(
            selected_routes, yo_vals, yi_vals, ys_vals)
        # 分配航班号
        flight_numbers = self._assign_flight_numbers(selected_routes)

        # 写CSV
        routes_path = os.path.join(output_dir, 'q2-routes.csv')
        self._write_routes(selected_routes, flight_numbers, routes_path)
        print(f"  路径文件: {routes_path}")

        assign_path = os.path.join(output_dir, 'q2-assignments.csv')
        self._write_assignments(person_assignments, flight_numbers, assign_path)
        print(f"  分配文件: {assign_path}")

        # 计算指标
        metrics = self._calc_metrics(
            selected_routes, yo_vals, yi_vals, ys_vals, person_assignments)
        metrics_path = os.path.join(output_dir, 'q2-metrics.txt')
        self._write_metrics(metrics, metrics_path)
        print(f"  指标文件: {metrics_path}")

        return metrics

    def _assign_people(self, selected_routes, yo_vals, yi_vals, ys_vals):
        """
        将具体人员分配到选中路径

        出海: 按目的地设施分组，优先匹配起点机场
        海返: 按起点设施分组，优先匹配终点机场
        穿梭: 按OD对分组，无需机场匹配
        """
        # 分组出海人员
        out_pending = defaultdict(list)
        for p in self.outbound:
            out_pending[p['destination_id']].append({
                'pid': p['person_id'],
                'origin': p['origin_id'],
                'done': False
            })
        for f in out_pending:
            out_pending[f].sort(key=lambda x: (x['origin'] == 'LAND', x['origin']))

        # 分组海返人员
        in_pending = defaultdict(list)
        for p in self.inbound:
            in_pending[p['origin_id']].append({
                'pid': p['person_id'],
                'dest': p['destination_id'],
                'done': False
            })
        for f in in_pending:
            in_pending[f].sort(key=lambda x: (x['dest'] == 'LAND', x['dest']))

        # 分组穿梭人员
        shut_pending = defaultdict(list)
        for p in self.shuttle:
            key = (p['origin_id'], p['destination_id'])
            shut_pending[key].append({
                'pid': p['person_id'],
                'done': False
            })

        result = {}  # {person_id: (route, pickup_pos, delivery_pos)}

        # 分配出海人员
        for route in selected_routes:
            pre = self.precomputed[route.route_id]
            for f, pos in pre['facility_pos'].items():
                n = yo_vals.get((route.route_id, f), 0)
                if n == 0 or f not in out_pending:
                    continue
                assigned = 0
                # 第一轮：起点匹配
                for p in out_pending[f]:
                    if assigned >= n:
                        break
                    if not p['done'] and p['origin'] == route.airport:
                        result[p['pid']] = (route, 0, pos)
                        p['done'] = True
                        assigned += 1
                # 第二轮：LAND补充
                for p in out_pending[f]:
                    if assigned >= n:
                        break
                    if not p['done'] and p['origin'] == 'LAND':
                        result[p['pid']] = (route, 0, pos)
                        p['done'] = True
                        assigned += 1

        # 分配海返人员
        for route in selected_routes:
            pre = self.precomputed[route.route_id]
            last_pos = len(route.path) - 1
            for f, pos in pre['facility_pos'].items():
                n = yi_vals.get((route.route_id, f), 0)
                if n == 0 or f not in in_pending:
                    continue
                assigned = 0
                # 第一轮：终点匹配
                for p in in_pending[f]:
                    if assigned >= n:
                        break
                    if not p['done'] and p['dest'] == route.airport:
                        result[p['pid']] = (route, pos, last_pos)
                        p['done'] = True
                        assigned += 1
                # 第二轮：LAND补充
                for p in in_pending[f]:
                    if assigned >= n:
                        break
                    if not p['done'] and p['dest'] == 'LAND':
                        result[p['pid']] = (route, pos, last_pos)
                        p['done'] = True
                        assigned += 1

        # 分配穿梭人员
        for route in selected_routes:
            pre = self.precomputed[route.route_id]
            for (f1, f2) in pre['shuttle_info']:
                n = ys_vals.get((route.route_id, f1, f2), 0)
                if n == 0 or (f1, f2) not in shut_pending:
                    continue
                pos1 = pre['facility_pos'][f1]
                pos2 = pre['facility_pos'][f2]
                assigned = 0
                for p in shut_pending[(f1, f2)]:
                    if assigned >= n:
                        break
                    if not p['done']:
                        result[p['pid']] = (route, pos1, pos2)
                        p['done'] = True
                        assigned += 1

        return result

    @staticmethod
    def _assign_flight_numbers(selected_routes):
        """按机型从1起连续编号"""
        flight_no_map = {}
        counters = defaultdict(int)
        for r in selected_routes:
            counters[r.aircraft_type] += 1
            flight_no_map[r.route_id] = counters[r.aircraft_type]
        return flight_no_map

    def _write_routes(self, selected_routes, flight_no_map, path):
        """写q2-routes.csv"""
        with open(path, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow(['aircraft_type', 'flight_no', 'stop_order', 'facility_id', 'refuel'])
            for r in selected_routes:
                fno = flight_no_map[r.route_id]
                for idx, fac in enumerate(r.path):
                    refuel = 1 if idx in r.refuel_at else 0
                    if fac in AIRPORTS:
                        refuel = 0
                    w.writerow([r.aircraft_type, fno, idx, fac, refuel])

    def _write_assignments(self, person_assignments, flight_no_map, path):
        """写q2-assignments.csv"""
        with open(path, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow(['person_id', 'aircraft_type', 'flight_no',
                        'pickup_stop_order', 'delivery_stop_order'])
            for p in self.demands:
                pid = p['person_id']
                if pid in person_assignments:
                    route, pickup, delivery = person_assignments[pid]
                    fno = flight_no_map[route.route_id]
                    w.writerow([pid, route.aircraft_type, fno, pickup, delivery])

    def _calc_metrics(self, selected_routes, yo_vals, yi_vals, ys_vals,
                      person_assignments):
        """计算五项指标"""
        # 1. 总飞机使用时间
        total_time = sum(r.time for r in selected_routes)
        # 2. 总在途时间
        total_transit = 0
        for pid, (route, pickup, delivery) in person_assignments.items():
            p = self.person_lookup[pid]
            o = p['origin_id']
            de = p['destination_id']
            pre = self.precomputed[route.route_id]
            # 判断需求类型并计算在途时间
            airport_set = set(AIRPORTS) | {'LAND'}
            if o in airport_set and de.startswith('F'):
                # 出海
                total_transit += pre['time_to'].get(de, 0)
            elif o.startswith('F') and de in airport_set:
                # 海返
                total_transit += pre['time_from'].get(o, 0)
            elif o.startswith('F') and de.startswith('F'):
                # 穿梭
                si = pre['shuttle_info'].get((o, de), {})
                total_transit += si.get('time', 0)
        # 3. 总架次数
        total_flights = len(selected_routes)
        # 4. 总燃油
        total_fuel = sum(r.fuel for r in selected_routes)
        # 5. 座位利用率
        pass_km = 0.0
        seat_km = 0.0
        for r in selected_routes:
            pre = self.precomputed[r.route_id]
            seat_km += r.capacity * r.total_distance
            for f in pre['facility_pos']:
                n_out = yo_vals.get((r.route_id, f), 0)
                pass_km += n_out * pre['dist_to'].get(f, 0)
                n_in = yi_vals.get((r.route_id, f), 0)
                pass_km += n_in * pre['dist_from'].get(f, 0)
            for (f1, f2) in pre['shuttle_info']:
                n_shut = ys_vals.get((r.route_id, f1, f2), 0)
                pass_km += n_shut * pre['shuttle_info'][(f1, f2)]['dist']
        seat_util = pass_km / seat_km if seat_km > 0 else 0

        return {
            'total_time': total_time,
            'total_transit': total_transit,
            'total_flights': total_flights,
            'total_fuel': total_fuel,
            'seat_util': seat_util,
            'pass_km': pass_km,
            'seat_km': seat_km
        }

    def _write_metrics(self, m, path):
        """写指标文件"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n问题2 求解结果指标\n" + "=" * 50 + "\n\n")
            f.write(f"总飞机使用时间: {m['total_time']} 分钟 ({m['total_time']/60:.2f} 小时)\n")
            f.write(f"人员总在途时间: {m['total_transit']} 分钟 ({m['total_transit']/60:.2f} 小时)\n")
            f.write(f"总架次数: {m['total_flights']}\n")
            f.write(f"总燃油消耗量: {fmt6(m['total_fuel'])} kg\n")
            f.write(f"座位利用率: {m['seat_util']*100:.2f}%\n")
            f.write(f"  客座公里: {fmt6(m['pass_km'])}\n")
            f.write(f"  可用座公里: {fmt6(m['seat_km'])}\n")


# ==================== 主函数 ====================
def main():
    """主函数：编排整体求解流程"""
    print("=" * 60)
    print("问题2: 出海、海返与穿梭联合运输 — COPT优化求解")
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
    print(f"  出海: {len(outbound)}人, 覆盖设施: {len(outbound_demand)}个")
    print(f"  海返: {len(inbound)}人, 覆盖设施: {len(inbound_demand)}个")
    print(f"  穿梭: {len(shuttle)}人, OD对: {len(shuttle_demand)}个")

    # 2. 路径生成
    print("\n[2] 路径生成...")
    t0 = time_module.time()
    shuttle_pairs = list(shuttle_demand.keys())
    gen = Q2RouteGenerator(distances)
    routes = gen.generate_all_q2(shuttle_pairs)
    t1 = time_module.time()
    print(f"  耗时: {t1 - t0:.1f}s")

    # 3. 路由预处理
    print("\n[3] 路由预处理...")
    t0 = time_module.time()
    precomputed = {}
    for r in routes:
        precomputed[r.route_id] = Q2RoutePreprocessor.preprocess(r, distances)
    t1 = time_module.time()
    print(f"  预处理: {len(routes)} 条路径, 耗时 {t1 - t0:.1f}s")

    # 统计变量规模
    n_yo = sum(1 for r in routes for f in precomputed[r.route_id]['facility_pos']
               if f in outbound_demand)
    n_yi = sum(1 for r in routes for f in precomputed[r.route_id]['facility_pos']
               if f in inbound_demand)
    n_ys = sum(1 for r in routes
               for (f1, f2) in precomputed[r.route_id]['shuttle_info']
               if (f1, f2) in shuttle_demand)
    print(f"  预计变量: {len(routes)} binary + {n_yo}+{n_yi}+{n_ys} integer "
          f"= {len(routes) + n_yo + n_yi + n_ys}")

    # 4. COPT求解
    print("\n[4] COPT MIP求解...")
    solver = Q2COPTSolver()
    selected, yo_vals, yi_vals, ys_vals = solver.solve(
        routes, precomputed, outbound_demand, inbound_demand, shuttle_demand)

    if not selected:
        print("错误: 无可行解!")
        return

    # 5. 结果输出
    print("\n[5] 结果输出...")
    writer = Q2OutputWriter(
        distances, demands, precomputed, outbound, inbound, shuttle)
    metrics = writer.write_all(
        selected, yo_vals, yi_vals, ys_vals, OUTPUT_DIR_Q2)

    # 6. 打印指标
    print("\n" + "=" * 50)
    print("求解结果:")
    print("=" * 50)
    print(f"  总飞机使用时间: {metrics['total_time']} 分钟 "
          f"({metrics['total_time']/60:.2f}h)")
    print(f"  人员总在途时间: {metrics['total_transit']} 分钟 "
          f"({metrics['total_transit']/60:.2f}h)")
    print(f"  总架次数: {metrics['total_flights']}")
    print(f"  总燃油消耗量: {fmt6(metrics['total_fuel'])} kg")
    print(f"  座位利用率: {metrics['seat_util']*100:.2f}%")
    print("=" * 50)


if __name__ == '__main__':
    main()
