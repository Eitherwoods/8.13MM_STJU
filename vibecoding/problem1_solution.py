# -*- coding: utf-8 -*-
"""
文件: problem1_solution.py
功能: 问题1求解 — 单向出海运输
方法: 两阶段法（启发式路径生成 + COPT集合划分MIP）
创新点:
  1. 加油中转点嵌入：当路径燃油不可行时，自动插入加油设施作为中转站
  2. 节省算法路径生成：基于Clarke-Wright节省值排序生成多设施路径
  3. 字典序多目标优化：按题目优先级 time→transit→seat_util→fuel 四层优化
  4. 起点感知需求约束：LAND人员可选任意机场，指定机场人员必须匹配
  5. Dinkelbach迭代：将分数目标 max(pass_km/seat_km) 线性化为 max(pass_km-λ×seat_km)
项目: 2026策联杯 B题
"""

import csv
import os
import math
import time as time_module
from collections import defaultdict

import coptpy as cp
from config import (
    AIRCRAFT, AIRCRAFT_TYPES, AIRPORTS, REFUEL_FACILITIES,
    MAX_LANDINGS, OUTPUT_DIR, DATA_DIR, fmt6
)
from base_models import (
    DataLoader, FuelChecker, TSPSolver, Route,
    calculate_time, calculate_transit_and_dist, make_route
)


# ==================== 路径生成器 ====================
class RouteGenerator:
    """
    路径生成器：使用节省算法和加油中转点嵌入生成候选路径池
    """

    def __init__(self, distances):
        self.distances = distances
        self.facilities = sorted([f for f in distances if f.startswith('F')])

    def generate_all(self):
        """生成所有候选路径并返回去重后的列表"""
        routes = []
        counter = [0]

        def next_id():
            counter[0] += 1
            return counter[0]

        # 1. 单设施路径
        print("  [1/4] 生成单设施路径...")
        single = self._gen_single(next_id)
        routes.extend(single)
        print(f"        可行: {len(single)} 条")

        # 2. 双设施路径
        print("  [2/4] 生成双设施路径...")
        r2 = self._gen_pair(next_id)
        routes.extend(r2)
        print(f"        可行: {len(r2)} 条")

        # 3. 三设施路径
        print("  [3/4] 生成三设施路径...")
        r3 = self._gen_triple(next_id, r2)
        routes.extend(r3)
        print(f"        可行: {len(r3)} 条")

        # 4. 四-五设施路径
        print("  [4/4] 生成四-五设施路径...")
        r45 = self._gen_quad_quint(next_id, r3)
        routes.extend(r45)
        print(f"        可行: {len(r45)} 条")

        # 去重
        routes = self._deduplicate(routes)
        print(f"  去重后总计: {len(routes)} 条")
        return routes

    def _gen_single(self, next_id):
        """生成单设施路径：枚举所有(机场, 设施, 机型)组合"""
        routes = []
        for airport in AIRPORTS:
            for fac in self.facilities:
                for atype in AIRCRAFT_TYPES:
                    r = make_route(next_id(), airport, atype, [fac], self.distances)
                    if r is not None:
                        routes.append(r)
        return routes

    def _gen_pair(self, next_id):
        """生成双设施路径：基于Clarke-Wright节省值排序"""
        routes = []
        for airport in AIRPORTS:
            # 计算所有设施对的节省值
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
            # 每个机场取前200对
            for s, f1, f2 in savings[:200]:
                best_order, _ = TSPSolver.best_order(airport, [f1, f2], self.distances)
                for atype in AIRCRAFT_TYPES:
                    r = make_route(next_id(), airport, atype, best_order, self.distances)
                    if r is not None:
                        routes.append(r)
        return routes

    def _gen_triple(self, next_id, routes_2):
        """生成三设施路径：扩展最优双设施路径"""
        routes = []
        # 按机场+设施集合分组，取每组最佳
        groups = defaultdict(list)
        for r in routes_2:
            key = (r.airport, frozenset(r.dest_facilities))
            groups[key].append(r)
        best_pairs = [min(g, key=lambda r: r.time) for g in groups.values()]

        count = 0
        for r2 in best_pairs:
            if count >= 300:
                break
            existing = list(r2.dest_facilities)
            for f3 in self.facilities:
                if f3 in r2.dest_facilities:
                    continue
                # 只考虑100km内的设施
                min_d = min(self.distances[f3][f] for f in existing)
                if min_d > 100:
                    continue
                best_order, _ = TSPSolver.best_order(
                    r2.airport, existing + [f3], self.distances)
                for atype in AIRCRAFT_TYPES:
                    r = make_route(next_id(), r2.airport, atype, best_order, self.distances)
                    if r is not None:
                        routes.append(r)
                        count += 1
        return routes

    def _gen_quad_quint(self, next_id, routes_3):
        """生成四-五设施路径：扩展最优三设施路径"""
        routes = []
        groups = defaultdict(list)
        for r in routes_3:
            key = (r.airport, frozenset(r.dest_facilities))
            groups[key].append(r)
        best_triples = [min(g, key=lambda r: r.time) for g in groups.values()]

        count = 0
        for r3 in best_triples:
            if count >= 200:
                break
            existing = list(r3.dest_facilities)
            for f4 in self.facilities:
                if f4 in r3.dest_facilities:
                    continue
                min_d = min(self.distances[f4][f] for f in existing)
                if min_d > 80:
                    continue
                best_order, _ = TSPSolver.best_order(
                    r3.airport, existing + [f4], self.distances)
                for atype in AIRCRAFT_TYPES:
                    r = make_route(next_id(), r3.airport, atype, best_order, self.distances)
                    if r is not None:
                        routes.append(r)
                        count += 1

        # 五设施路径（从四设施中扩展）
        groups4 = defaultdict(list)
        for r in routes:
            if len(r.dest_facilities) == 4:
                key = (r.airport, frozenset(r.dest_facilities))
                groups4[key].append(r)
        best_quads = [min(g, key=lambda r: r.time) for g in groups4.values()]

        for r4 in best_quads:
            if count >= 300:
                break
            existing = list(r4.dest_facilities)
            for f5 in self.facilities:
                if f5 in r4.dest_facilities:
                    continue
                min_d = min(self.distances[f5][f] for f in existing)
                if min_d > 60:
                    continue
                best_order, _ = TSPSolver.best_order(
                    r4.airport, existing + [f5], self.distances)
                for atype in AIRCRAFT_TYPES:
                    r = make_route(next_id(), r4.airport, atype, best_order, self.distances)
                    if r is not None:
                        routes.append(r)
                        count += 1

        return routes

    @staticmethod
    def _deduplicate(routes):
        """去重：相同(机场, 路径, 机型)只保留一条"""
        seen = set()
        unique = []
        for r in routes:
            key = (r.airport, tuple(r.path), r.aircraft_type)
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique


# ==================== COPT求解器 ====================
class COPTSolver:
    """
    COPT MIP求解器：集合划分 + 字典序四目标优化
    创新点3: 使用COPT setObjectiveN按题目优先级实现四层字典序优化
      - 优先级10: 最小化总飞机使用时间（主要考核目标）
      - 优先级7:  最小化人员总在途时间（重要因素）
      - 优先级4:  最大化座位利用率（Dinkelbach线性化: max pass_km/seat_km）
      - 优先级1:  最小化总燃油消耗量（其他因素）
    创新点4: 起点感知需求约束——指定机场人员必须由该机场路径服务
    创新点5: Dinkelbach迭代求解分数目标 pass_km/seat_km
      座位利用率 = pass_km / seat_km 是分数目标，无法直接作为MIP目标。
      Dinkelbach算法将其线性化为 max(pass_km - λ×seat_km)，通过迭代收敛λ。
    """

    # Dinkelbach参数
    DINKELBACH_MAX_ITERS = 3        # 最大迭代次数
    DINKELBACH_TOL = 0.002          # 收敛阈值
    LAMBDA_INIT = 0.45              # 初始利用率估计
    TIME_LIMITS = [300, 300, 600]   # 每次迭代的求解时间限制(秒)

    @staticmethod
    def _has_solution(model):
        """检查模型是否有可行解（OPTIMAL或TIMEOUT且有解）"""
        if model.status in (cp.COPT.OPTIMAL, cp.COPT.TIMEOUT,
                            cp.COPT.NODELIMIT, cp.COPT.INTERRUPTED):
            try:
                _ = model.objval
                return True
            except Exception:
                return False
        return False

    def solve(self, routes, demands_by_facility):
        """
        求解集合划分MIP（Dinkelbach迭代 + COPT字典序多目标优化）

        创新点5: Dinkelbach迭代
          座位利用率 = pass_km / seat_km 是分数目标
          线性化为: max(pass_km - λ×seat_km)，等价于 min(λ×seat_km - pass_km)
          迭代更新 λ = pass_km* / seat_km* 直至收敛

        返回:
            (selected_routes, assignments)
        """
        print("\n===== COPT MIP 求解 (Dinkelbach迭代) =====")

        lam = self.LAMBDA_INIT
        best_result = None
        best_util = -1.0

        for iteration in range(self.DINKELBACH_MAX_ITERS):
            tl = self.TIME_LIMITS[min(iteration, len(self.TIME_LIMITS) - 1)]
            print(f"\n--- Dinkelbach迭代 {iteration + 1}/{self.DINKELBACH_MAX_ITERS} "
                  f"(λ={lam:.4f}, 时间限制={tl}s) ---")

            result = self._solve_with_lambda(routes, demands_by_facility, lam, tl)
            if result is None:
                print("  无可行解，终止迭代")
                break

            selected_routes, assignments, pass_km, seat_km = result
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
            return [], {}

        selected_routes, assignments, _, _ = best_result
        print(f"\n选中路径: {len(selected_routes)}, 分配人数: {sum(assignments.values())}")
        print(f"最终座位利用率: {best_util:.4f}")
        return selected_routes, assignments

    def _solve_with_lambda(self, routes, demands_by_facility, lam, time_limit):
        """
        使用给定的λ构建并求解COPT模型

        第三目标: min(λ×seat_km - pass_km)
          等价于 max(pass_km - λ×seat_km)
          即Dinkelbach线性化后的分数目标 max(pass_km/seat_km)

        返回:
            (selected_routes, assignments, pass_km, seat_km) 或 None
        """
        env = cp.Envr()
        model = env.createModel('Q1_SetPartitioning')
        model.setParam(cp.COPT.Param.TimeLimit, time_limit)
        model.setParam(cp.COPT.Param.RelGap, 0.05)
        model.setParam(cp.COPT.Param.LogToConsole, 0)

        # --- 创建变量 ---
        print("  创建变量...")
        x = {}   # x[route_id] = binary: 是否启用该路径
        y = {}   # y[(route_id, facility)] = integer: 该路径运往该设施的人数

        for r in routes:
            x[r.route_id] = model.addVar(vtype=cp.COPT.BINARY, name=f'x_{r.route_id}')
            for f in r.dest_facilities:
                if f in demands_by_facility:
                    y[(r.route_id, f)] = model.addVar(
                        vtype=cp.COPT.INTEGER, lb=0, name=f'y_{r.route_id}_{f}')

        n_vars = len(x) + len(y)
        print(f"  变量: {len(x)} binary + {len(y)} integer = {n_vars}")

        # --- 约束1: 需求满足 ---
        print("  添加约束...")
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

        # --- 设置字典序多目标 ---
        print("  设置字典序多目标优化...")

        # 目标0(最高优先级): 最小化总飞机使用时间 — 主要考核目标
        time_expr = cp.quicksum(r.time * x[r.route_id] for r in routes)
        model.setObjectiveN(0, time_expr, sense=cp.COPT.MINIMIZE,
                            priority=10, reltol=0.01)
        print("    [优先级10] 最小化总飞机使用时间")

        # 目标1: 最小化人员总在途时间 — 重要因素
        transit_terms = []
        for r in routes:
            for f in r.dest_facilities:
                if (r.route_id, f) in y and f in r.transit_times:
                    transit_terms.append(r.transit_times[f] * y[(r.route_id, f)])
        if transit_terms:
            transit_expr = cp.quicksum(transit_terms)
            model.setObjectiveN(1, transit_expr, sense=cp.COPT.MINIMIZE,
                                priority=7, reltol=0.03)
            print("    [优先级7]  最小化人员总在途时间")

        # 目标2: 最大化座位利用率 — Dinkelbach线性化
        # seat_util = pass_km / seat_km (分数目标)
        # 线性化: min(λ×seat_km - pass_km) 等价于 max(pass_km - λ×seat_km)
        passkm_terms = []
        for r in routes:
            for f in r.dest_facilities:
                if (r.route_id, f) in y and f in r.dist_to_facility:
                    passkm_terms.append(r.dist_to_facility[f] * y[(r.route_id, f)])
        passkm_expr = cp.quicksum(passkm_terms) if passkm_terms else 0
        seatkm_expr = cp.quicksum(r.capacity * r.total_distance * x[r.route_id]
                                  for r in routes)
        util_expr = lam * seatkm_expr - passkm_expr
        model.setObjectiveN(2, util_expr, sense=cp.COPT.MINIMIZE,
                            priority=4, reltol=0.03)
        print(f"    [优先级4]  最大化座位利用率 (Dinkelbach: λ={lam:.4f})")

        # 目标3(最低优先级): 最小化总燃油消耗量 — 其他因素
        fuel_expr = cp.quicksum(r.fuel * x[r.route_id] for r in routes)
        model.setObjectiveN(3, fuel_expr, sense=cp.COPT.MINIMIZE,
                            priority=1, reltol=0.05)
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
        assignments = {}
        for r in routes:
            if x[r.route_id].x > 0.5:
                for f in r.dest_facilities:
                    if (r.route_id, f) in y:
                        val = int(round(y[(r.route_id, f)].x))
                        if val > 0:
                            assignments[(r.route_id, f)] = val

        # --- 计算pass_km和seat_km用于Dinkelbach更新 ---
        pass_km = 0.0
        seat_km = 0.0
        for r in selected_routes:
            seat_km += r.capacity * r.total_distance
            for f in r.dest_facilities:
                n = assignments.get((r.route_id, f), 0)
                pass_km += n * r.dist_to_facility.get(f, 0)

        return selected_routes, assignments, pass_km, seat_km


# ==================== 结果输出器 ====================
class OutputWriter:
    """结果输出器：生成CSV文件和计算指标"""

    def __init__(self, distances, demands, demands_by_facility):
        self.distances = distances
        self.demands = demands
        self.demands_by_facility = demands_by_facility
        # 预建人员查找表
        self.person_lookup = {p['person_id']: p for p in demands}

    def write_all(self, selected_routes, assignments, output_dir):
        """输出所有结果文件并返回指标"""
        os.makedirs(output_dir, exist_ok=True)

        # 分配具体人员
        person_assignments = self._assign_people(selected_routes, assignments)
        # 分配航班号
        flight_numbers = self._assign_flight_numbers(selected_routes)

        # 写CSV
        routes_path = os.path.join(output_dir, 'q1-routes.csv')
        self._write_routes(selected_routes, flight_numbers, routes_path)
        print(f"  路径文件: {routes_path}")

        assign_path = os.path.join(output_dir, 'q1-assignments.csv')
        self._write_assignments(person_assignments, flight_numbers, assign_path)
        print(f"  分配文件: {assign_path}")

        # 计算指标
        metrics = self._calc_metrics(selected_routes, assignments, person_assignments)
        metrics_path = os.path.join(output_dir, 'q1-metrics.txt')
        self._write_metrics(metrics, metrics_path)
        print(f"  指标文件: {metrics_path}")

        return metrics

    def _assign_people(self, selected_routes, assignments):
        """
        将具体人员分配到选中路径
        优先匹配起点，LAND人员补充
        """
        # 按设施分组
        pending = defaultdict(list)
        for p in self.demands:
            pending[p['destination_id']].append({
                'pid': p['person_id'],
                'origin': p['origin_id'],
                'done': False
            })
        # 特定机场优先
        for f in pending:
            pending[f].sort(key=lambda x: (x['origin'] == 'LAND', x['origin']))

        result = {}  # {person_id: (route, delivery_stop)}
        for route in selected_routes:
            for f in route.dest_facilities:
                n = assignments.get((route.route_id, f), 0)
                if n == 0:
                    continue
                delivery_stop = route.path.index(f)
                assigned = 0
                # 第一轮：起点匹配
                for p in pending[f]:
                    if assigned >= n:
                        break
                    if not p['done'] and p['origin'] == route.airport:
                        result[p['pid']] = (route, delivery_stop)
                        p['done'] = True
                        assigned += 1
                # 第二轮：LAND补充
                for p in pending[f]:
                    if assigned >= n:
                        break
                    if not p['done'] and p['origin'] == 'LAND':
                        result[p['pid']] = (route, delivery_stop)
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
        """写q1-routes.csv"""
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
        """写q1-assignments.csv"""
        with open(path, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow(['person_id', 'aircraft_type', 'flight_no',
                        'pickup_stop_order', 'delivery_stop_order'])
            for p in self.demands:
                pid = p['person_id']
                if pid in person_assignments:
                    route, del_stop = person_assignments[pid]
                    fno = flight_no_map[route.route_id]
                    w.writerow([pid, route.aircraft_type, fno, 0, del_stop])

    def _calc_metrics(self, selected_routes, assignments, person_assignments):
        """计算五项指标"""
        # 1. 总飞机使用时间
        total_time = sum(r.time for r in selected_routes)
        # 2. 总在途时间
        total_transit = 0
        for pid, (route, _) in person_assignments.items():
            f = self.person_lookup[pid]['destination_id']
            total_transit += route.transit_times.get(f, 0)
        # 3. 总架次数
        total_flights = len(selected_routes)
        # 4. 总燃油
        total_fuel = sum(r.fuel for r in selected_routes)
        # 5. 座位利用率
        pass_km = 0.0
        seat_km = 0.0
        for r in selected_routes:
            seat_km += r.capacity * r.total_distance
            for f in r.dest_facilities:
                n = assignments.get((r.route_id, f), 0)
                pass_km += n * r.dist_to_facility.get(f, 0)
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
            f.write("=" * 50 + "\n问题1 求解结果指标\n" + "=" * 50 + "\n\n")
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
    print("问题1: 单向出海运输 — COPT优化求解")
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
    t0 = time_module.time()
    gen = RouteGenerator(distances)
    routes = gen.generate_all()
    t1 = time_module.time()
    print(f"  耗时: {t1-t0:.1f}s")
    n_y = sum(len(r.dest_facilities) for r in routes)
    print(f"  预计变量: {len(routes)} + {n_y} = {len(routes)+n_y}")

    # 3. COPT求解
    print("\n[3] COPT MIP求解...")
    solver = COPTSolver()
    selected, assignments = solver.solve(routes, demands_by_facility)
    if not selected:
        print("错误: 无可行解!")
        return

    # 4. 结果输出
    print("\n[4] 结果输出...")
    writer = OutputWriter(distances, demands, demands_by_facility)
    metrics = writer.write_all(selected, assignments, OUTPUT_DIR)

    # 5. 打印指标
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
