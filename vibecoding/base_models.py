# -*- coding: utf-8 -*-
"""
文件: base_models.py
功能: 公共模型模块，包含数据加载、燃油可行性检查、TSP求解、路径数据结构
项目: 2026策联杯 B题 - 海上油田人员直升机运载计划编排
创新点: 加油中转点嵌入、贪心最优加油策略
"""

import csv
import os
import math
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from itertools import permutations

from config import (
    AIRCRAFT, AIRCRAFT_TYPES, AIRPORTS, REFUEL_FACILITIES,
    MAX_LANDINGS, STOP_TIME_NO_REFUEL, STOP_TIME_REFUEL, DATA_DIR
)


# ==================== 数据加载器 ====================
class DataLoader:
    """数据加载器：加载距离矩阵和需求数据"""

    def load_distances(self) -> Dict[str, Dict[str, float]]:
        """加载距离矩阵，返回 dist[from][to] 格式的字典"""
        dist_path = os.path.join(DATA_DIR, 'distances.csv')
        distances = {}
        with open(dist_path, encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            locs = header[1:]  # 列名（地点编号）
            for row in reader:
                from_loc = row[0]
                distances[from_loc] = {}
                for i, to_loc in enumerate(locs):
                    distances[from_loc][to_loc] = float(row[i + 1])
        return distances

    def load_demands_q1(self) -> List[Dict]:
        """加载Q1需求数据，返回 [{person_id, origin, destination}, ...]"""
        demands = []
        path = os.path.join(DATA_DIR, 'peopleQ1.csv')
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
    def count_demands(demands: List[Dict]) -> Dict[str, Dict]:
        """按设施统计需求，返回 {facility: {'total': n, 'by_origin': {origin: count}}}"""
        result = {}
        for d in demands:
            f = d['destination_id']
            if f not in result:
                result[f] = {'total': 0, 'by_origin': {}}
            result[f]['total'] += 1
            origin = d['origin_id']
            result[f]['by_origin'][origin] = result[f]['by_origin'].get(origin, 0) + 1
        return result


# ==================== 燃油检查器 ====================
class FuelChecker:
    """
    燃油可行性检查器
    创新点1: 暴力枚举加油组合，找到最少加油次数的可行方案
    创新点2: 加油中转点嵌入——当路径燃油不可行时，尝试插入加油设施作为中转站
    """

    @staticmethod
    def check_fuel(path: List[str], aircraft_type: str,
                   distances: Dict[str, Dict[str, float]]) -> Tuple[bool, Set[int], float]:
        """
        检查路径的燃油可行性，返回最优加油策略

        参数:
            path: [airport, stop1, ..., stopK, airport]
            aircraft_type: 机型 T1/T2/T3
            distances: 距离矩阵

        返回:
            (feasible, refuel_indices, total_fuel)
            refuel_indices: 需要加油的停靠点在path中的索引集合
            total_fuel: 总油耗 (kg)
        """
        params = AIRCRAFT[aircraft_type]
        tank = params['tank']
        min_reserve = params['min_reserve']
        fuel_rate = params['fuel_rate']

        # 找出路径中可加油的位置（不含机场首尾）
        refuel_positions = [i for i in range(1, len(path) - 1)
                            if path[i] in REFUEL_FACILITIES]

        # 计算总航程
        total_distance = sum(distances[path[i]][path[i + 1]] for i in range(len(path) - 1))
        total_fuel = total_distance * fuel_rate

        best_refuel = None
        min_refuel_count = float('inf')

        # 暴力枚举所有加油组合（最多2^5=32种）
        n_refuel = len(refuel_positions)
        for mask in range(1 << n_refuel):
            refuel_set = set()
            for j in range(n_refuel):
                if mask & (1 << j):
                    refuel_set.add(refuel_positions[j])

            # 模拟飞行过程
            fuel = tank
            feasible = True
            for i in range(len(path) - 1):
                if i in refuel_set:
                    fuel = tank  # 加满油
                d = distances[path[i]][path[i + 1]]
                fuel -= d * fuel_rate
                if fuel < min_reserve - 1e-6:
                    feasible = False
                    break

            if feasible and len(refuel_set) < min_refuel_count:
                min_refuel_count = len(refuel_set)
                best_refuel = refuel_set

        if best_refuel is not None:
            return True, best_refuel, total_fuel
        return False, set(), total_fuel

    @staticmethod
    def try_insert_refuel(path: List[str], aircraft_type: str,
                          distances: Dict[str, Dict[str, float]],
                          max_landings: int = MAX_LANDINGS) -> Optional[Tuple[List[str], Set[int], float]]:
        """
        创新点2: 加油中转点嵌入
        当原始路径燃油不可行时，尝试在路径中插入加油设施作为中转站

        参数:
            path: 原始路径 [airport, stop1, ..., stopK, airport]
            aircraft_type: 机型
            distances: 距离矩阵
            max_landings: 最大海上着陆次数

        返回:
            (augmented_path, refuel_indices, total_fuel) 或 None
        """
        current_landings = len(path) - 2  # 不含首尾机场
        if current_landings >= max_landings:
            return None  # 无法插入更多停靠

        best_result = None
        best_distance = float('inf')

        # 尝试在每个位置插入每个加油设施
        for refuel_fac in REFUEL_FACILITIES:
            if refuel_fac in path:
                continue  # 已在路径中
            for pos in range(1, len(path)):  # 在中间位置插入
                new_path = path[:pos] + [refuel_fac] + path[pos:]
                if len(new_path) - 2 > max_landings:
                    continue

                feasible, refuel_set, fuel = FuelChecker.check_fuel(
                    new_path, aircraft_type, distances)
                if feasible:
                    total_dist = sum(distances[new_path[i]][new_path[i + 1]]
                                     for i in range(len(new_path) - 1))
                    if total_dist < best_distance:
                        best_distance = total_dist
                        best_result = (new_path, refuel_set, fuel)

        return best_result


# ==================== TSP求解器 ====================
class TSPSolver:
    """小规模TSP暴力求解器（≤5个设施）"""

    @staticmethod
    def best_order(airport: str, facilities: List[str],
                   distances: Dict[str, Dict[str, float]]) -> Tuple[List[str], float]:
        """
        求解从机场出发，经过所有设施后返回机场的最短路径

        参数:
            airport: 起降机场
            facilities: 待访问设施列表
            distances: 距离矩阵

        返回:
            (最优设施顺序, 总距离)
        """
        if len(facilities) == 0:
            return [], 0.0
        if len(facilities) == 1:
            f = facilities[0]
            return [f], distances[airport][f] + distances[f][airport]

        best_order = None
        best_dist = float('inf')

        for perm in permutations(facilities):
            d = distances[airport][perm[0]]
            for i in range(len(perm) - 1):
                d += distances[perm[i]][perm[i + 1]]
            d += distances[perm[-1]][airport]
            if d < best_dist:
                best_dist = d
                best_order = list(perm)

        return best_order, best_dist


# ==================== 路径数据结构 ====================
@dataclass
class Route:
    """
    路径数据结构
    存储一条完整架次的所有信息
    """
    route_id: int
    airport: str                          # 起降机场
    aircraft_type: str                     # 机型 T1/T2/T3
    path: List[str]                        # 完整路径 [airport, stop1, ..., stopK, airport]
    dest_facilities: Set[str]              # 目的地设施集合（有人下机的设施）
    refuel_at: Set[int]                    # 加油点在path中的索引集合
    time: int                              # 总时间（分钟，向上取整）
    fuel: float                            # 总油耗 (kg)
    total_distance: float                  # 总航程 (km)
    transit_times: Dict[str, int] = field(default_factory=dict)  # {facility: 在途时间(分钟)}
    dist_to_facility: Dict[str, float] = field(default_factory=dict)  # {facility: 机场到该设施的沿程距离}

    @property
    def num_landings(self) -> int:
        """海上着陆次数（不含首尾机场）"""
        return len(self.path) - 2

    @property
    def capacity(self) -> int:
        """座位数"""
        return AIRCRAFT[self.aircraft_type]['seats']

    @property
    def facility_list(self) -> List[str]:
        """目的地设施列表（路径中的顺序）"""
        return [f for f in self.path[1:-1] if f in self.dest_facilities]


# ==================== 辅助函数 ====================
def calculate_time(path: List[str], refuel_indices: Set[int], aircraft_type: str,
                   distances: Dict[str, Dict[str, float]]) -> int:
    """计算路径总时间（分钟，向上取整）"""
    params = AIRCRAFT[aircraft_type]
    speed = params['speed']  # km/h

    # 飞行时间
    flight_time = 0.0
    for i in range(len(path) - 1):
        d = distances[path[i]][path[i + 1]]
        flight_time += d / speed * 60.0  # 转为分钟

    # 停靠时间（不含首尾机场）
    stop_time = 0
    for i in range(1, len(path) - 1):
        if i in refuel_indices:
            stop_time += STOP_TIME_REFUEL
        else:
            stop_time += STOP_TIME_NO_REFUEL

    return math.ceil(flight_time + stop_time)


def calculate_transit_and_dist(path: List[str], refuel_indices: Set[int],
                               aircraft_type: str,
                               distances: Dict[str, Dict[str, float]],
                               dest_facilities: Set[str]) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    计算每个目的地设施的在途时间和沿程距离

    返回:
        (transit_times, dist_to_facility)
        transit_times: {facility: 在途时间(分钟)}
        dist_to_facility: {facility: 机场到该设施的沿程距离(km)}
    """
    params = AIRCRAFT[aircraft_type]
    speed = params['speed']

    transit_times = {}
    dist_to_facility = {}

    for f in dest_facilities:
        idx = path.index(f)

        # 飞行时间（从机场到该设施）
        flight_time = 0.0
        cum_dist = 0.0
        for i in range(idx):
            d = distances[path[i]][path[i + 1]]
            flight_time += d / speed * 60.0
            cum_dist += d

        # 停靠时间（该设施之前的停靠）
        stop_time = 0
        for i in range(1, idx):
            if i in refuel_indices:
                stop_time += STOP_TIME_REFUEL
            else:
                stop_time += STOP_TIME_NO_REFUEL

        transit_times[f] = math.ceil(flight_time + stop_time)
        dist_to_facility[f] = cum_dist

    return transit_times, dist_to_facility


def make_route(route_id: int, airport: str, aircraft_type: str,
               dest_facilities: List[str], distances: Dict[str, Dict[str, float]]) -> Optional[Route]:
    """
    创建路径对象（包含燃油检查和加油中转点嵌入）

    参数:
        route_id: 路径编号
        airport: 起降机场
        aircraft_type: 机型
        dest_facilities: 目的地设施有序列表
        distances: 距离矩阵

    返回:
        Route对象 或 None（不可行时）
    """
    # 构建基本路径
    path = [airport] + list(dest_facilities) + [airport]

    # 检查燃油可行性
    feasible, refuel_set, fuel = FuelChecker.check_fuel(path, aircraft_type, distances)

    if not feasible:
        # 创新点: 尝试加油中转点嵌入
        result = FuelChecker.try_insert_refuel(path, aircraft_type, distances)
        if result is not None:
            path, refuel_set, fuel = result
        else:
            return None  # 即使插入加油点也不可行

    # 计算路径指标
    total_distance = sum(distances[path[i]][path[i + 1]] for i in range(len(path) - 1))
    time = calculate_time(path, refuel_set, aircraft_type, distances)
    dest_set = set(dest_facilities)
    transit_times, dist_to_fac = calculate_transit_and_dist(
        path, refuel_set, aircraft_type, distances, dest_set)

    return Route(
        route_id=route_id,
        airport=airport,
        aircraft_type=aircraft_type,
        path=path,
        dest_facilities=dest_set,
        refuel_at=refuel_set,
        time=time,
        fuel=fuel,
        total_distance=total_distance,
        transit_times=transit_times,
        dist_to_facility=dist_to_fac
    )
