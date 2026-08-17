# -*- coding: utf-8 -*-
"""
文件: problem3_service_day.py
功能: 问题3最终求解器（方案一：服务日入列 + 事件驱动时刻 + 覆盖兜底）
项目: 2026策联杯 B题 - 海上油田人员直升机运载计划编排

设计思想（相对旧版 problem3_solution.py 的结构性改进）:
  S1. 服务日自由（规则A）: 跨天需求不再由贪心分配锁死日期，而是出现在
      其时间窗覆盖每一天的候选列中，由 MILP 全局决定服务日与拼座。
  S2. 事件驱动时刻: 候选起飞时刻取各需求可行窗口的最早时刻（上限 max_deps
      个，等距采样），替代旧版 14 个均匀偏移的粗网格。
  S3. 满员列装载: 每个候选时刻按 EDF 双序把兼容人员装到容量可行为止，
      MILP 只需做选列（集合覆盖），不再面对需求-路径二元变量矩阵。
  S4. 覆盖兜底: 任一非临时人员若未被任何列覆盖，为其生成专属直达列族，
      保证 3840 人全周覆盖，杜绝“失败日重分配不重解”造成的漏人。
  S5. 机队并发约束事件化: 只在活动架次数超过在场机数的时刻点加约束
      （区间图完美性），替代 84 个 10 分钟槽全量约束。
  S6. 阶段二逐日 prize-collecting: 日预算 = 该日阶段一总时间，逐日小 MIP
      最大化临时任务覆盖，秒级收敛。

运行（Powershell）:
  python problem3_service_day.py                  # 全量（阶段一 1800s）
  python problem3_service_day.py --smoke          # 冒烟（小列池+短时限）
  python problem3_service_day.py --time-limit 600 # 自定义阶段一时限
  python problem3_service_day.py --prune 40 10    # 自定义列池剪枝
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from itertools import combinations

import numpy as np
import scipy.sparse as sp
import coptpy as cp
from coptpy import COPT

# ==================== 模块路径 ====================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VIBECODING_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_DIR = os.path.dirname(_VIBECODING_DIR)
for _p in (_VIBECODING_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import AIRCRAFT, AIRPORTS, REFUEL_FACILITIES, MAX_LANDINGS  # noqa: E402

# ==================== 全局常量 ====================
DAY0 = datetime(2026, 8, 3, 0, 0)
PLANNING_DATES = [date(2026, 8, d) for d in range(3, 10)]
N_DAYS = 7

# 运营时段（日内分钟）
T0 = 360        # 06:00 最早起飞
DEP_END = 1080  # 18:00 最晚起飞
CURFEW = 1200   # 20:00 最晚返场
TURN = 30       # 机场周转 30 min

# 任务优先级（应急 > 增储上产 > 常规倒班 > 临时）
PRIO = {"emergency": 0, "production": 1, "shift": 2, "temporary": 3}

# 机队：机场 -> 机型 -> 数量（题目表3）
FLEET = {
    "A01": {"T1": 3, "T2": 3, "T3": 2},
    "A02": {"T1": 2, "T2": 4, "T3": 2},
    "A03": {"T1": 2, "T2": 3, "T3": 3},
}
N_AT = {(a, t): n for a, f in FLEET.items() for t, n in f.items()}
AIRCRAFT_LIST = [f"{a}-{t}-H{i:02d}" for a, f in FLEET.items()
                 for t, n in f.items() for i in range(1, n + 1)]
HOME = {x: x.split("-")[0] for x in AIRCRAFT_LIST}
AC_TYPE = {x: x.split("-")[1] for x in AIRCRAFT_LIST}

# 燃油参数换算为 0.1kg 整数（与题目口径一致，避免浮点误差）
RATE10 = {t: int(round(AIRCRAFT[t]["fuel_rate"] * 10)) for t in AIRCRAFT}
TANK10 = {t: int(round(AIRCRAFT[t]["tank"] * 10)) for t in AIRCRAFT}
RESERVE10 = {t: int(round(AIRCRAFT[t]["min_reserve"] * 10)) for t in AIRCRAFT}
SEATS = {t: AIRCRAFT[t]["seats"] for t in AIRCRAFT}
SPEED = {t: AIRCRAFT[t]["speed"] for t in AIRCRAFT}
STOP_MIN = 10
REFUEL_STOP_MIN = 20
TYPES = tuple(AIRCRAFT.keys())

OUTPUT_DIR_Q3 = os.path.join(_PROJECT_DIR, "output", "Problem3")


# ==================== 数据表示 ====================
class Q3Rec:
    """一条 Q3 需求（绝对分钟时间窗 + OD 类型）"""

    __slots__ = ("pid", "kind", "o", "d", "ep", "la", "temp", "prio")

    def __init__(self, pid, kind, o, d, ep, la, task):
        self.pid = pid
        self.kind = kind
        self.o = o
        self.d = d
        self.ep = ep          # 绝对分钟（自 08-03 00:00 起）
        self.la = la
        self.temp = task == "temporary"
        self.prio = PRIO[task]


class Q3Route:
    """一个可行架次：机场满油起飞，按 stops 顺序停靠并返回同一机场"""

    __slots__ = ("airport", "ac_type", "stops", "refuel")

    def __init__(self, airport, ac_type, stops, refuel):
        self.airport = airport
        self.ac_type = ac_type
        self.stops = stops          # 海上设施有序停靠（可重复）
        self.refuel = refuel        # 与 stops 平行的加油标记

    def seq(self):
        """完整节点序列：机场 -> 各停靠 -> 机场"""
        return (self.airport, *self.stops, self.airport)

    def n_landings(self):
        """海上着陆次数（不含首尾机场）"""
        return len(self.stops)


# ==================== 数据加载 ====================
def _find_data_dir():
    """定位附件目录（目录名含全角引号，需动态查找）"""
    topic_dir = os.path.join(_PROJECT_DIR, "题目和思路")
    for name in os.listdir(topic_dir):
        full = os.path.join(topic_dir, name)
        if os.path.isdir(full) and "附件" in name:
            return full
    raise FileNotFoundError("未找到附件目录")


def load_data():
    """加载距离矩阵与 4000 条 Q3 需求，返回 (distances, recs)"""
    data_dir = _find_data_dir()
    distances = {}
    with open(os.path.join(data_dir, "distances.csv"), encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        locs = header[1:]
        for row in reader:
            distances[row[0]] = {}
            for i, to_loc in enumerate(locs):
                distances[row[0]][to_loc] = float(row[i + 1])

    recs = []
    with open(os.path.join(data_dir, "peopleQ3.csv"), encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["person_id"]
            o = row["origin_id"]
            d = row["destination_id"]
            ep_dt = datetime.strptime(row["earliest_pickup_time"].strip(), "%Y-%m-%d %H:%M")
            la_dt = datetime.strptime(row["latest_arrival_time"].strip(), "%Y-%m-%d %H:%M")
            kind = "S" if (o.startswith("F") and d.startswith("F")) else (
                "O" if d.startswith("F") else "B")
            recs.append(Q3Rec(
                pid, kind, o, d,
                int((ep_dt - DAY0).total_seconds() // 60),
                int((la_dt - DAY0).total_seconds() // 60),
                row["task_type"],
            ))
    recs.sort(key=lambda r: r.pid)
    return distances, recs


# ==================== 架次构建（燃油可行性） ====================
def _flight_min(ac_type, dist_km):
    """航段飞行分钟 = ceil(60*距离/速度)"""
    return (60 * int(dist_km) + SPEED[ac_type] - 1) // SPEED[ac_type]


def _fuel_profile(route, distances):
    """逐到达点剩余油量（0.1kg），起飞满油，加油站检查余油后加满"""
    fuel = TANK10[route.ac_type]
    seq = route.seq()
    prof = []
    for k in range(1, len(seq)):
        fuel -= RATE10[route.ac_type] * int(distances[seq[k - 1]][seq[k]])
        prof.append(fuel)
        if k <= len(route.stops) and route.refuel[k - 1]:
            fuel = TANK10[route.ac_type]
    return prof


def _fuel_ok(route, distances):
    """每个到达点（含返场）剩余油量 >= 最低安全余油"""
    reserve = RESERVE10[route.ac_type]
    return all(f >= reserve for f in _fuel_profile(route, distances))


def _best_refuel_pattern(airport, ac_type, stops, distances):
    """固定停靠顺序下自动推导加油决策：最少加油次数的可行方案"""
    cand = [i for i, s in enumerate(stops) if s in REFUEL_FACILITIES]
    for r in range(0, len(cand) + 1):
        for combo in combinations(cand, r):
            flags = tuple(i in combo for i in range(len(stops)))
            route = Q3Route(airport, ac_type, stops, flags)
            if _fuel_ok(route, distances):
                return flags
    return None


def build_route(airport, ac_type, stops, distances, max_insert=2, _cache={}):
    """构造可行架次：先试原停靠上的加油决策，再尝试插入 1~2 个加油站。

    返回 (Q3Route 或 None, 是否插入过加油站)。结果按 (airport, ac_type, stops) 缓存。
    """
    key = (airport, ac_type, stops)
    if key in _cache:
        return _cache[key]
    flags = _best_refuel_pattern(airport, ac_type, stops, distances)
    if flags is not None:
        result = (Q3Route(airport, ac_type, stops, flags), False)
        _cache[key] = result
        return result

    best = None
    best_t = None

    def consider(cand_stops):
        """评估一个候选停靠序列是否可行，保留总时间最短者"""
        nonlocal best, best_t
        fl = _best_refuel_pattern(airport, ac_type, cand_stops, distances)
        if fl is not None:
            r = Q3Route(airport, ac_type, cand_stops, fl)
            t = route_time(r, distances)
            if best is None or t < best_t:
                best, best_t = r, t

    n = len(stops)
    if max_insert >= 1 and n < MAX_LANDINGS:
        for rf in REFUEL_FACILITIES:
            for pos in range(n + 1):
                consider(stops[:pos] + (rf,) + stops[pos:])
    if best is None and max_insert >= 2 and n <= MAX_LANDINGS - 2:
        seen = set()
        for r1 in REFUEL_FACILITIES:
            for p1 in range(n + 1):
                s1 = stops[:p1] + (r1,) + stops[p1:]
                for r2 in REFUEL_FACILITIES:
                    for p2 in range(n + 2):
                        s2 = s1[:p2] + (r2,) + s1[p2:]
                        if s2 in seen:
                            continue
                        seen.add(s2)
                        consider(s2)
    if best is None:
        _cache[key] = (None, False)
        return None, False
    _cache[key] = (best, True)
    return best, True


def route_time(route, distances):
    """架次使用时间 = Σ飞行分钟 + Σ停靠分钟"""
    seq = route.seq()
    fly = sum(_flight_min(route.ac_type, distances[seq[k - 1]][seq[k]])
              for k in range(1, len(seq)))
    stop = sum(REFUEL_STOP_MIN if rf else STOP_MIN for rf in route.refuel)
    return fly + stop


def route_fuel10(route, distances):
    """架次总油耗（0.1kg 整数）"""
    seq = route.seq()
    return sum(RATE10[route.ac_type] * int(distances[seq[k - 1]][seq[k]])
               for k in range(1, len(seq)))


def _stop_min_of(route, k):
    """第 k 个停靠站的停时（加油 20min，否则 10min）"""
    return REFUEL_STOP_MIN if route.refuel[k - 1] else STOP_MIN


def profile(route, dep, distances):
    """按日内起飞时刻 dep 计算各节点到达时刻（arr[0]=dep, arr[-1]=返场时刻）"""
    seq = route.seq()
    arr = [dep]
    t = dep
    for k in range(1, len(seq)):
        t += _flight_min(route.ac_type, distances[seq[k - 1]][seq[k]])
        arr.append(t)
        if k <= len(route.stops):
            t += _stop_min_of(route, k)
    return arr, arr[-1]


def _first_pos(route, fac, after=0):
    """返回设施 fac 在停靠序列中第 after 个之后首次出现的位置（1 起）"""
    for i, s in enumerate(route.stops, 1):
        if s == fac and i > after:
            return i
    return None


def rec_feasible(rec, route, arr, day):
    """人员 rec 可否由 day 日、时刻剖面 arr 的架次承运，返回 (pickup, delivery) 或 None"""
    base = day * 1440
    n = len(route.stops)
    if rec.kind == "O":
        if rec.o not in ("LAND", route.airport):
            return None
        dl = _first_pos(route, rec.d)
        if dl is None or base + arr[0] < rec.ep or base + arr[dl] > rec.la:
            return None
        return (0, dl)
    if rec.kind == "B":
        if rec.d not in ("LAND", route.airport):
            return None
        pk = _first_pos(route, rec.o)
        if (pk is None or base + arr[pk] + _stop_min_of(route, pk) < rec.ep
                or base + arr[n + 1] > rec.la):
            return None
        return (pk, n + 1)
    pk = _first_pos(route, rec.o)
    dl = _first_pos(route, rec.d, after=pk) if pk else None
    if pk is None or dl is None:
        return None
    if base + arr[pk] + _stop_min_of(route, pk) < rec.ep or base + arr[dl] > rec.la:
        return None
    return (pk, dl)


def dep_window(rec, route, day, off, distances):
    """人员 rec 在架次（剖面 off）上对起飞时刻的可行区间 [lo, hi]（日内分钟）"""
    base = day * 1440
    ep_d, la_d = rec.ep - base, rec.la - base
    hard_hi = min(DEP_END, CURFEW - off[-1])
    if rec.kind == "O":
        if rec.o not in ("LAND", route.airport):
            return None
        dl = _first_pos(route, rec.d)
        if dl is None:
            return None
        lo, hi = ep_d, la_d - off[dl]
    elif rec.kind == "B":
        if rec.d not in ("LAND", route.airport):
            return None
        pk = _first_pos(route, rec.o)
        if pk is None:
            return None
        lo, hi = ep_d - (off[pk] + _stop_min_of(route, pk)), la_d - off[-1]
    else:
        pk = _first_pos(route, rec.o)
        dl = _first_pos(route, rec.d, after=pk) if pk else None
        if pk is None or dl is None:
            return None
        lo, hi = ep_d - (off[pk] + _stop_min_of(route, pk)), la_d - off[dl]
    lo, hi = max(lo, T0), min(hi, hard_hi)
    return (lo, hi) if lo <= hi else None


def _onboard_per_leg(route, loads):
    """逐航段在机人数：loads 为 [(rec, pickup, delivery)]，航段编号 1..n+1"""
    n_leg = len(route.stops) + 1
    cnt = [0] * n_leg
    for item in loads:
        if len(item) == 3:
            pk, dl = item[1], item[2]
        else:
            pk, dl = item
        for k in range(pk + 1, dl + 1):
            cnt[k - 1] += 1
    return cnt


def _capacity_ok(route, loads, add=None):
    """检查给定装载（可追加一名 add）是否满足全部航段容量"""
    all_loads = list(loads) + ([add] if add else [])
    seats = SEATS[route.ac_type]
    return all(o <= seats for o in _onboard_per_leg(route, all_loads))


def _nearest_facilities(fac, k, distances):
    """距 fac 最近的 k 座其他海上设施"""
    others = [g for g in distances if g.startswith("F") and g != fac]
    others.sort(key=lambda g: distances[fac][g])
    return others[:k]


# ==================== 事件驱动列生成 ====================
def enum_day_columns(distances, day, recs, k_pair=3, max_deps=12, k_triple=0):
    """生成 day 日的候选列池。

    候选起飞时刻 = {06:00} ∪ 各可承运人员窗口最早时刻（事件驱动，禁止均匀粗网格），
    超过 max_deps 个时等距子采样。装载顺序 mandatory 优先、temporary 填残座，
    双序（EDF + ep 降序）保证紧窗与晚窗人员都能被覆盖。
    支持扩展：k_pair 最近邻双设施 + k_triple 最近邻三设施（k_triple>0 时启用）。
    返回 [(route, dep, loads, rt)]，loads 为 [(rec, pk, dl)]。
    """
    facs = sorted({r.d for r in recs if r.kind == "O"}
                  | {r.o for r in recs if r.kind != "O"}
                  | {r.d for r in recs if r.kind == "S"})
    seqs = set()
    for a in AIRPORTS:
        for f in facs:
            seqs.add((a, (f,)))
        # 双设施最近邻
        for f in facs:
            for g in _nearest_facilities(f, k_pair, distances):
                seqs.add((a, (f, g)))
                seqs.add((a, (g, f)))  # 反向顺序也加入（拼座方向依赖很关键）
        # 三设施最近邻（k_triple>0 时启用，MAX_LANDINGS<=5 时三设施尚留余量）
        if k_triple > 0:
            for f in facs:
                near = _nearest_facilities(f, k_triple, distances)
                for i in range(len(near)):
                    g = near[i]
                    seqs.add((a, (f, g)))  # 双设施
                    # 三设施：(f, g, h) 全排列（6种）过于爆炸，仅按距离升序两种方向
                    for h in near[i + 1:]:
                        seqs.add((a, (f, g, h)))
                        seqs.add((a, (f, h, g)))
                        seqs.add((a, (g, f, h)))
        for r in recs:
            if r.kind == "S":
                seqs.add((a, (r.o, r.d)))

    order = sorted(recs, key=lambda r: (r.prio, r.ep, r.pid))
    cols = {}
    for a, seq in sorted(seqs):
        for t in TYPES:
            route, _ = build_route(a, t, seq, distances)
            if route is None:
                continue
            off = profile(route, 0, distances)[0]
            wins = []
            for r in order:
                w = dep_window(r, route, day, off, distances)
                if w:
                    wins.append((w[0], w[1], r))
            if not wins:
                continue
            deps = sorted({T0} | {lo for lo, _hi, _r in wins})
            if len(deps) > max_deps:
                step = (len(deps) - 1) / (max_deps - 1)
                deps = sorted({deps[round(i * step)] for i in range(max_deps)})
            rt = route_time(route, distances)
            for dep in deps:
                arr, _ = profile(route, dep, distances)
                elig = [(lo, hi, r) for lo, hi, r in wins if lo <= dep <= hi]
                for order_key in (
                        lambda w: (w[1], w[2].prio, w[2].ep, w[2].pid),
                        lambda w: (-w[2].ep, w[1], w[2].prio, w[2].pid)):
                    elig.sort(key=order_key)
                    loads = []
                    mand = [w for w in elig if not w[2].temp]
                    temp = [w for w in elig if w[2].temp]
                    for _lo, _hi, r in mand + temp:
                        pos = rec_feasible(r, route, arr, day)
                        if pos and _capacity_ok(route, [(l[1], l[2]) for l in loads],
                                                add=pos):
                            loads.append((r, pos[0], pos[1]))
                    if not loads:
                        continue
                    key = (tuple(l[0].pid for l in loads), a, t)
                    cur = (route, dep, loads, rt)
                    if key not in cols or (rt, dep) < (cols[key][3], cols[key][1]):
                        cols[key] = cur
    out = sorted(cols.values(), key=lambda c: (c[3], c[0].airport, c[0].ac_type, c[1]))
    return _ensure_coverage(distances, day, recs, out)


def _ensure_coverage(distances, day, recs, cols):
    """覆盖兜底：任一非临时人员若不在任何列中，生成专属直达列族保证可覆盖"""
    covered = {l[0].pid for c in cols for l in c[2]}
    missing = [r for r in recs if not r.temp and r.pid not in covered]
    for r in sorted(missing, key=lambda r: r.pid):
        seq = ((r.d,) if r.kind == "O" else (r.o,) if r.kind == "B" else (r.o, r.d))
        variants = []
        for a in AIRPORTS:
            for t in TYPES:
                route, _ = build_route(a, t, seq, distances)
                if route is None:
                    continue
                off = profile(route, 0, distances)[0]
                w = dep_window(r, route, day, off, distances)
                if w is None:
                    continue
                variants.append((route, w[0], off))
        if not variants:
            raise RuntimeError(
                f"人员 {r.pid} 在 {day} 日无可行直达架次（窗口过紧）: "
                f"{r.kind} {r.o}->{r.d} ep={r.ep} la={r.la}")
        for route, dep, off in variants:
            arr, _ = profile(route, dep, distances)
            pos0 = rec_feasible(r, route, arr, day)
            assert pos0 is not None
            loads = [(r, pos0[0], pos0[1])]
            wins = []
            for r2 in recs:
                if r2.pid == r.pid:
                    continue
                w2 = dep_window(r2, route, day, off, distances)
                if w2 and w2[0] <= dep <= w2[1]:
                    wins.append((w2[1], r2))
            for _h, r2 in sorted(wins, key=lambda w: (w[0], w[1].prio, w[1].ep, w[1].pid)):
                pos = rec_feasible(r2, route, arr, day)
                if pos and _capacity_ok(route, [(l[1], l[2]) for l in loads], add=pos):
                    loads.append((r2, pos[0], pos[1]))
            cols.append((route, dep, loads, route_time(route, distances)))
    return cols


def servable_on_day(distances, rec, day):
    """人员 rec 在 day 日是否存在任何可行直达架次（规则 A 按日预筛）"""
    seq = ((rec.d,) if rec.kind == "O" else (rec.o,) if rec.kind == "B"
           else (rec.o, rec.d))
    for a in AIRPORTS:
        for t in TYPES:
            route, _ = build_route(a, t, seq, distances)
            if route is None:
                continue
            off = profile(route, 0, distances)[0]
            if dep_window(rec, route, day, off, distances):
                return True
    return False


def prune_columns(cols, cap_row=40, cap_grp=10):
    """列池剪枝：每需求行保留全局最省 cap_row 列 + 每 (行,日,机场,机型) 最省
    cap_grp 列，保证每行覆盖数不变坏且保留机队多样性。"""
    keep = set()
    by_row = defaultdict(list)
    for j, c in enumerate(cols):
        for l in c[3]:
            by_row[l[0].pid].append(j)
    for _pid, js in by_row.items():
        # 列结构: (day, route, dep, loads, rt)
        key = lambda j: (cols[j][4], cols[j][0], cols[j][2], cols[j][1].airport)
        keep.update(sorted(js, key=key)[:cap_row])
        byg = defaultdict(list)
        for j in js:
            byg[(cols[j][0], cols[j][1].airport, cols[j][1].ac_type)].append(j)
        for gjs in byg.values():
            keep.update(sorted(gjs, key=key)[:cap_grp])
    out = [c for j, c in enumerate(cols) if j in keep]
    print(f"[prune] {len(cols)} -> {len(out)} 列", flush=True)
    return out


def build_columns(distances, recs, k_pair=3, max_deps=12, prune=(40, 10), k_triple=0):
    """规则 A 全周列池：每天对当日可行需求生成候选列，再全局剪枝。

    返回 [(day, route, dep, loads, rt)]。
    """
    all_cols = []
    for day in range(N_DAYS):
        day_recs = [r for r in recs if servable_on_day(distances, r, day)]
        t0 = time.time()
        cols = enum_day_columns(distances, day, day_recs,
                                k_pair=k_pair, max_deps=max_deps, k_triple=k_triple)
        all_cols += [(day, *c) for c in cols]
        print(f"[m2-A] day{day} 列 {len(cols)}（{time.time() - t0:.1f}s）", flush=True)
    cov = {l[0].pid for c in all_cols for l in c[3]}
    miss = [r.pid for r in recs if not r.temp and r.pid not in cov]
    if miss:
        raise RuntimeError(f"规则A 全周仍不可服务 {len(miss)} 人，例 {miss[:5]}")
    if prune:
        all_cols = prune_columns(all_cols, cap_row=prune[0], cap_grp=prune[1])
    return all_cols


def concurrency_groups(cols_with_day):
    """按 (day, 机场, 机型) 分组列下标，用于机队并发约束与飞机指派"""
    groups = defaultdict(list)
    for j, (day, route, dep, _loads, _rt) in enumerate(cols_with_day):
        groups[(day, route.airport, route.ac_type)].append(j)
    return groups


# ==================== COPT MILP ====================
def _copt_model(name, log_file):
    """创建 COPT 模型；日志先写系统临时目录（避免中文路径）再复制"""
    env = cp.Envr()
    m = env.createModel(name)
    tmp = None
    if log_file:
        fd, tmp = tempfile.mkstemp(prefix="copt_q3_", suffix=".log")
        os.close(fd)
        m.setLogFile(tmp)
    return m, tmp


def _finish_log(tmp, log_file):
    """把 COPT 临时日志复制到目标位置并清理"""
    if tmp:
        if log_file and os.path.exists(tmp):
            try:
                shutil.copy(tmp, log_file)
            except OSError:
                pass
        try:
            os.unlink(tmp)
        except OSError:
            pass


def solve_milp(distances, recs, cols, phase, t_star=None, log_file=None, time_limit=1800,
               warm_cols=None, rel_gap=None):
    """阶段一/阶段二 MILP 求解。

    phase="one": min 总时间，s.t. 非临时全覆盖 + 机队并发。
    phase="two": max 临时覆盖，s.t. + 总时间 <= t_star + 临时每人 <= 1。
    warm_cols: 保底可行 incumbent 的列下标列表。
    rel_gap:   COPT 相对间隙阈值（None 时用默认）；
               渐进式收紧流程可通过传参覆盖 Param.RelGap。
    返回 (obj, xs, info)。
    """
    mand = [r for r in recs if not r.temp]
    temps = [r for r in recs if r.temp]
    if phase == "two" and not temps:
        # 当日无临时任务可服务：直接返回空解，由调用方回退到阶段一架次
        return None, np.zeros(n_c := len(cols)), {
            "solve_s": 0.0, "status": "no_temps", "n_cols": len(cols),
            "n_concurrency_rows": 0, "obj": None}
    midx = {r.pid: i for i, r in enumerate(mand)}
    tidx = {r.pid: i for i, r in enumerate(temps)}
    n_c = len(cols)
    cost = np.array([c[4] for c in cols], dtype=np.float64)

    rows_m, cols_m, data_m = [], [], []
    rows_t, cols_t, data_t = [], [], []
    for j, (_d, _r, _dep, loads, _rt) in enumerate(cols):
        for rec, _p, _l in loads:
            if rec.temp:
                rows_t.append(tidx[rec.pid])
                cols_t.append(j)
                data_t.append(1.0)
            else:
                rows_m.append(midx[rec.pid])
                cols_m.append(j)
                data_m.append(1.0)
    A_m = sp.csr_matrix((data_m, (rows_m, cols_m)), shape=(len(mand), n_c))
    A_t = sp.csr_matrix((data_t, (rows_t, cols_t)), shape=(len(temps), n_c))

    # 机队并发约束：任一事件时刻活动架次数 <= 在场机数（区间图 => 充分）
    rets = [profile(c[1], c[2], distances)[1] for c in cols]
    groups = concurrency_groups([(c[0], c[1], c[2], c[3], c[4]) for c in cols])
    crow, ccol, cdat, crhs = [], [], [], []
    n_con = 0
    for (day, a, t), js in sorted(groups.items()):
        cap = N_AT[(a, t)]
        evs = sorted({cols[j][2] for j in js})
        starts = sorted((cols[j][2], j) for j in js)
        ends = sorted((rets[j] + TURN, j) for j in js)
        si = ei = 0
        active = set()
        for tau in evs:
            while si < len(starts) and starts[si][0] <= tau:
                active.add(starts[si][1])
                si += 1
            while ei < len(ends) and ends[ei][0] <= tau:
                active.discard(ends[ei][1])
                ei += 1
            if len(active) > cap:
                for j in active:
                    crow.append(n_con)
                    ccol.append(j)
                    cdat.append(1.0)
                crhs.append(float(cap))
                n_con += 1
    A_c = sp.csr_matrix((cdat, (crow, ccol)), shape=(n_con, n_c))

    m, tmp = _copt_model(f"q3_m2_{phase}", log_file)
    x = m.addMVar(n_c, lb=0.0, vtype=COPT.INTEGER)
    if len(mand) > 0:
        m.addConstr(A_m @ x >= np.ones(len(mand)))
    if n_con:
        m.addConstr(A_c @ x <= np.array(crhs))
    if phase == "one":
        m.setObjective(cost @ x, COPT.MINIMIZE)
    else:
        if len(temps) > 0:
            z = m.addMVar(len(temps), lb=0.0, ub=1.0, vtype=COPT.CONTINUOUS)
            m.addConstr(A_t @ x - z >= np.zeros(len(temps)))
            m.addConstr(cost @ x <= float(t_star) + 1e-6)
            m.setObjective(np.ones(len(temps)) @ z, COPT.MAXIMIZE)
        else:
            m.setObjective(np.zeros(n_c) @ x, COPT.MINIMIZE)
    m.setParam(COPT.Param.TimeLimit, time_limit)
    if rel_gap is not None:
        m.setParam(COPT.Param.RelGap, float(rel_gap))
    if warm_cols:
        for j in warm_cols:
            try:
                m.setMipStart(x[j], 1.0)
            except Exception:
                pass
    t0 = time.time()
    m.solve()
    el = time.time() - t0
    _finish_log(tmp, log_file)
    try:
        obj = m.objval
    except Exception:
        obj = None
    try:
        best_bound = m.bestbound
    except Exception:
        best_bound = None
    try:
        xs = np.array([float(v) for v in x.x])
        if len(xs) != n_c:
            xs = np.zeros(n_c)
    except Exception:
        xs = np.zeros(n_c)
    info = {"solve_s": round(el, 3), "status": str(m.status),
            "n_cols": n_c, "n_concurrency_rows": n_con, "obj": obj,
            "best_bound": best_bound}
    return obj, xs, info


def solve_phase1_adaptive(distances, recs, cols, logs_dir, time_limit_per_round=3600,
                          gaps=(0.05, 0.02, 0.01), early_stopping=True):
    """渐进式 RelGap 收紧：多轮 phase="one"，上一轮解向量作为下一轮 warm-start。

    gaps: 每轮间隙阈值（默认 0.05→0.02→0.01，项目记忆要求最终 ≤0.01）。
    early_stopping: 相邻两轮 obj 未改进则提前终止（不浪费算力）。
    返回 (obj, xs, rounds_info)。
    """
    xs_prev = None
    obj_prev = None
    rounds_info = []
    for idx, gap in enumerate(gaps):
        warm_cols = (list(np.where(xs_prev > 0.5)[0].tolist())
                     if xs_prev is not None else None)
        log_tag = f"m2A_phase1_r{idx + 1}_gap{int(gap * 10000)}ppm"
        print(f"\n  [phase1-adaptive] Round {idx + 1}/{len(gaps)}: RelGap={gap}, "
              f"TL={time_limit_per_round}s, warm={warm_cols is not None}", flush=True)
        obj, xs, info = solve_milp(
            distances, recs, cols, "one",
            log_file=os.path.join(logs_dir, f"{log_tag}.log"),
            time_limit=time_limit_per_round,
            warm_cols=warm_cols,
            rel_gap=gap)
        info["gap_target"] = gap
        rounds_info.append(info)
        print(f"    obj={obj}, best_bound={info.get('best_bound')}, "
              f"status={info['status']}, {info['solve_s']}s", flush=True)
        # 早停判定：T* 未改进且上一轮非超时（说明已收敛到紧于 gap 的精度）
        if (early_stopping and obj_prev is not None and obj is not None
                and obj >= obj_prev - 0.5):
            print(f"    早停触发：obj {obj_prev:.2f} -> {obj:.2f} 未改善，跳过后续更紧轮次",
                  flush=True)
            return obj, xs, rounds_info
        xs_prev = xs
        obj_prev = obj
    return obj, xs, rounds_info


def _dedup_loads(sel):
    """选中列物化：同一人员若被多列覆盖，只上首列（卸载不破坏容量/窗口）"""
    seen = set()
    out = []
    for day, route, dep, loads, rt in sel:
        nl = []
        for rec, pk, dl in loads:
            if rec.pid in seen:
                continue
            seen.add(rec.pid)
            nl.append((rec, pk, dl))
        out.append((day, route, dep, nl, rt))
    return out


def solve_phase2_by_day(distances, recs, cols, trips, logs_dir, time_limit=300):
    """规则A阶段二：逐日 prize-collecting。

    日预算 = 该日阶段一（抛光后）架次总时间；阶段一实际架次作为附加列保证可行。
    临时人员按日序先到先得（跨日去重）。
    返回 (final_trips, info2)。
    """
    served_temp = set()
    final_trips = []
    sub = []
    # 阶段一语义不含临时任务：剥掉列生成时顺带装载的临时人员，
    # 避免阶段二重复计数（临时任务只由阶段二统一决定是否承运）
    for t in trips:
        t["loads"] = [l for l in t["loads"] if not l[0].temp]
    for day in range(N_DAYS):
        day_trips = [t for t in trips if t["day"] == day]
        day_mand = {l[0].pid for t in day_trips for l in t["loads"]}
        allow = set(day_mand) | {r.pid for r in recs if r.temp
                                 and r.pid not in served_temp
                                 and servable_on_day(distances, r, day)}
        day_recs = [r for r in recs if r.pid in allow]
        budget = sum(route_time(t["route"], distances) for t in day_trips)
        day_cols = []
        for c in cols:
            if c[0] != day:
                continue
            nl = [l for l in c[3] if l[0].pid in allow]
            if nl:
                day_cols.append((c[0], c[1], c[2], nl, c[4]))
        extra = [(t["day"], t["route"], t["dep"], t["loads"],
                  route_time(t["route"], distances)) for t in day_trips]
        cols_d = day_cols + extra
        _obj, xs2, info = solve_milp(
            distances, day_recs, cols_d, "two", t_star=budget,
            log_file=os.path.join(logs_dir, f"m2A_phase2_d{day}.log"),
            time_limit=time_limit,
            warm_cols=list(range(len(day_cols), len(cols_d))))
        sel = _dedup_loads([cols_d[j] for j, v in enumerate(xs2) if v > 0.5])
        if not sel:
            sel = _dedup_loads(extra)
            info = {**info, "fallback": True}
        for c in sel:
            final_trips.append(c)
            for l in c[3]:
                if l[0].temp:
                    served_temp.add(l[0].pid)
        sub.append({"day": day, "budget": budget, "status": info["status"],
                    "solve_s": info["solve_s"]})
    info2 = {"by_day": sub, "solve_s": sum(s["solve_s"] for s in sub),
             "status": "by_day_mixed"}
    return final_trips, info2


def assign(distances, selected):
    """把选中 (day, route, dep, loads, rt) 指派到具体飞机。

    并发约束已保证可行：同 (day,机场,机型) 按 dep 排序，分给最早可用飞机。
    返回 (trips, fails)。
    """
    av = {}
    trips = []
    fails = 0
    for day, route, dep, loads, _rt in sorted(selected, key=lambda c: (c[0], c[2])):
        cand = [x for x in AIRCRAFT_LIST
                if HOME[x] == route.airport and AC_TYPE[x] == route.ac_type
                and av.get((day, x), T0) <= dep]
        if not cand:
            fails += 1
            continue
        acid = min(cand, key=lambda x: (av.get((day, x), T0), x))
        arr, ret = profile(route, dep, distances)
        trips.append({"day": day, "acid": acid, "route": route, "dep": dep,
                      "arr": arr, "ret": ret, "loads": loads})
        av[(day, acid)] = ret + TURN
    return trips, fails


# ==================== 输出与指标 ====================
def _fmt(abs_min):
    """绝对分钟 -> 'YYYY-MM-DD HH:MM' 字符串"""
    return (DAY0 + timedelta(minutes=int(abs_min))).strftime(
        "%Y-%m-%d %H:%M")


def write_outputs(trips, all_pids, out_dir):
    """写 q3-routes.csv、q3-assignments.csv（题目模板格式）"""
    os.makedirs(out_dir, exist_ok=True)
    routes_path = os.path.join(out_dir, "q3-routes.csv")
    assign_path = os.path.join(out_dir, "q3-assignments.csv")

    # 备份旧文件（可恢复）
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak_dir = os.path.join(out_dir, "backup_" + ts)
    os.makedirs(bak_dir, exist_ok=True)
    for old in (routes_path, assign_path):
        if os.path.exists(old):
            shutil.copy(old, os.path.join(bak_dir, os.path.basename(old)))

    by_acid = defaultdict(list)
    for t in trips:
        by_acid[t["acid"]].append(t)
    fno = {}
    for acid, ts_list in by_acid.items():
        for i, t in enumerate(sorted(ts_list, key=lambda z: (z["day"], z["dep"])), 1):
            fno[id(t)] = (acid, i)

    with open(routes_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["aircraft_id", "flight_no", "stop_order", "facility_id",
                    "arrival_time", "departure_time", "refuel"])
        for t in sorted(trips, key=lambda z: (z["day"], z["dep"], z["acid"])):
            acid, no = fno[id(t)]
            r = t["route"]
            base = t["day"] * 1440
            n = len(r.stops)
            w.writerow([acid, no, 0, r.airport, "", _fmt(base + t["arr"][0]), 0])
            for k in range(1, n + 1):
                stop = REFUEL_STOP_MIN if r.refuel[k - 1] else STOP_MIN
                w.writerow([acid, no, k, r.stops[k - 1],
                            _fmt(base + t["arr"][k]),
                            _fmt(base + t["arr"][k] + stop),
                            1 if r.refuel[k - 1] else 0])
            w.writerow([acid, no, n + 1, r.airport, _fmt(base + t["arr"][n + 1]), "", 0])

    amap = {}
    for t in trips:
        acid, no = fno[id(t)]
        for rec, pk, dl in t["loads"]:
            amap[rec.pid] = (acid, no, pk, dl)
    with open(assign_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["person_id", "aircraft_id", "flight_no",
                    "pickup_stop_order", "delivery_stop_order"])
        for pid in all_pids:
            if pid in amap:
                acid, no, pk, dl = amap[pid]
                w.writerow([pid, acid, no, pk, dl])
            else:
                w.writerow([pid, "", "", "", ""])
    return routes_path, assign_path


def compute_metrics(distances, trips):
    """从最终 trips 实算五项指标 + 临时承运数（不依赖模型值）"""
    total_time = sum(route_time(t["route"], distances) for t in trips)
    total_fuel10 = sum(route_fuel10(t["route"], distances) for t in trips)
    pax_km = 0.0
    seat_km = 0.0
    total_transit = 0
    n_temp = 0
    n_mand = 0
    seen_pax = set()
    for t in trips:
        r = t["route"]
        seq = r.seq()
        obs = _onboard_per_leg(r, [(l[1], l[2]) for l in t["loads"]])
        for k in range(1, len(seq)):
            d = distances[seq[k - 1]][seq[k]]
            pax_km += obs[k - 1] * d
            seat_km += SEATS[r.ac_type] * d
        base = t["day"] * 1440
        for rec, pk, dl in t["loads"]:
            if rec.pid in seen_pax:
                continue
            seen_pax.add(rec.pid)
            dep_pk = base + (t["arr"][0] if pk == 0 else t["arr"][pk] + _stop_min_of(r, pk))
            total_transit += base + t["arr"][dl] - dep_pk
            if rec.temp:
                n_temp += 1
            else:
                n_mand += 1
    return {
        "total_time_min": int(total_time),
        "total_flights": len(trips),
        "total_fuel_kg": round(total_fuel10 / 10.0, 1),
        "total_transit_min": int(total_transit),
        "seat_utilization": round(pax_km / seat_km, 6) if seat_km else 0.0,
        "temp_served": n_temp,
        "mandatory_served": n_mand,
    }


def write_metrics(metrics, out_dir):
    """写 q3-metrics.txt（与旧版同目录，供论文与人工核对）"""
    path = os.path.join(out_dir, "q3-metrics.txt")
    lines = [
        f"总飞机使用时间 T*: {metrics['total_time_min']} 分钟 ({metrics['total_time_min']/60:.2f}h)",
        f"总架次数: {metrics['total_flights']}",
        f"总燃油消耗量: {metrics['total_fuel_kg']} kg",
        f"人员在途时间: {metrics['total_transit_min']} 分钟",
        f"座位利用率: {metrics['seat_utilization']:.4f}",
        f"非临时任务承运: {metrics['mandatory_served']}/3840",
        f"临时任务满足: {metrics['temp_served']}/160",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def recompute_metrics_from_csv(out_dir):
    """从 q3-routes.csv 与 q3-assignments.csv 重算指标（CSV 口径）。

    用途：metrics 与最终交付文件保持一致，避免内存内 trips 与 CSV 的计数偏差。
    """
    routes_path = os.path.join(out_dir, "q3-routes.csv")
    assign_path = os.path.join(out_dir, "q3-assignments.csv")
    distances, recs = load_data()
    rec_by_pid = {r.pid: r for r in recs}

    # 重建航班：aircraft_id -> [(stop_order, fac, arr, dep, refuel)]
    flights = defaultdict(list)
    with open(routes_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["aircraft_id"].strip(), int(row["flight_no"].strip()))
            arr = datetime.strptime(row["arrival_time"].strip(), "%Y-%m-%d %H:%M") \
                if row["arrival_time"].strip() else None
            dep = datetime.strptime(row["departure_time"].strip(), "%Y-%m-%d %H:%M") \
                if row["departure_time"].strip() else None
            flights[key].append((int(row["stop_order"].strip()),
                                 row["facility_id"].strip(), arr, dep,
                                 int(row["refuel"].strip())))
    for key in flights:
        flights[key].sort(key=lambda s: s[0])

    # 重建 trips：每航班 -> (day, acid, route, dep, arr, loads)
    trips_by_key = {}
    for (acid, no), stops in sorted(flights.items()):
        facs = [s[1] for s in stops]
        dep0 = stops[0][3]
        day = (dep0 - DAY0).days
        arr_abs = [dep0] + [s[2] for s in stops[1:]]
        arr = [int((a - DAY0 - timedelta(days=day)).total_seconds() // 60) for a in arr_abs]
        ac_type = acid.split("-")[1]
        route = Q3Route(facs[0], ac_type, tuple(facs[1:-1]),
                        tuple(bool(s[4]) for s in stops[1:-1]))
        trips_by_key[(acid, no)] = {
            "day": day, "acid": acid, "route": route,
            "dep": int((dep0 - DAY0 - timedelta(days=day)).total_seconds() // 60),
            "arr": arr, "ret": arr[-1], "loads": []}

    # 装载人员
    with open(assign_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["aircraft_id"].strip():
                continue
            key = (row["aircraft_id"].strip(), int(row["flight_no"].strip()))
            rec = rec_by_pid[row["person_id"].strip()]
            trip = trips_by_key.get(key)
            if trip is None:
                continue
            trip["loads"].append((rec, int(row["pickup_stop_order"].strip()),
                                  int(row["delivery_stop_order"].strip())))
    return compute_metrics(distances, list(trips_by_key.values()))


# ==================== 主流程 ====================
def main():
    """服务日入列 + 事件驱动时刻 + 覆盖兜底 + 逐日阶段二的完整求解入口"""
    parser = argparse.ArgumentParser(description="问题3 服务日入列求解器")
    parser.add_argument("--smoke", action="store_true", help="冒烟模式（小列池+短时限）")
    parser.add_argument("--time-limit", type=int, default=7200,
                        help="阶段一每轮 COPT 时限（秒），默认 7200（项目记忆 >= 2h）")
    parser.add_argument("--phase2-limit", type=int, default=7200,
                        help="阶段二每日 COPT 时限（秒），默认 7200")
    parser.add_argument("--prune", type=int, nargs=2, default=(40, 10),
                        help="列池剪枝 (cap_row, cap_grp)，默认 40 10")
    parser.add_argument("--max-deps", type=int, default=12,
                        help="每日候选起飞时刻上限，默认 12")
    parser.add_argument("--k-pair", type=int, default=8,
                        help="双设施最近邻数（默认 8，实验2=扩大列池）")
    parser.add_argument("--k-triple", type=int, default=4,
                        help="三设施最近邻数（默认 4，>0 启用三设施停靠；0 关闭）")
    parser.add_argument("--gaps", type=float, nargs="+", default=[0.05, 0.02, 0.01],
                        help="渐进式 RelGap 序列（默认 0.05 0.02 0.01，项目记忆 <= 0.01）")
    parser.add_argument("--no-adaptive", action="store_true",
                        help="禁用渐进式收紧，改为单次 solve_milp（等价老行为）")
    parser.add_argument("--recompute-metrics", action="store_true",
                        help="仅从 q3-routes.csv/q3-assignments.csv 重算指标")
    args = parser.parse_args()

    if args.recompute_metrics:
        print("从 CSV 重算 Q3 指标...")
        metrics = recompute_metrics_from_csv(OUTPUT_DIR_Q3)
        print(metrics)
        path = write_metrics(metrics, OUTPUT_DIR_Q3)
        print(f"已写入 {path}")
        return

    smoke = args.smoke
    time_limit = 120 if smoke else args.time_limit
    phase2_limit = 30 if smoke else args.phase2_limit
    prune = (8, 2) if smoke else tuple(args.prune)
    max_deps = 6 if smoke else args.max_deps
    k_pair = 1 if smoke else args.k_pair
    k_triple = 0 if smoke else args.k_triple
    gaps = tuple([0.1, 0.05]) if smoke else tuple(args.gaps)

    logs_dir = os.path.join(OUTPUT_DIR_Q3, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    print("=" * 60)
    print("问题3（方案一）：服务日入列 + 事件驱动时刻 + 覆盖兜底")
    print(f"模式: {'SMOKE' if smoke else 'FULL'}")
    print(f"列池: k_pair={k_pair}, k_triple={k_triple}, max_deps={max_deps}, prune={prune}")
    print(f"阶段一: {'渐进式 RelGap ' + str(gaps) if not args.no_adaptive else f'单次 TL={time_limit}s'}")
    print(f"阶段二: 逐日 prize-collecting, TL={phase2_limit}s/日")
    print("=" * 60)

    t_all = time.time()
    print("\n[1] 加载数据...")
    distances, recs = load_data()
    n_mand = sum(1 for r in recs if not r.temp)
    n_temp = sum(1 for r in recs if r.temp)
    print(f"  总需求 {len(recs)}（非临时 {n_mand}，临时 {n_temp}）")

    print("\n[2] 构建全周候选列池（规则A）...")
    cols = build_columns(distances, recs, k_pair=k_pair, max_deps=max_deps,
                         prune=prune, k_triple=k_triple)
    print(f"  全周列池: {len(cols)}")

    print("\n[3] 阶段一 MILP（min 总时间，3840 非临时全覆盖）...")
    info1_rounds = []
    if args.no_adaptive:
        obj1, xs1, info1 = solve_milp(
            distances, recs, cols, "one",
            log_file=os.path.join(logs_dir, "m2A_phase1.log"),
            time_limit=time_limit)
        info1_rounds = [info1]
    else:
        # 项目记忆约束：每轮 time_limit >= 7200s；gaps 0.05→0.02→0.01；邻轮 T* 不改善早停
        obj1, xs1, info1_rounds = solve_phase1_adaptive(
            distances, recs, cols, logs_dir,
            time_limit_per_round=time_limit,
            gaps=gaps, early_stopping=True)
        info1 = info1_rounds[-1]
    print(f"  阶段一(最终轮): obj={obj1}, best_bound={info1.get('best_bound')}, "
          f"状态={info1['status']}, 耗时={info1['solve_s']}s, "
          f"并发行={info1['n_concurrency_rows']}")
    sel1 = _dedup_loads([cols[j] for j, v in enumerate(xs1) if v > 0.5])
    if not sel1:
        raise RuntimeError(f"阶段一无可行解（状态 {info1['status']}）")
    trips1, fails1 = assign(distances, sel1)
    print(f"  阶段一选列 {len(sel1)}，飞机指派失败 {fails1}")
    if fails1:
        raise RuntimeError("阶段一飞机指派失败，并发约束与指派逻辑不一致")

    print("\n[4] 阶段二逐日 prize-collecting（max 临时覆盖，<= 当日预算）...")
    trips2, info2 = solve_phase2_by_day(
        distances, recs, cols, trips1, os.path.join(logs_dir),
        time_limit=phase2_limit)
    trips2, fails2 = assign(distances, trips2)
    print(f"  阶段二各日: {[(s['day'], s['status'], s['solve_s']) for s in info2['by_day']]}")
    print(f"  阶段二飞机指派失败 {fails2}")
    if fails2:
        raise RuntimeError("阶段二飞机指派失败")

    print("\n[5] 指标与输出...")
    metrics = compute_metrics(distances, trips2)
    print(metrics)
    routes_path, assign_path = write_outputs(trips2, [r.pid for r in recs], OUTPUT_DIR_Q3)
    metrics_path = write_metrics(metrics, OUTPUT_DIR_Q3)
    print(f"  routes: {routes_path}")
    print(f"  assign: {assign_path}")
    print(f"  metrics: {metrics_path}")
    print(f"\n总耗时 {time.time() - t_all:.1f}s")

    print("\n==================== 实验对比汇总 ====================")
    print("基准 (k_pair=3, 无三设施, 单次 1800s): T*=28046, 临时=93/160")
    print(f"本次 (k_pair={k_pair}, k_triple={k_triple}, "
          f"{'渐进 ' + str(gaps) if not args.no_adaptive else '单次'}):")
    print(f"  T* = {metrics['total_time_min']} min (基准差 {metrics['total_time_min'] - 28046:+d}),"
          f" 架次 = {metrics['total_flights']}")
    print(f"  非临时承运 = {metrics['mandatory_served']}/3840, "
          f"临时满足 = {metrics['temp_served']}/160 "
          f"(基准差 {metrics['temp_served'] - 93:+d})")
    if not args.no_adaptive and info1_rounds:
        print(f"  阶段一各轮 obj 轨迹: "
              + " → ".join(f"r{i+1}={r.get('obj')}({r.get('gap_target')})"
                            for i, r in enumerate(info1_rounds)))


if __name__ == "__main__":
    main()
