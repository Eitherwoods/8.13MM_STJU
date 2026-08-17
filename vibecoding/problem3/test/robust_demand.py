# -*- coding: utf-8 -*-
"""
文件: robust_demand.py（2026-08-17 适配版）
功能: 鲁棒优化 — 需求人数波动下的单日路径排班（2026-08-03）
项目: 2026策联杯 B题 - 海上油田人员直升机运载计划编排

适配说明:
  旧版依赖已删除的 problem3_solution（贪心按天分配 + 单日两层 MIP）。
  新版复用 problem3_service_day 的服务日入列列生成与阶段一 MILP：
    - 单日需求集 = 全周名义解中服务日恰为 08-03 的非临时需求
      （由 q3-assignments.csv / q3-routes.csv 提取，与主结果口径一致）；
    - 名义解 S_0 与鲁棒解 S_10%/S_19% 分别按放大后需求求解单日；
    - 覆盖评估 = 固定选中架次后，把放大需求重新装载，
      统计可承运人数 / 放大后需求总数。

运行方式:
    python robust_demand.py
    $env:ROBUST_DEMAND_TL=180; python robust_demand.py   # 自定义单日 MIP 时限
"""

from __future__ import annotations

import csv
import os
import random
import sys
from datetime import datetime

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROBLEM3_DIR = os.path.dirname(_THIS_DIR)
_VIBECODING_DIR = os.path.dirname(_PROBLEM3_DIR)
_PROJECT_DIR = os.path.dirname(_VIBECODING_DIR)
for _p in (_VIBECODING_DIR, _PROBLEM3_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from problem3_service_day import (  # noqa: E402
    AIRPORTS, OUTPUT_DIR_Q3, Q3Rec, _capacity_ok, _dedup_loads, assign,
    build_columns, dep_window, enum_day_columns, load_data, profile,
    prune_columns, rec_feasible, route_time, solve_milp,
)

SCALES = [1.0, 1.1, 1.2]
EVAL_SCALES = [1.0, 1.1, 1.2]
MIP_TIME_LIMIT = int(os.environ.get('ROBUST_DEMAND_TL', '120'))
SEED = 20260814
OUTPUT_DIR = os.path.join(_PROJECT_DIR, 'output', 'Problem3')
TASK_BY_PRIO = {0: 'emergency', 1: 'production', 2: 'shift'}


def extract_day0_pids():
    """从全周名义解提取服务日 08-03 的非临时人员编号。

    通过 q3-routes.csv 中 08-03 的 (aircraft_id, flight_no) 与
    q3-assignments.csv 的航班匹配得到，保证与主结果同一口径。
    """
    routes_path = os.path.join(OUTPUT_DIR_Q3, 'q3-routes.csv')
    assign_path = os.path.join(OUTPUT_DIR_Q3, 'q3-assignments.csv')
    day0_keys = set()
    with open(routes_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            dep = row.get('departure_time', '').strip()
            if dep.startswith('2026-08-03') and row['stop_order'].strip() == '0':
                day0_keys.add((row['aircraft_id'].strip(),
                               int(row['flight_no'].strip())))
    pids = []
    with open(assign_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if not row['aircraft_id'].strip():
                continue
            key = (row['aircraft_id'].strip(), int(row['flight_no'].strip()))
            if key in day0_keys:
                pids.append(row['person_id'].strip())
    return set(pids)


def scale_recs(recs, scale, seed):
    """复制部分需求以模拟设施级人数按比例增加（保留原需求 + 随机复制）"""
    if scale <= 1.0:
        return list(recs)
    rng = random.Random(seed)
    scaled = list(recs)
    extra = int(len(recs) * (scale - 1))
    for k in range(extra):
        src = rng.choice(recs)
        task = TASK_BY_PRIO.get(src.prio, 'shift')
        scaled.append(Q3Rec(f"{src.pid}_dup{k}", src.kind, src.o, src.d,
                            src.ep, src.la, task))
    return scaled


def solve_day0(distances, day_recs, time_limit):
    """用服务日入列单日列生成 + 阶段一 MILP 求解 08-03 单日问题。

    返回 (trips, total_time) 或 None。
    """
    cols_raw = enum_day_columns(distances, 0, day_recs, k_pair=3, max_deps=12)
    cols = [(0, *c) for c in cols_raw]
    cols = prune_columns(cols, cap_row=40, cap_grp=10)
    print(f"  单日列池(剪枝后): {len(cols)}", flush=True)
    _obj, xs, info = solve_milp(
        distances, day_recs, cols, "one",
        log_file=os.path.join(OUTPUT_DIR_Q3, 'logs', 'robust_demand_phase1.log'),
        time_limit=time_limit)
    print(f"  阶段一: status={info['status']}, obj={info['obj']}, "
          f"solve_s={info['solve_s']}", flush=True)
    sel = _dedup_loads([cols[j] for j, v in enumerate(xs) if v > 0.5])
    if not sel:
        return None
    trips, fails = assign(distances, sel)
    if fails:
        return None
    total_time = sum(route_time(t['route'], distances) for t in trips)
    return trips, total_time


def coverage_rate(distances, trips, eval_recs):
    """固定选中架次后，把放大需求重新装载，返回可承运覆盖率。

    对每个选中列，剩余航段容量内按优先级/最早登机顺序装入
    尚未承运的评估需求（与原装载不冲突），统计最终可承运人数。
    """
    # 已承运人员（原名义装载）
    served = {l[0].pid for t in trips for l in t['loads']}
    pending = [r for r in eval_recs if r.pid not in served]
    pending.sort(key=lambda r: (r.prio, r.ep, r.pid))
    for t in trips:
        t['loads'] = list(t['loads'])
    for rec in pending:
        for t in trips:
            route = t['route']
            arr = t['arr']
            off = profile(route, 0, distances)[0]
            w = dep_window(rec, route, 0, off, distances)
            if w is None or not (w[0] <= t['dep'] <= w[1]):
                continue
            pos = rec_feasible(rec, route, arr, 0)
            if pos and _capacity_ok(route, [(l[1], l[2]) for l in t['loads']],
                                    add=pos):
                t['loads'].append((rec, pos[0], pos[1]))
                served.add(rec.pid)
                break
    return len(served) / len(eval_recs) if eval_recs else 1.0


def main():
    """求解名义/鲁棒解、评估人数波动下的可行性、输出对比"""
    print('=' * 70)
    print('鲁棒优化实验（适配版）：需求人数波动下的 08-03 单日排班')
    print('=' * 70)

    distances, all_recs = load_data()
    day0_pids = extract_day0_pids()
    day_recs = [r for r in all_recs
                if not r.temp and r.pid in day0_pids]
    print(f'  08-03 服务日非临时需求: {len(day_recs)} 人')

    solutions = []
    for scale in SCALES:
        tag = '名义' if scale == 1.0 else f'鲁棒+{int((scale-1)*100)}%'
        print(f'\n--- {tag} ---')
        scaled = scale_recs(day_recs, scale, SEED)
        sol = solve_day0(distances, scaled, MIP_TIME_LIMIT)
        if sol is None:
            print(f'  {tag} 求解失败')
            continue
        trips, total_time = sol
        print(f'  {tag}: 时间={total_time}min, 架次={len(trips)}')
        solutions.append({'scale': scale, 'trips': trips,
                          'total_time': total_time, 'n_flights': len(trips)})

    if not solutions:
        print('错误: 所有方案求解失败')
        return

    print('\n评估人数波动下的覆盖可行性率...')
    feas_matrix = []
    for sol in solutions:
        row = []
        for s in EVAL_SCALES:
            eval_recs = scale_recs(day_recs, s, SEED)
            rate = coverage_rate(distances, sol['trips'], eval_recs)
            row.append(rate)
        feas_matrix.append(row)
        print(f'  S_{int((sol["scale"]-1)*100)}%: ' + '  '.join(
            f'+{int((s-1)*100)}%:{r*100:.1f}%' for s, r in zip(EVAL_SCALES, row)))

    lines = []
    lines.append('=' * 70)
    lines.append('鲁棒优化实验（适配版）：需求人数波动下的单日路径排班')
    lines.append('=' * 70)
    lines.append('一、各求解方案的成本')
    lines.append(f'{"方案":<14}{"总时间(min)":>14}{"架次":>8}')
    for sol in solutions:
        tag = '名义' if sol['scale'] == 1.0 else f'鲁棒+{int((sol["scale"]-1)*100)}%'
        lines.append(f'{tag:<14}{sol["total_time"]:>14}{sol["n_flights"]:>8}')
    lines.append('')
    lines.append('二、人数波动下的覆盖可行性率矩阵')
    header = '方案\\波动   ' + ''.join(f'+{int((s-1)*100)}%'.rjust(10)
                                      for s in EVAL_SCALES)
    lines.append(header)
    for i, sol in enumerate(solutions):
        name = '名义' if sol['scale'] == 1.0 else f'鲁棒+{int((sol["scale"]-1)*100)}%'
        row = f'{name:<10}' + ''.join(
            f'{feas_matrix[i][j]*100:>9.1f}%' for j in range(len(EVAL_SCALES)))
        lines.append(row)
    lines.append('=' * 70)
    content = '\n'.join(lines) + '\n'
    path = os.path.join(OUTPUT_DIR, 'robust_demand_results.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(content)
    print(f'结果已写入: {path}')


if __name__ == '__main__':
    main()
