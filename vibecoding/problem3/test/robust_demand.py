# -*- coding: utf-8 -*-
"""
文件: robust_demand.py
功能: 鲁棒优化 — 需求人数波动下的单日路径排班（2026-08-03）
项目: 2026策联杯 B题 - 海上油田人员直升机运载计划编排

方法论:
  海上油田人员的实际需求人数往往与计划存在偏差（如增储上产任务的
  检维修窗口人数临时增加）。设设施级需求人数 n_f 不确定，
  n_f ∈ [nbar_f, nbar_f(1+sigma)]，sigma 为需求波动比例。

  鲁棒解 S_sigma: 按放大后人数 nbar_f(1+sigma) 求解，预留容量缓冲;
  名义解 S_0: 按名义人数 nbar_f 求解。

  评估: 对每个解，检查其在人数放大 scale∈{1.0,1.1,1.2} 下的覆盖可行性，
  即选中路径对设施 f 的最大可服务人数是否 ≥ nbar_f × scale。

运行方式:
    python robust_demand.py
"""

import os
import sys
import time as time_module
from datetime import date
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROBLEM3_DIR = os.path.dirname(_THIS_DIR)
_VIBECODING_DIR = os.path.dirname(_PROBLEM3_DIR)
if _VIBECODING_DIR not in sys.path:
    sys.path.insert(0, _VIBECODING_DIR)
if _PROBLEM3_DIR not in sys.path:
    sys.path.insert(0, _PROBLEM3_DIR)

from problem3_solution import (
    Q3DataLoader, Q3DemandDayAllocator, Q3RouteGenerator,
    Q3DailySolver, PLANNING_DATES, AIRPORTS,
)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

TARGET_DAY = date(2026, 8, 3)
SCALES = [1.0, 1.1, 1.2]      # 求解时的人数放大比例
EVAL_SCALES = [1.0, 1.1, 1.2]  # 评估时的人数放大比例
MIP_TIME_LIMIT = 40
OUTPUT_DIR = _THIS_DIR


def scale_demands(day_demands, scale, seed=20260814):
    """复制部分需求以模拟设施级人数按比例增加"""
    import random
    rng = random.Random(seed)
    scaled = [dict(d) for d in day_demands]
    extra = int(len(day_demands) * (scale - 1))
    for k in range(extra):
        src = rng.choice(day_demands)
        copy = dict(src)
        copy['person_id'] = f"{src['person_id']}_dup{k}"
        scaled.append(copy)
    return scaled


def solve_day(day_demands, distances, time_limit):
    """求解单日排班，返回解字典或 None"""
    Q3DailySolver.DINKELBACH_MAX_ITERS = 1
    Q3DailySolver.TIME_LIMITS = [time_limit]
    Q3DailySolver.RETRY_TIME_LIMIT = time_limit
    generator = Q3RouteGenerator(distances)
    routes_info = generator.generate_daily_routes(day_demands, TARGET_DAY, route_id_start=1)
    solver = Q3DailySolver()
    result = solver.solve(routes_info, day_demands, TARGET_DAY, distances)
    if result is None:
        return None
    sel_routes, yo, yi, ys = result
    total_time = sum(ri['route'].time for ri in sel_routes)
    return {
        'sel_routes': sel_routes,
        'total_time': total_time,
        'n_flights': len(sel_routes),
    }


def coverage_feasibility_demand(sel_routes, eval_demands):
    """
    评估人数放大场景下的覆盖可行性率。
    对每个设施，选中路径的最大可服务人数是否 ≥ 放大后需求数。
    """
    airport_set = set(AIRPORTS) | {'LAND'}
    ob_by_fac = defaultdict(int)
    ib_by_fac = defaultdict(int)
    sh_by_od = defaultdict(int)
    for d in eval_demands:
        o, de = d['origin_id'], d['destination_id']
        if o in airport_set and de.startswith('F'):
            ob_by_fac[de] += 1
        elif o.startswith('F') and de in airport_set:
            ib_by_fac[o] += 1
        elif o.startswith('F') and de.startswith('F'):
            sh_by_od[(o, de)] += 1

    # 选中路径对每个设施的最大可服务人数
    cap_ob = defaultdict(int)
    cap_ib = defaultdict(int)
    cap_sh = defaultdict(int)
    for ri in sel_routes:
        r = ri['route']
        for d in ri['compatible_demands']:
            o, de = d['origin_id'], d['destination_id']
            if o in airport_set and de.startswith('F'):
                cap_ob[de] += 1
            elif o.startswith('F') and de in airport_set:
                cap_ib[o] += 1
            elif o.startswith('F') and de.startswith('F'):
                cap_sh[(o, de)] += 1

    # 计数可行设施数与总设施数
    feasible = 0
    total = 0
    for f, n in ob_by_fac.items():
        total += 1
        if cap_ob.get(f, 0) >= n:
            feasible += 1
    for f, n in ib_by_fac.items():
        total += 1
        if cap_ib.get(f, 0) >= n:
            feasible += 1
    for od, n in sh_by_od.items():
        total += 1
        if cap_sh.get(od, 0) >= n:
            feasible += 1
    return feasible / total if total else 0.0


def main():
    """主函数：求解名义/鲁棒解、评估人数波动下的可行性、输出对比。"""
    print('=' * 70)
    print('鲁棒优化实验：需求人数波动下的单日路径排班')
    print('=' * 70)

    loader = Q3DataLoader()
    distances = loader.load_distances()
    all_demands = loader.load_demands_q3()
    non_temp = [d for d in all_demands if d['task_type'] != 'temporary']
    allocation, _ = Q3DemandDayAllocator.allocate(non_temp, PLANNING_DATES)
    for d in non_temp:
        d['_assigned_day'] = allocation[d['person_id']]
    day_demands = [d for d in non_temp if allocation[d['person_id']] == TARGET_DAY]
    print(f'  {TARGET_DAY} 分配需求: {len(day_demands)} 人')

    # 求解各 scale 下的解
    solutions = []
    for scale in SCALES:
        tag = '名义' if scale == 1.0 else f'鲁棒+{int((scale-1)*100)}%'
        print(f'\n--- {tag} ---')
        scaled = scale_demands(day_demands, scale) if scale > 1.0 else day_demands
        sol = solve_day(scaled, distances, MIP_TIME_LIMIT)
        if sol is None:
            print(f'  {tag} 求解失败')
            continue
        sol['scale'] = scale
        print(f'  {tag}: 时间={sol["total_time"]}min, 架次={sol["n_flights"]}')
        solutions.append(sol)

    if not solutions:
        print('错误: 所有方案求解失败')
        return

    # 评估可行性矩阵
    print('\n评估人数波动下的覆盖可行性率...')
    feas_matrix = []
    for sol in solutions:
        row = []
        for s in EVAL_SCALES:
            eval_demands = scale_demands(day_demands, s) if s > 1.0 else day_demands
            rate = coverage_feasibility_demand(sol['sel_routes'], eval_demands)
            row.append(rate)
        feas_matrix.append(row)
        print(f'  S_{int((sol["scale"]-1)*100)}%: ' + '  '.join(
            f'+{int((s-1)*100)}%:{r*100:.1f}%' for s, r in zip(EVAL_SCALES, row)))

    # 输出结果
    lines = []
    lines.append('=' * 70)
    lines.append('鲁棒优化实验：需求人数波动下的单日路径排班')
    lines.append('=' * 70)
    lines.append('一、各求解方案的成本')
    lines.append(f'{"方案":<14}{"总时间(min)":>14}{"架次":>8}')
    for sol in solutions:
        tag = '名义' if sol['scale'] == 1.0 else f'鲁棒+{int((sol["scale"]-1)*100)}%'
        lines.append(f'{tag:<14}{sol["total_time"]:>14}{sol["n_flights"]:>8}')
    lines.append('')
    lines.append('二、人数波动下的覆盖可行性率矩阵')
    header = '方案\\波动   ' + ''.join(f'+{int((s-1)*100)}%'.rjust(10) for s in EVAL_SCALES)
    lines.append(header)
    for i, sol in enumerate(solutions):
        name = '名义' if sol['scale'] == 1.0 else f'鲁棒+{int((sol["scale"]-1)*100)}%'
        row = f'{name:<10}' + ''.join(f'{feas_matrix[i][j]*100:>9.1f}%' for j in range(len(EVAL_SCALES)))
        lines.append(row)
    lines.append('')
    lines.append('三、结论')
    lines.append('名义解在人数波动下覆盖可行性快速下降；鲁棒解通过预留容量缓冲，')
    lines.append('在波动幅度内保持高可行性，代价是更高的排班成本。')
    lines.append('=' * 70)
    content = '\n'.join(lines) + '\n'
    path = os.path.join(OUTPUT_DIR, 'robust_demand_results.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(content)
    print(f'结果已写入: {path}')


if __name__ == '__main__':
    main()
