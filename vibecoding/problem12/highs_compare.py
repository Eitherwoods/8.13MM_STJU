# -*- coding: utf-8 -*-
"""
文件: highs_compare.py
功能: COPT vs HiGHS 全面对比 - Q1 阶段一（MIP整数解 + LP松弛下界）
说明: LP松弛仅作下界参考，最终结果必须是MIP整数解
"""

import os, sys, time as time_module
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VIBECODING_DIR = os.path.dirname(_THIS_DIR)
if _VIBECODING_DIR not in sys.path:
    sys.path.insert(0, _VIBECODING_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import coptpy as cp
import highspy
from base_models import DataLoader
from problem1_solution import RouteGenerator

SOLVE_TIME_LIMIT = 60


def load_data():
    """加载 Q1 路径与需求数据"""
    loader = DataLoader()
    distances = loader.load_distances()
    demands = loader.load_demands_q1()
    demands_by_facility = loader.count_demands(demands)
    gen = RouteGenerator(distances)
    routes = gen.generate_all()
    return routes, demands_by_facility, distances


def _build_copt_model(routes, demands_by_facility, relax=False):
    """
    构建 COPT 模型

    关键: relax=True 时显式设置变量边界，与 HiGHS 保持一致
    - x: [0, 1]（连续或二元）
    - y: [0, demand_f]（连续或整数）
    """
    env = cp.Envr()
    model = env.createModel('Q1_Stage1')
    model.setParam(cp.COPT.Param.TimeLimit, SOLVE_TIME_LIMIT)
    model.setParam(cp.COPT.Param.RelGap, 0.03)
    model.setParam(cp.COPT.Param.LogToConsole, 0)

    vtype_x = cp.COPT.CONTINUOUS if relax else cp.COPT.BINARY
    vtype_y = cp.COPT.CONTINUOUS if relax else cp.COPT.INTEGER

    x, y = {}, {}
    for r in routes:
        x[r.route_id] = model.addVar(vtype=vtype_x, lb=0, ub=1,
                                      name=f'x_{r.route_id}')
        for f in r.dest_facilities:
            if f in demands_by_facility:
                y[(r.route_id, f)] = model.addVar(
                    vtype=vtype_y, lb=0,
                    ub=float(demands_by_facility[f]['total']),
                    name=f'y_{r.route_id}_{f}')

    for f, info in demands_by_facility.items():
        terms = [y[(r.route_id, f)] for r in routes
                 if f in r.dest_facilities and (r.route_id, f) in y]
        if terms:
            model.addConstr(cp.quicksum(terms) == info['total'])

    for f, info in demands_by_facility.items():
        for origin, cnt in info['by_origin'].items():
            if origin != 'LAND' and cnt > 0:
                terms = [y[(r.route_id, f)] for r in routes
                         if f in r.dest_facilities and r.airport == origin
                         and (r.route_id, f) in y]
                if terms:
                    model.addConstr(cp.quicksum(terms) >= cnt)

    for r in routes:
        terms = [y[(r.route_id, f)] for f in r.dest_facilities
                 if (r.route_id, f) in y]
        if terms:
            model.addConstr(cp.quicksum(terms) <= r.capacity * x[r.route_id])

    time_expr = cp.quicksum(r.time * x[r.route_id] for r in routes)
    model.setObjective(time_expr, sense=cp.COPT.MINIMIZE)
    return env, model, x, y


def solve_copt(routes, demands_by_facility, relax=False):
    """COPT 求解（MIP 或 LP 松弛）"""
    _, model, _, _ = _build_copt_model(routes, demands_by_facility, relax=relax)
    t0 = time_module.time()
    model.solve()
    elapsed = time_module.time() - t0
    status = model.status
    obj = float('nan')
    if status in (cp.COPT.OPTIMAL, cp.COPT.TIMEOUT,
                  cp.COPT.NODELIMIT, cp.COPT.INTERRUPTED):
        try:
            obj = model.objval
        except Exception:
            pass
    return obj, elapsed, status


def solve_highs(routes, demands_by_facility, relax=False):
    """HiGHS 求解（MIP 或 LP 松弛）"""
    h = highspy.Highs()
    h.setOptionValue('output_flag', False)
    h.setOptionValue('time_limit', SOLVE_TIME_LIMIT)

    col_idx = 0
    x_vars, y_vars = {}, {}

    for r in routes:
        vtype = highspy.HighsVarType.kContinuous if relax else highspy.HighsVarType.kInteger
        h.addCol(r.time, 0.0, 1.0, 0,
                 np.array([], dtype=np.int32), np.array([], dtype=np.float64))
        h.changeColIntegrality(col_idx, vtype)
        x_vars[r.route_id] = col_idx
        col_idx += 1
        for f in r.dest_facilities:
            if f in demands_by_facility:
                vtype_y = highspy.HighsVarType.kContinuous if relax else highspy.HighsVarType.kInteger
                h.addCol(0.0, 0.0, float(demands_by_facility[f]['total']), 0,
                         np.array([], dtype=np.int32), np.array([], dtype=np.float64))
                h.changeColIntegrality(col_idx, vtype_y)
                y_vars[(r.route_id, f)] = col_idx
                col_idx += 1

    row_starts, col_indices_list, values_list = [], [], []
    row_lower, row_upper = [], []

    def _add_row(indices, vals, lb, ub):
        row_starts.append(len(col_indices_list))
        col_indices_list.extend(indices)
        values_list.extend(vals)
        row_lower.append(lb)
        row_upper.append(ub)

    for f, info in demands_by_facility.items():
        idx, vals = [], []
        for r in routes:
            if f in r.dest_facilities and (r.route_id, f) in y_vars:
                idx.append(y_vars[(r.route_id, f)])
                vals.append(1.0)
        if idx:
            _add_row(idx, vals, float(info['total']), float(info['total']))

    for f, info in demands_by_facility.items():
        for origin, cnt in info['by_origin'].items():
            if origin != 'LAND' and cnt > 0:
                idx, vals = [], []
                for r in routes:
                    if f in r.dest_facilities and r.airport == origin \
                            and (r.route_id, f) in y_vars:
                        idx.append(y_vars[(r.route_id, f)])
                        vals.append(1.0)
                if idx:
                    _add_row(idx, vals, float(cnt), float('inf'))

    for r in routes:
        idx, vals = [], []
        for f in r.dest_facilities:
            if (r.route_id, f) in y_vars:
                idx.append(y_vars[(r.route_id, f)])
                vals.append(1.0)
        if idx:
            idx.append(x_vars[r.route_id])
            vals.append(-float(r.capacity))
            _add_row(idx, vals, -float('inf'), 0.0)

    n_rows = len(row_starts)
    h.addRows(n_rows,
              np.array(row_lower, dtype=np.float64),
              np.array(row_upper, dtype=np.float64),
              len(col_indices_list),
              np.array(row_starts, dtype=np.int32),
              np.array(col_indices_list, dtype=np.int32),
              np.array(values_list, dtype=np.float64))

    h.setOptionValue('presolve', 'on')
    t0 = time_module.time()
    h.run()
    elapsed = time_module.time() - t0

    status_str = str(h.getModelStatus())
    try:
        s, obj_val = h.getInfoValue('objective_function_value')
        obj = obj_val if s == highspy.HighsStatus.kOk else float('nan')
    except Exception:
        obj = float('nan')
    return obj, elapsed, status_str


def main():
    """主函数：四路对比（LP下界 + MIP整数解）"""
    print("=" * 70)
    print("COPT vs HiGHS 全面对比 - Q1 阶段一")
    print("  LP松弛 = 分数解（下界参考，不可直接使用）")
    print("  MIP    = 整数解（最终方案）")
    print("=" * 70)

    routes, demands_by_facility, distances = load_data()
    n_x = len(routes)
    n_y = sum(1 for r in routes for f in r.dest_facilities if f in demands_by_facility)
    print(f"\n问题规模: {n_x} 路径, {n_y} 分配变量, "
          f"{len(demands_by_facility)} 设施, 时限={SOLVE_TIME_LIMIT}s\n")

    results = []

    print("[1/4] COPT LP 松弛（分数解，下界） ...")
    obj, t, st = solve_copt(routes, demands_by_facility, relax=True)
    results.append(('COPT', 'LP松弛', obj, t, st))
    print(f"      目标={obj:.1f}  耗时={t:.3f}s")

    print("[2/4] HiGHS LP 松弛（分数解，下界） ...")
    obj, t, st = solve_highs(routes, demands_by_facility, relax=True)
    results.append(('HiGHS', 'LP松弛', obj, t, st))
    print(f"      目标={obj:.1f}  耗时={t:.3f}s")

    print("[3/4] COPT MIP（整数解） ...")
    obj, t, st = solve_copt(routes, demands_by_facility, relax=False)
    results.append(('COPT', 'MIP', obj, t, st))
    print(f"      目标={obj:.0f}  耗时={t:.2f}s")

    print("[4/4] HiGHS MIP（整数解） ...")
    obj, t, st = solve_highs(routes, demands_by_facility, relax=False)
    results.append(('HiGHS', 'MIP', obj, t, st))
    print(f"      目标={obj:.0f}  耗时={t:.2f}s")

    # 汇总
    print("\n" + "=" * 70)
    print("汇总对比")
    print("=" * 70)
    print(f"{'求解器':<8}{'模式':<10}{'目标(min)':>12}{'耗时(s)':>10}{'性质'}")
    print("-" * 70)
    for solver, mode, obj, elapsed, _ in results:
        if mode == 'LP松弛':
            print(f"{solver:<8}{mode:<10}{obj:>12.1f}{elapsed:>10.3f}  分数解（下界）")
        else:
            print(f"{solver:<8}{mode:<10}{obj:>12.0f}{elapsed:>10.2f}  整数解（可用）")
    print("-" * 70)

    lp = {s: o for s, m, o, _, _ in results if m == 'LP松弛'}
    mip = {s: o for s, m, o, _, _ in results if m == 'MIP'}

    if 'COPT' in lp and 'HiGHS' in lp:
        diff = abs(lp['COPT'] - lp['HiGHS'])
        print(f"\nLP 下界: COPT={lp['COPT']:.1f} vs HiGHS={lp['HiGHS']:.1f}  差={diff:.2f}")

    for s in ['COPT', 'HiGHS']:
        if s in lp and s in mip and np.isfinite(lp[s]) and np.isfinite(mip[s]):
            gap = (mip[s] - lp[s]) / mip[s] * 100
            print(f"{s} MIP(整数) vs LP(分数) 间隙: {mip[s]:.0f} vs {lp[s]:.1f} = {gap:.2f}%")

    if 'COPT' in mip and 'HiGHS' in mip:
        diff = (mip['HiGHS'] - mip['COPT']) / mip['COPT'] * 100
        print(f"\nMIP 整数解对比: HiGHS 比 COPT 差 {diff:+.2f}%")
        print(f"  COPT = {mip['COPT']:.0f} min（更优）")
        print(f"  HiGHS = {mip['HiGHS']:.0f} min")


if __name__ == '__main__':
    main()
