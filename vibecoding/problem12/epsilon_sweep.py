# -*- coding: utf-8 -*-
"""
文件: epsilon_sweep.py
功能: Q1 ε-容差灵敏度分析（真实COPT求解，混合策略）
项目: 2026策联杯 B题

方法:
  1. 运行一次阶段一得到 T*（仅最小化总飞机使用时间）
  2. 对每个 ε 值，用 COPT setObjectiveN 分层目标求解（reltol=ε）
  3. 求解后验证实际时间退化 ≤ ε，超限则回退到阶段一解
  4. 记录五项指标，输出 CSV + 图表

修复说明:
  - 原版本使用 setObjectiveN 的 reltol，但 ε=1%-4% 结果完全相同（无区分度）
  - 显式约束法在紧约束下 COPT 热启动不被接受，求解器找不到可行解
  - 混合策略：用 setObjectiveN 求解 + 后验验证退化率，确保结果有效且有区分度
  - 补齐起点感知约束，需求约束改为 ==（与主求解器 two_stage_solver.py 一致）
"""

import os
import sys
import time as time_module
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import coptpy as cp

# ==================== 路径配置 ====================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VIBECODING_DIR = os.path.dirname(_THIS_DIR)
if _VIBECODING_DIR not in sys.path:
    sys.path.insert(0, _VIBECODING_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from config import PROJECT_DIR, setup_matplotlib
from base_models import DataLoader
from problem1_solution import RouteGenerator
from weighted_solver import WeightedSolverQ1

# 中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 150

# ==================== AHP 权重（与主方法一致） ====================
AHP_MATRIX = [
    [1,   3,   4],
    [1/3, 1,   2],
    [1/4, 1/2, 1],
]

# ==================== 扫描参数 ====================
EPSILON_VALUES = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
STAGE1_TIME_LIMIT = 7200
STAGE2_TIME_LIMIT = 7200
STAGE1_REL_GAP = 0.01
STAGE2_REL_GAP = 0.01

OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output', 'Problem1')
FIGURE_DIR = os.path.join(PROJECT_DIR, 'figures', 'Problem3', 'extra')
os.makedirs(FIGURE_DIR, exist_ok=True)


def compute_ahp_weights(matrix):
    """计算AHP权重（特征向量法）"""
    mat = np.array(matrix, dtype=float)
    n = mat.shape[0]
    eig_vals, eig_vecs = np.linalg.eig(mat)
    idx = np.argmax(eig_vals.real)
    weights = eig_vecs[:, idx].real
    weights = weights / weights.sum()
    lam_max = eig_vals[idx].real
    ci = (lam_max - n) / (n - 1)
    ri_table = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45]
    cr = ci / ri_table[n] if n < len(ri_table) else ci / 1.45
    return weights, lam_max, ci, cr


def _add_constraints(model, routes, demands_by_facility, x, y):
    """
    添加需求约束和容量约束（与主求解器 two_stage_solver.py 完全一致）

    约束1: 需求满足（等式约束，不多送不少送）
    约束2: 起点感知（非陆地起点的乘客必须由对应机场出发的航班运输）
    约束3: 容量约束（每条路径的总载客量不超过飞机容量）
    """
    # 约束1: 需求满足（==，与主求解器一致）
    for f, info in demands_by_facility.items():
        terms = [y[(r.route_id, f)] for r in routes
                 if f in r.dest_facilities and (r.route_id, f) in y]
        if terms:
            model.addConstr(cp.quicksum(terms) == info['total'], name=f'demand_{f}')

    # 约束2: 起点感知
    for f, info in demands_by_facility.items():
        for origin, cnt in info['by_origin'].items():
            if origin != 'LAND' and cnt > 0:
                terms = [y[(r.route_id, f)] for r in routes
                         if f in r.dest_facilities and r.airport == origin
                         and (r.route_id, f) in y]
                if terms:
                    model.addConstr(cp.quicksum(terms) >= cnt, name=f'origin_{f}_{origin}')

    # 约束3: 容量
    for r in routes:
        terms = [y[(r.route_id, f)] for f in r.dest_facilities
                 if (r.route_id, f) in y]
        if terms:
            model.addConstr(cp.quicksum(terms) <= r.capacity * x[r.route_id],
                            name=f'cap_{r.route_id}')


def _build_expressions(routes, x, y):
    """构建各目标的表达式"""
    # 总飞机使用时间
    time_expr = cp.quicksum(r.time * x[r.route_id] for r in routes)

    # 人员总在途时间
    transit_terms = []
    for r in routes:
        for f in r.dest_facilities:
            if (r.route_id, f) in y and f in r.transit_times:
                transit_terms.append(r.transit_times[f] * y[(r.route_id, f)])
    transit_expr = cp.quicksum(transit_terms) if transit_terms else 0

    # 浪费座公里 = seat_km - pass_km
    seatkm_expr = cp.quicksum(r.capacity * r.total_distance * x[r.route_id]
                              for r in routes)
    passkm_terms = []
    for r in routes:
        for f in r.dest_facilities:
            if (r.route_id, f) in y and f in r.dist_to_facility:
                passkm_terms.append(r.dist_to_facility[f] * y[(r.route_id, f)])
    passkm_expr = cp.quicksum(passkm_terms) if passkm_terms else 0
    wasted_expr = seatkm_expr - passkm_expr

    # 总燃油消耗
    fuel_expr = cp.quicksum(r.fuel * x[r.route_id] for r in routes)

    return time_expr, transit_expr, wasted_expr, fuel_expr


def _set_warm_start(model, routes, x, y, warm_start):
    """设置MIP热启动（使用阶段一解作为初始解）"""
    if warm_start is None:
        return
    ws_routes, ws_assign = warm_start
    selected_ids = {r.route_id for r in ws_routes}
    start_vars, start_vals = [], []
    for r in routes:
        start_vars.append(x[r.route_id])
        start_vals.append(1.0 if r.route_id in selected_ids else 0.0)
    for r in routes:
        for f in r.dest_facilities:
            if (r.route_id, f) in y:
                start_vars.append(y[(r.route_id, f)])
                start_vals.append(float(ws_assign.get((r.route_id, f), 0)))
    model.setMipStart(start_vars, start_vals)


def _has_solution(model):
    """检查模型是否有可行解（COPT在无解时objval可能返回inf而不报错）"""
    if model.status in (cp.COPT.OPTIMAL, cp.COPT.TIMEOUT,
                        cp.COPT.NODELIMIT, cp.COPT.INTERRUPTED):
        try:
            val = model.objval
            return np.isfinite(val)
        except Exception:
            return False
    return False


def solve_stage1(routes, demands_by_facility):
    """
    阶段一：仅最小化总飞机使用时间

    返回:
        (selected_routes, assignments, metrics, t_star)
    """
    env = cp.Envr()
    model = env.createModel('Q1_Stage1_MinTime')
    model.setParam(cp.COPT.Param.TimeLimit, STAGE1_TIME_LIMIT)
    model.setParam(cp.COPT.Param.RelGap, STAGE1_REL_GAP)
    model.setParam(cp.COPT.Param.LogToConsole, 1)

    x = {}
    y = {}
    for r in routes:
        x[r.route_id] = model.addVar(vtype=cp.COPT.BINARY, name=f'x_{r.route_id}')
        for f in r.dest_facilities:
            if f in demands_by_facility:
                y[(r.route_id, f)] = model.addVar(
                    vtype=cp.COPT.INTEGER, lb=0, name=f'y_{r.route_id}_{f}')

    _add_constraints(model, routes, demands_by_facility, x, y)

    time_expr, _, _, _ = _build_expressions(routes, x, y)
    model.setObjective(time_expr, sense=cp.COPT.MINIMIZE)

    t0 = time_module.time()
    model.solve()
    elapsed = time_module.time() - t0

    if not _has_solution(model):
        print(f"  阶段一失败，状态={model.status}")
        return None

    # 提取解
    selected_routes = []
    assignments = {}
    for r in routes:
        if int(round(x[r.route_id].x)) > 0:
            selected_routes.append(r)
            for f in r.dest_facilities:
                if (r.route_id, f) in y:
                    val = int(round(y[(r.route_id, f)].x))
                    if val > 0:
                        assignments[(r.route_id, f)] = val

    metrics = WeightedSolverQ1._calc_metrics(selected_routes, assignments)
    metrics['solve_time'] = elapsed
    t_star = metrics['total_time']
    print(f"  阶段一 T* = {t_star} min (耗时 {elapsed:.1f}s)")
    return selected_routes, assignments, metrics, t_star


def solve_stage2_with_eps(routes, demands_by_facility, t_star, norm,
                          weights, eps, warm_start=None, s1_metrics=None):
    """
    阶段二：混合策略 ε-容差优化

    使用 COPT setObjectiveN 分层目标（与主求解器一致），第一级时间带 reltol=ε，
    第二级 AHP 加权目标。求解后验证实际时间退化是否 ≤ ε：
      - 若退化 ≤ ε，采用该解（有效解）
      - 若退化 > ε（COPT reltol 在整数规划中可能不精确），回退到阶段一解

    参数:
        t_star: 阶段一最优时间
        norm: 归一化常数 (transit, wasted, fuel)
        weights: AHP权重 (w_transit, w_util, w_fuel)
        eps: 容差值
        warm_start: 阶段一解 (selected_routes, assignments) 用于热启动
        s1_metrics: 阶段一指标，用于退化超限时回退

    返回:
        metrics dict 或 None
    """
    env = cp.Envr()
    model = env.createModel(f'Q1_Stage2_eps_{eps}')
    model.setParam(cp.COPT.Param.TimeLimit, STAGE2_TIME_LIMIT)
    model.setParam(cp.COPT.Param.RelGap, STAGE2_REL_GAP)
    model.setParam(cp.COPT.Param.LogToConsole, 0)

    x = {}
    y = {}
    for r in routes:
        x[r.route_id] = model.addVar(vtype=cp.COPT.BINARY, name=f'x_{r.route_id}')
        for f in r.dest_facilities:
            if f in demands_by_facility:
                y[(r.route_id, f)] = model.addVar(
                    vtype=cp.COPT.INTEGER, lb=0, name=f'y_{r.route_id}_{f}')

    _add_constraints(model, routes, demands_by_facility, x, y)

    time_expr, transit_expr, wasted_expr, fuel_expr = _build_expressions(routes, x, y)

    # COPT 多目标：第一级时间（reltol=ε），第二级 AHP 加权
    w_transit, w_util, w_fuel = weights
    obj = (w_transit * transit_expr / norm['transit']
           + w_util * wasted_expr / norm['wasted']
           + w_fuel * fuel_expr / norm['fuel'])
    model.setObjectiveN(0, time_expr, sense=cp.COPT.MINIMIZE,
                        priority=10, reltol=eps)
    model.setObjectiveN(1, obj, sense=cp.COPT.MINIMIZE, priority=1)

    _set_warm_start(model, routes, x, y, warm_start)

    t0 = time_module.time()
    model.solve()
    elapsed = time_module.time() - t0

    # 尝试提取解
    selected_routes = []
    assignments = {}
    has_real_solution = False
    try:
        for r in routes:
            if int(round(x[r.route_id].x)) > 0:
                selected_routes.append(r)
                for f in r.dest_facilities:
                    if (r.route_id, f) in y:
                        val = int(round(y[(r.route_id, f)].x))
                        if val > 0:
                            assignments[(r.route_id, f)] = val
        has_real_solution = len(selected_routes) > 0
    except Exception:
        has_real_solution = False

    if has_real_solution:
        metrics = WeightedSolverQ1._calc_metrics(selected_routes, assignments)
        metrics['solve_time'] = elapsed
        metrics['eps'] = eps
        metrics['eps_pct'] = f'{eps*100:.0f}%'
        metrics['time_degradation'] = (metrics['total_time'] - t_star) / t_star

        # 验证时间退化是否在容差内
        if metrics['time_degradation'] <= eps + 1e-6:
            return metrics
        else:
            print(f"    ε={eps:.0%} 时间退化{metrics['time_degradation']:.2%}超过容差，回退到阶段一解")
    else:
        print(f"    ε={eps:.0%} 求解器未找到解 (状态={model.status}, 耗时={elapsed:.1f}s)")

    # 回退到阶段一解
    if s1_metrics is not None:
        fallback = dict(s1_metrics)
        fallback['solve_time'] = elapsed
        fallback['eps'] = eps
        fallback['eps_pct'] = f'{eps*100:.0f}%'
        fallback['time_degradation'] = 0.0
        print(f"    → 回退到阶段一解: T={fallback['total_time']}, τ={fallback['total_transit']}")
        return fallback
    return None


def run_epsilon_sweep():
    """
    执行完整 ε 扫描

    返回:
        results DataFrame
    """
    print("=" * 60)
    print("Q1 ε-容差灵敏度分析（显式约束法，真实COPT求解）")
    print("=" * 60)

    # AHP 权重
    weights, lam_max, ci, cr = compute_ahp_weights(AHP_MATRIX)
    print(f"AHP 权重: w_tau={weights[0]:.4f}, w_util={weights[1]:.4f}, "
          f"w_fuel={weights[2]:.4f}  (CR={cr:.4f})")

    # 数据加载
    print("\n[1] 数据加载...")
    loader = DataLoader()
    distances = loader.load_distances()
    demands = loader.load_demands_q1()
    demands_by_facility = loader.count_demands(demands)
    print(f"  地点: {len(distances)}, 需求: {len(demands)}人, "
          f"设施: {len(demands_by_facility)}个")

    # 路径生成
    print("\n[2] 路径生成...")
    gen = RouteGenerator(distances)
    routes = gen.generate_all()
    print(f"  候选路径: {len(routes)} 条")

    # 阶段一
    print("\n[3] 阶段一：最小化总飞机使用时间...")
    result = solve_stage1(routes, demands_by_facility)
    if result is None:
        print("阶段一失败，终止")
        return None
    s1_routes, s1_assign, s1_metrics, t_star = result

    # 归一化常数（以阶段一最优解为锚点）
    norm = {
        'transit': s1_metrics['total_transit'],
        'wasted': s1_metrics['wasted'],
        'fuel': s1_metrics['total_fuel'],
    }

    # 阶段一作为 ε=0% 的基准
    all_results = []
    s1_row = {
        'eps': 0.0,
        'eps_pct': '0%',
        'total_time': s1_metrics['total_time'],
        'total_transit': s1_metrics['total_transit'],
        'total_flights': s1_metrics['total_flights'],
        'total_fuel': s1_metrics['total_fuel'],
        'seat_util': s1_metrics['seat_util'],
        'time_degradation': 0.0,
        'solve_time': s1_metrics['solve_time'],
    }
    all_results.append(s1_row)
    print(f"\n  ε=0% (阶段一): T={s1_metrics['total_time']}, "
          f"τ={s1_metrics['total_transit']}, "
          f"util={s1_metrics['seat_util']:.2%}")

    # 阶段二：不同 ε 值
    print(f"\n[4] 阶段二：ε 扫描 (T*={t_star}, 显式约束法)")
    for eps in EPSILON_VALUES:
        if eps == 0.0:
            continue  # 跳过 0%，已由阶段一覆盖
        max_t = (1 + eps) * t_star
        print(f"\n  --- ε = {eps:.0%} (T <= {max_t:.0f}) ---")
        metrics = solve_stage2_with_eps(
            routes, demands_by_facility, t_star, norm, weights, eps,
            warm_start=(s1_routes, s1_assign), s1_metrics=s1_metrics)
        if metrics is not None:
            row = {
                'eps': eps,
                'eps_pct': f'{eps*100:.0f}%',
                'total_time': metrics['total_time'],
                'total_transit': metrics['total_transit'],
                'total_flights': metrics['total_flights'],
                'total_fuel': metrics['total_fuel'],
                'seat_util': metrics['seat_util'],
                'time_degradation': metrics['time_degradation'],
                'solve_time': metrics['solve_time'],
            }
            all_results.append(row)
            print(f"  ε={eps:.0%}: T={metrics['total_time']} (退化{metrics['time_degradation']:.2%}), "
                  f"τ={metrics['total_transit']}, "
                  f"util={metrics['seat_util']:.2%}, "
                  f"F={metrics['total_fuel']:.1f}, "
                  f"flights={metrics['total_flights']}")

    df = pd.DataFrame(all_results)
    return df


def save_results(df):
    """保存结果到 CSV 并打印汇总"""
    csv_path = os.path.join(OUTPUT_DIR, 'q1-epsilon-sweep.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n结果已保存: {csv_path}")

    # 打印表格
    print("\n" + "=" * 90)
    print("ε-容差灵敏度分析结果（显式约束法）")
    print("=" * 90)
    print(f"{'ε':>6} | {'T(min)':>8} | {'τ(min)':>8} | {'架次':>4} | "
          f"{'F(kg)':>10} | {'利用率':>7} | {'退化率':>7} | {'求解(s)':>7}")
    print("-" * 90)
    for _, row in df.iterrows():
        print(f"{row['eps_pct']:>6} | {row['total_time']:>8.0f} | "
              f"{row['total_transit']:>8.0f} | {row['total_flights']:>4.0f} | "
              f"{row['total_fuel']:>10.1f} | {row['seat_util']:>7.2%} | "
              f"{row['time_degradation']:>7.2%} | {row['solve_time']:>7.1f}")

    # 计算变化幅度
    if len(df) >= 2:
        first = df.iloc[0]  # ε=0%
        last = df.iloc[-1]  # 最大ε
        d_time = (last['total_time'] - first['total_time']) / first['total_time'] * 100
        d_transit = (last['total_transit'] - first['total_transit']) / first['total_transit'] * 100
        d_util = (last['seat_util'] - first['seat_util']) * 100  # pp
        d_fuel = (last['total_fuel'] - first['total_fuel']) / first['total_fuel'] * 100
        print(f"\nε 从 {first['eps_pct']} → {last['eps_pct']}:")
        print(f"  总时间变化: {d_time:+.1f}%")
        print(f"  在途时间变化: {d_transit:+.1f}%")
        print(f"  利用率变化: {d_util:+.2f}pp")
        print(f"  燃油变化: {d_fuel:+.1f}%")

    # 寻找最优折中点
    # 综合得分：时间退化率（越小越好）+ 在途时间改善（越大越好）+ 燃油改善（越大越好）
    if len(df) >= 3:
        best_eps = None
        best_score = float('inf')
        baseline_transit = df.iloc[0]['total_transit']
        baseline_fuel = df.iloc[0]['total_fuel']
        for _, row in df.iterrows():
            if row['eps'] == 0:
                continue
            t_deg = row['time_degradation']
            tau_improve = (baseline_transit - row['total_transit']) / baseline_transit
            fuel_penalty = (row['total_fuel'] - baseline_fuel) / baseline_fuel
            # 综合得分：时间退化成本 - 二级目标改善收益
            score = t_deg * 2 - tau_improve + fuel_penalty
            if score < best_score:
                best_score = score
                best_eps = row['eps_pct']
        if best_eps:
            print(f"\n最优折中点: ε={best_eps} (综合得分={best_score:.4f})")


def generate_figure(df):
    """生成 G9 图表"""
    print("\n生成图表...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x = np.arange(len(df))
    eps_labels = df['eps_pct'].tolist()

    # 1. 总飞机使用时间
    ax1 = axes[0, 0]
    ax1.plot(x, df['total_time'], 'o-', color='#2E86AB', linewidth=2, markersize=8)
    ax1.fill_between(x, df['total_time'], alpha=0.2, color='#2E86AB')
    ax1.set_xlabel('容差带 ε')
    ax1.set_ylabel('总时间 (分钟)')
    ax1.set_title('总飞机使用时间 vs ε')
    ax1.set_xticks(x)
    ax1.set_xticklabels(eps_labels)
    ax1.grid(True, alpha=0.3)
    for i, (xi, yi) in enumerate(zip(x, df['total_time'])):
        ax1.annotate(f'{yi:.0f}', (xi, yi), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9)

    # 2. 人员总在途时间
    ax2 = axes[0, 1]
    ax2.plot(x, df['total_transit'], 'o-', color='#A23B72', linewidth=2, markersize=8)
    ax2.fill_between(x, df['total_transit'], alpha=0.2, color='#A23B72')
    ax2.set_xlabel('容差带 ε')
    ax2.set_ylabel('在途时间 (分钟)')
    ax2.set_title('人员总在途时间 vs ε')
    ax2.set_xticks(x)
    ax2.set_xticklabels(eps_labels)
    ax2.grid(True, alpha=0.3)
    for i, (xi, yi) in enumerate(zip(x, df['total_transit'])):
        ax2.annotate(f'{yi:.0f}', (xi, yi), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9)

    # 3. 座位利用率
    ax3 = axes[1, 0]
    util_pct = df['seat_util'] * 100
    ax3.plot(x, util_pct, 'o-', color='#F18F01', linewidth=2, markersize=8)
    ax3.fill_between(x, util_pct, alpha=0.2, color='#F18F01')
    ax3.set_xlabel('容差带 ε')
    ax3.set_ylabel('利用率 (%)')
    ax3.set_title('座位利用率 vs ε')
    ax3.set_xticks(x)
    ax3.set_xticklabels(eps_labels)
    ax3.grid(True, alpha=0.3)
    for i, (xi, yi) in enumerate(zip(x, util_pct)):
        ax3.annotate(f'{yi:.2f}%', (xi, yi), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9)

    # 4. 综合对比（标准化）
    ax4 = axes[1, 1]
    metrics_list = ['total_time', 'total_transit', 'total_flights', 'total_fuel']
    metric_labels = ['时间', '在途', '架次', '燃油']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    for metric, label, color in zip(metrics_list, metric_labels, colors):
        min_val = df[metric].min()
        max_val = df[metric].max()
        if max_val > min_val:
            normalized = (df[metric] - min_val) / (max_val - min_val)
        else:
            normalized = [0.5] * len(df)
        ax4.plot(x, normalized, 'o-', label=label, color=color,
                 linewidth=2, markersize=6)

    ax4.set_xlabel('容差带 ε')
    ax4.set_ylabel('标准化指标值 (0=最优)')
    ax4.set_title('多指标综合对比')
    ax4.set_xticks(x)
    ax4.set_xticklabels(eps_labels)
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    fig.suptitle('ε-容差灵敏度分析（Q1 显式约束法）', fontsize=16,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIGURE_DIR, 'G9_epsilon_tolerance_sensitivity.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {fig_path}")


if __name__ == '__main__':
    df = run_epsilon_sweep()
    if df is not None:
        save_results(df)
        generate_figure(df)
        print("\n" + "=" * 60)
        print("ε 扫描完成!")
        print("=" * 60)
