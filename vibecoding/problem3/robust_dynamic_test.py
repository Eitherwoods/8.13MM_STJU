# -*- coding: utf-8 -*-
"""
文件: robust_dynamic_test.py
功能: 模型推广实验（适配版）— 服务日入列框架下的鲁棒性与动态调度数值测试
项目: 2026策联杯 B题 - 海上油田人员直升机运载计划编排

设计说明（2026-08-17 适配）:
  旧版依赖已删除的 problem3_solution.Q3DemandDayAllocator（贪心按天分配）。
  新框架下服务日由 MILP 全局决定，因此鲁棒性度量升级为更基础的
  **可行日期集合稳定率**：时间窗扰动后，需求可行日期集完全不变的比例。
  该度量不依赖具体分配规则，是服务日入列框架下解稳定的必要条件，
  且同样体现“跨天时间窗缓冲”这一鲁棒性来源。

  实验:
  1. 鲁棒性: 6 个扰动场景 x MC_RUNS 次蒙特卡洛（numpy 向量化），
     统计可行集稳定率（均值/标准差）、平均可行天数、不可行需求数。
  2. 动态调度: 7 天滚动，每日扰动 5% 未执行需求（最晚到达提前 18~36h），
     对比“静态不调整”与“滚动重分配”的可行性、调整成本与负载均衡。

输出:
  output/Problem3/robust_dynamic_results.txt
  figures/Problem3/extra/R1_robust_stability_bars.png

运行:
  python robust_dynamic_test.py
  $env:ROBUST_MC_RUNS=2000; python robust_dynamic_test.py   # 控制蒙特卡洛次数
"""

from __future__ import annotations

import os
import random
import sys
from collections import defaultdict
from datetime import timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VIBECODING_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_DIR = os.path.dirname(_VIBECODING_DIR)
if _VIBECODING_DIR not in sys.path:
    sys.path.insert(0, _VIBECODING_DIR)

from problem3_service_day import (  # noqa: E402
    CURFEW, DAY0, N_DAYS, T0, load_data,
)

# ==================== 全局配置 ====================
MC_RUNS = int(os.environ.get('ROBUST_MC_RUNS', '5000'))
DYNAMIC_PERTURB_RATIO = 0.05
SEED = 20260814
FIGURES_DIR = os.path.join(_PROJECT_DIR, 'figures', 'Problem3', 'extra')
RESULTS_PATH = os.path.join(_PROJECT_DIR, 'output', 'Problem3',
                            'robust_dynamic_results.txt')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 场景定义：label, mode, magnitude(分钟)
SCENARIOS = [
    ('相对 ±10%', 'relative', 0.10),
    ('相对 ±20%', 'relative', 0.20),
    ('绝对 ±60 min', 'absolute', 60),
    ('绝对 ±120 min', 'absolute', 120),
    ('紧急化 ≤12 h', 'tighten', 720),
    ('紧急化 ≤24 h', 'tighten', 1440),
]


# ==================== 数据与可行日期集 ====================
def _day_bounds():
    """返回每日运营时段（绝对分钟）：[start_d, end_d)"""
    starts = np.array([d * 1440 + T0 for d in range(N_DAYS)], dtype=np.int64)
    ends = np.array([d * 1440 + CURFEW for d in range(N_DAYS)], dtype=np.int64)
    return starts, ends


def feasible_matrix(ep, la, starts, ends):
    """需求可行日期矩阵：req x day，布尔（与论文/验证器口径一致）"""
    return (ep[:, None] < ends[None, :]) & (la[:, None] > starts[None, :])


def perturb_arrays(ep, la, mode, magnitude, rng):
    """对 ep/la 数组施加随机扰动，返回扰动后的 (ep2, la2)"""
    n = len(ep)
    width = np.maximum((la - ep).astype(np.float64), 1.0)
    if mode == 'relative':
        de = rng.uniform(-magnitude, magnitude, n) * width
        dl = rng.uniform(-magnitude, magnitude, n) * width
    elif mode == 'tighten':
        de = np.zeros(n)
        dl = -rng.uniform(0.0, magnitude, n)
    else:
        de = rng.uniform(-magnitude, magnitude, n)
        dl = rng.uniform(-magnitude, magnitude, n)
    ep2 = ep + np.rint(de).astype(np.int64)
    la2 = la + np.rint(dl).astype(np.int64)
    # 扰动后窗口退化为空时保留 1 分钟有效宽度
    la2 = np.maximum(la2, ep2 + 1)
    return ep2, la2


def _feasible_days_for_rec(rec, dates):
    """单条需求的可行日期列表（绝对分钟口径，供动态调度模拟使用）"""
    out = []
    for d in dates:
        ds = d * 1440 + T0
        de = d * 1440 + CURFEW
        if rec.ep < de and rec.la > ds:
            out.append(d)
    return out


def greedy_alloc(recs, dates):
    """贪心负载均衡分配（仅用于动态调度模拟基准，非主求解器）。

    按可行日期数升序处理，选择负载最小的可行日，使每日负载尽量均衡。
    """
    by_pid = {}
    load = defaultdict(int)
    ordered = sorted(recs, key=lambda r: len(_feasible_days_for_rec(r, dates)))
    for r in ordered:
        feas = _feasible_days_for_rec(r, dates)
        if not feas:
            by_pid[r.pid] = dates[0]
            load[dates[0]] += 1
            continue
        best = min(feas, key=lambda d: load[d])
        by_pid[r.pid] = best
        load[best] += 1
    return by_pid, load


def load_std_cv(load, dates):
    """每日负载的标准差与变异系数"""
    vals = np.array([load.get(d, 0) for d in dates], dtype=np.float64)
    std = vals.std()
    cv = std / vals.mean() if vals.mean() > 0 else 0.0
    return float(std), float(cv)


# ==================== 鲁棒性实验 ====================
def run_robustness(recs, mc_runs, seed):
    """6 场景 x mc_runs 次蒙特卡洛，返回每场景汇总指标"""
    ep = np.array([r.ep for r in recs], dtype=np.int64)
    la = np.array([r.la for r in recs], dtype=np.int64)
    starts, ends = _day_bounds()
    feas_base = feasible_matrix(ep, la, starts, ends)
    n = len(recs)

    results = []
    for label, mode, mag in SCENARIOS:
        # 确定性种子：seed + 标签字符和（避免 Python hash 随机化）
        rng = np.random.default_rng(seed + sum(ord(c) for c in label))
        stable_list = []
        avg_feas_list = []
        infeas_list = []
        for _ in range(mc_runs):
            ep2, la2 = perturb_arrays(ep, la, mode, mag, rng)
            feas = feasible_matrix(ep2, la2, starts, ends)
            stable_list.append((feas == feas_base).all(axis=1).mean())
            n_feas = feas.sum(axis=1)
            avg_feas_list.append(n_feas.mean())
            infeas_list.append(int((n_feas == 0).sum()))
        s = np.array(stable_list)
        results.append({
            'label': label,
            'stability_mean': float(s.mean()),
            'stability_std': float(s.std()),
            'avg_feasible': float(np.mean(avg_feas_list)),
            'infeasible': int(np.mean(infeas_list)),
            'n': n,
        })
        print(f'  [{label}] 稳定率={s.mean()*100:.1f}% ± {s.std()*100:.2f}%, '
              f'平均可行天数={np.mean(avg_feas_list):.2f}, '
              f'不可行需求={int(np.mean(infeas_list))}', flush=True)
    return results


# ==================== 动态调度实验 ====================
def run_dynamic(recs, perturb_ratio, seed):
    """7 天滚动重调度模拟：每日扰动 5% 未执行需求，la 提前 18~36h"""
    rng = random.Random(seed)
    dates = list(range(N_DAYS))
    base_alloc, base_load = greedy_alloc(recs, dates)
    base_std, base_cv = load_std_cv(base_load, dates)
    current_alloc = dict(base_alloc)
    current_load = dict(base_load)
    rec_by_pid = {r.pid: r for r in recs}
    perturbed_pids = set()

    total_affected = 0
    total_static_infeasible = 0
    total_adjusted = 0
    daily_log = []

    for day in dates:
        pending = [pid for pid, d0 in current_alloc.items()
                   if d0 >= day and pid not in perturbed_pids]
        if not pending:
            daily_log.append({'day': day, 'affected': 0,
                              'static_infeasible': 0, 'adjusted': 0})
            continue
        n_affected = max(1, int(len(pending) * perturb_ratio))
        affected = rng.sample(pending, n_affected)
        n_static = 0
        n_adjusted = 0
        for pid in affected:
            perturbed_pids.add(pid)
            rec = rec_by_pid[pid]
            shift = rng.uniform(1080, 2160)  # 最晚到达提前 18~36h
            rec.la = max(rec.ep + 1, rec.la - int(shift))
            old_day = current_alloc[pid]
            feas = _feasible_days_for_rec(rec, dates)
            if old_day not in feas:
                n_static += 1
                n_adjusted += 1
                new_day = min(feas, key=lambda d: current_load.get(d, 0)) if feas else old_day
                current_alloc[pid] = new_day
        current_load = defaultdict(int)
        for pid, d0 in current_alloc.items():
            current_load[d0] += 1
        total_affected += n_affected
        total_static_infeasible += n_static
        total_adjusted += n_adjusted
        daily_log.append({'day': day, 'affected': n_affected,
                          'static_infeasible': n_static, 'adjusted': n_adjusted})
        print(f'  day{day}: 受影响 {n_affected}, 静态不可行 {n_static}, '
              f'调整 {n_adjusted}', flush=True)

    final_std, final_cv = load_std_cv(current_load, dates)
    # 最终全部需求是否仍可行
    final_feasible = sum(
        1 for pid, d0 in current_alloc.items()
        if d0 in _feasible_days_for_rec(rec_by_pid[pid], dates))
    return {
        'daily_log': daily_log,
        'total_affected': total_affected,
        'static_infeasible': total_static_infeasible,
        'adjusted': total_adjusted,
        'adjust_rate': total_adjusted / total_affected if total_affected else 0.0,
        'static_infeasible_rate': (total_static_infeasible / total_affected
                                   if total_affected else 0.0),
        'final_feasible': final_feasible,
        'final_feasible_rate': final_feasible / len(recs),
        'base_cv': base_cv,
        'final_cv': final_cv,
        'final_std': final_std,
    }


# ==================== 输出 ====================
def write_results(robust_results, dynamic):
    """写结果文本文件并打印"""
    lines = []
    lines.append('=' * 70)
    lines.append('鲁棒性与动态调度实验（服务日入列框架适配版）')
    lines.append('=' * 70)
    lines.append('一、鲁棒性：可行日期集合稳定率（5000 次蒙特卡洛）')
    lines.append(f'{"扰动场景":<16}{"可行集稳定率":>14}{"标准差":>10}'
                 f'{"平均可行天数":>14}{"不可行需求":>10}')
    for r in robust_results:
        lines.append(f'{r["label"]:<16}{r["stability_mean"]*100:>12.1f}%'
                     f'{r["stability_std"]*100:>10.2f}%'
                     f'{r["avg_feasible"]:>14.2f}{r["infeasible"]:>10}')
    lines.append('')
    lines.append('二、动态调度（每日扰动 5%，la 提前 18~36h）')
    lines.append(f'累计受影响需求: {dynamic["total_affected"]}')
    lines.append(f'静态不可行: {dynamic["static_infeasible"]} '
                 f'({dynamic["static_infeasible_rate"]*100:.1f}%)')
    lines.append(f'滚动调整: {dynamic["adjusted"]} '
                 f'({dynamic["adjust_rate"]*100:.1f}%)')
    lines.append(f'最终可行率: {dynamic["final_feasible_rate"]*100:.2f}%')
    lines.append(f'负载变异系数: {dynamic["base_cv"]:.4f} -> {dynamic["final_cv"]:.4f}')
    lines.append('=' * 70)
    content = '\n'.join(lines) + '\n'
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        f.write(content)
    print(content)
    print(f'结果已写入: {RESULTS_PATH}')


def plot_stability(robust_results):
    """绘制 R1：六场景可行集稳定率柱状图（含标准差误差棒）"""
    labels = [r['label'] for r in robust_results]
    means = [r['stability_mean'] * 100 for r in robust_results]
    stds = [r['stability_std'] * 100 for r in robust_results]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.85,
                  color='#2C5F8D', edgecolor='black', linewidth=0.6)
    for xi, m in zip(x, means):
        ax.text(xi, m + 0.8, f'{m:.1f}%', ha='center', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('可行日期集合稳定率 (%)', fontsize=11)
    ax.set_ylim(0, 105)
    ax.axhline(100, color='gray', linestyle='--', alpha=0.6)
    ax.set_title('六场景可行日期集合稳定率对比', fontsize=13, fontweight='bold')
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, 'R1_robust_stability_bars.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'图表已保存: {path}')


def main():
    """运行鲁棒性与动态调度实验、输出结果并绘图"""
    print('加载需求数据...')
    _distances, recs = load_data()
    print(f'加载 {len(recs)} 条需求')

    print(f'运行鲁棒性实验（{len(SCENARIOS)} 个场景 x {MC_RUNS} 次蒙特卡洛）...')
    robust_results = run_robustness(recs, MC_RUNS, SEED)

    print('运行动态调度实验（7 天滚动）...')
    dynamic = run_dynamic(recs, DYNAMIC_PERTURB_RATIO, SEED)

    write_results(robust_results, dynamic)
    plot_stability(robust_results)


if __name__ == '__main__':
    main()
