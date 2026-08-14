# -*- coding: utf-8 -*-
"""
文件: robust_dynamic_test.py
功能: 模型推广实验 — 鲁棒优化框架与动态调度机制的数值测试与可视化
项目: 2026策联杯 B题 - 海上油田人员直升机运载计划编排

实验说明:
  本脚本复用问题三的 Q3DemandDayAllocator（需求-日期分配器），
  不重跑大规模 MIP，以保证实验轻量、可复现。
  1. 鲁棒性实验: 对需求时间窗施加相对/绝对/单边收紧扰动，测量分配稳定性，
     每个场景 2000 次蒙特卡洛，多进程并行加速。
  2. 动态调度实验: 按 7 天滚动，每天随机扰动一部分未执行需求，
     对比"静态不调整"与"滚动重分配"的可行性与调整成本。

输出:
  结果写入 output/Problem3/robust_dynamic_results.txt
  图表写入 figures/Problem3/extra/R1~R4
"""

import os
import sys
import random
import multiprocessing as mp
from datetime import datetime, date, timedelta
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==================== 模块路径 ====================
_PROJECT_DIR = r"D:\数模\26策联杯\8.13MM_STJU"
_VIBECODING_DIR = os.path.join(_PROJECT_DIR, 'vibecoding')
_THIS_DIR = os.path.join(_VIBECODING_DIR, 'problem3')
if _VIBECODING_DIR not in sys.path:
    sys.path.insert(0, _VIBECODING_DIR)
# 同目录导入 problem3_solution
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from problem3_solution import Q3DataLoader, Q3DemandDayAllocator, PLANNING_DATES

# ==================== 全局配置 ====================
MC_RUNS = int(os.environ.get('ROBUST_MC_RUNS', '5000'))  # 每个场景蒙特卡洛次数
DYNAMIC_PERTURB_RATIO = 0.05   # 动态调度每日扰动比例
SEED = 20260814
FIGURES_DIR = os.path.join(_PROJECT_DIR, 'figures', 'Problem3', 'extra')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 工具函数 ====================
def copy_demands(demands):
    """深拷贝需求列表（仅拷贝会被扰动的时间字段）"""
    return [
        {
            'person_id': d['person_id'],
            'origin_id': d['origin_id'],
            'destination_id': d['destination_id'],
            'earliest_pickup': d['earliest_pickup'],
            'latest_arrival': d['latest_arrival'],
            'task_type': d['task_type'],
        }
        for d in demands
    ]


def perturb_time_window(d, mode, magnitude, rng):
    """
    对单条需求的时间窗施加随机扰动

    参数:
        d: 需求字典（原地修改 earliest_pickup / latest_arrival）
        mode: 'relative' 相对扰动、'absolute' 绝对扰动、
              'tighten' 单边收紧（最晚到达时间最多提前 magnitude 分钟）
        magnitude: 扰动量级
        rng: random.Random 实例
    """
    e = d['earliest_pickup']
    l = d['latest_arrival']
    width = (l - e).total_seconds() / 60.0

    if mode == 'relative':
        d_e = rng.uniform(-magnitude, magnitude) * max(width, 1.0)
        d_l = rng.uniform(-magnitude, magnitude) * max(width, 1.0)
        new_e = e + timedelta(minutes=d_e)
        new_l = l + timedelta(minutes=d_l)
    elif mode == 'tighten':
        # 任务紧急化：最晚到达时间提前，最早登机时刻保持不变
        shift = rng.uniform(0, magnitude)
        new_e = e
        new_l = l - timedelta(minutes=shift)
    else:
        d_e = rng.uniform(-magnitude, magnitude)
        d_l = rng.uniform(-magnitude, magnitude)
        new_e = e + timedelta(minutes=d_e)
        new_l = l + timedelta(minutes=d_l)

    # 扰动后时间窗若退化为空，保持至少 1 分钟有效宽度
    if new_l <= new_e:
        new_l = new_e + timedelta(minutes=1)

    d['earliest_pickup'] = new_e
    d['latest_arrival'] = new_l


def allocation_stability(alloc_a, alloc_b, person_ids):
    """计算两次分配中分配到相同日期的需求比例"""
    same = 0
    for pid in person_ids:
        if alloc_a.get(pid) == alloc_b.get(pid):
            same += 1
    return same / len(person_ids) if person_ids else 0.0


def load_std(daily_load, dates):
    """计算每日负载的标准差与变异系数"""
    values = [daily_load.get(d, 0) for d in dates]
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = var ** 0.5
    cv = std / mean if mean > 0 else 0.0
    return std, cv


# ==================== 基准 ====================
def baseline_statistics(demands, dates):
    """计算确定性模型的基准分配与统计量"""
    alloc, load = Q3DemandDayAllocator.allocate(demands, dates)
    std, cv = load_std(load, dates)
    feasible_counts = [
        len(Q3DemandDayAllocator.compute_feasible_days(d, dates)) for d in demands
    ]
    avg_feasible = sum(feasible_counts) / len(feasible_counts) if feasible_counts else 0
    return {
        'alloc': alloc,
        'load': load,
        'std': std,
        'cv': cv,
        'avg_feasible_days': avg_feasible,
    }


# ==================== 鲁棒性场景工作函数（供多进程调用） ====================
def _scenario_worker(args):
    """
    单个鲁棒性场景的蒙特卡洛工作函数（多进程并行单元）

    参数:
        args: (scenario, demands, dates, mc_runs, seed_offset)
    返回:
        dict: 场景汇总指标与每次运行的稳定率列表
    """
    scenario, demands, dates, mc_runs, seed_offset = args
    mode = scenario['mode']
    magnitude = scenario['magnitude']
    label = scenario['label']

    person_ids = [d['person_id'] for d in demands]
    base_alloc, _ = Q3DemandDayAllocator.allocate(demands, dates)

    stability_list = []
    std_list = []
    avg_feasible_list = []
    infeasible_list = []

    for run in range(mc_runs):
        rng = random.Random(seed_offset + run)
        perturbed = copy_demands(demands)
        for d in perturbed:
            perturb_time_window(d, mode, magnitude, rng)

        alloc, load = Q3DemandDayAllocator.allocate(perturbed, dates)

        stability_list.append(
            allocation_stability(base_alloc, alloc, person_ids)
        )

        std, _ = load_std(load, dates)
        std_list.append(std)

        feasible_counts = [
            len(Q3DemandDayAllocator.compute_feasible_days(d, dates))
            for d in perturbed
        ]
        avg_feasible_list.append(sum(feasible_counts) / len(feasible_counts))
        infeasible_list.append(sum(1 for c in feasible_counts if c == 0))

    # 计算稳定率的分布统计量（均值、标准差、分位数、偏度、峰度）
    s_sorted = sorted(stability_list)
    n = len(s_sorted)
    s_mean = sum(s_sorted) / n
    s_var = sum((x - s_mean) ** 2 for x in s_sorted) / n
    s_std = s_var ** 0.5

    def _quantile(p):
        """返回样本的 p 分位数（线性插值近似）"""
        idx = min(n - 1, max(0, int(p * n)))
        return s_sorted[idx]

    # 三阶/四阶中心矩用于偏度与超额峰度
    if s_std > 0:
        s_skew = (sum((x - s_mean) ** 3 for x in s_sorted) / n) / (s_std ** 3)
        s_kurt = (sum((x - s_mean) ** 4 for x in s_sorted) / n) / (s_std ** 4) - 3.0
    else:
        s_skew = 0.0
        s_kurt = 0.0

    return {
        'label': label,
        'mode': mode,
        'magnitude': magnitude,
        'stability_mean': s_mean,
        'stability_std': s_std,
        'stability_min': min(stability_list),
        'stability_q25': _quantile(0.25),
        'stability_median': _quantile(0.50),
        'stability_q75': _quantile(0.75),
        'stability_max': max(stability_list),
        'stability_skew': s_skew,
        'stability_kurt': s_kurt,
        'std_mean': sum(std_list) / len(std_list),
        'avg_feasible_mean': sum(avg_feasible_list) / len(avg_feasible_list),
        'infeasible_mean': sum(infeasible_list) / len(infeasible_list),
        'stability_list': stability_list,
    }


def run_robustness_experiment(demands, dates, scenarios, mc_runs, seed):
    """鲁棒性实验：多进程并行执行各场景蒙特卡洛扰动"""
    tasks = [
        (scenario, demands, dates, mc_runs, seed + idx * mc_runs)
        for idx, scenario in enumerate(scenarios)
    ]
    # Windows 下使用 spawn 启动，工作函数必须可被导入
    with mp.Pool(processes=min(len(scenarios), mp.cpu_count())) as pool:
        results = pool.map(_scenario_worker, tasks)
    return results


# ==================== 实验二：动态调度 ====================
def run_dynamic_experiment(demands, dates, perturb_ratio, seed):
    """
    动态调度实验：按 7 天滚动，每天随机扰动一部分未执行需求

    对比"静态不调整"与"滚动重分配"两种策略
    """
    rng = random.Random(seed)
    person_ids = [d['person_id'] for d in demands]
    demand_map = {d['person_id']: d for d in demands}

    baseline = baseline_statistics(demands, dates)
    current_alloc = dict(baseline['alloc'])
    current_load = dict(baseline['load'])
    perturbed_pids = set()

    total_affected = 0
    total_static_infeasible = 0
    total_adjusted = 0
    daily_log = []

    for day in dates:
        pending = [
            pid for pid in person_ids
            if current_alloc[pid] >= day and pid not in perturbed_pids
        ]
        if not pending:
            daily_log.append({
                'day': day, 'affected': 0,
                'static_infeasible': 0, 'adjusted': 0,
            })
            continue

        n_affected = max(1, int(len(pending) * perturb_ratio))
        affected = rng.sample(pending, n_affected)

        n_static_infeasible = 0
        n_adjusted = 0

        for pid in affected:
            perturbed_pids.add(pid)
            d = demand_map[pid]
            # 模拟突发任务紧急化：最晚到达时间提前 18~36 小时
            tighten_minutes = rng.uniform(1080, 2160)
            perturb_time_window(d, 'tighten', tighten_minutes, rng)

            old_day = current_alloc[pid]
            feasible = Q3DemandDayAllocator.compute_feasible_days(d, dates)

            if old_day not in feasible:
                n_static_infeasible += 1
                n_adjusted += 1
                if feasible:
                    new_day = min(feasible, key=lambda dd: current_load.get(dd, 0))
                else:
                    new_day = old_day
                current_alloc[pid] = new_day

        current_load = defaultdict(int)
        for pid in person_ids:
            current_load[current_alloc[pid]] += 1

        total_affected += n_affected
        total_static_infeasible += n_static_infeasible
        total_adjusted += n_adjusted
        daily_log.append({
            'day': day,
            'affected': n_affected,
            'static_infeasible': n_static_infeasible,
            'adjusted': n_adjusted,
        })

    final_std, final_cv = load_std(current_load, dates)
    final_stability = allocation_stability(baseline['alloc'], current_alloc, person_ids)

    return {
        'daily_log': daily_log,
        'total_affected': total_affected,
        'total_static_infeasible': total_static_infeasible,
        'total_adjusted': total_adjusted,
        'static_infeasible_rate': (
            total_static_infeasible / total_affected if total_affected else 0.0
        ),
        'adjust_rate': total_adjusted / total_affected if total_affected else 0.0,
        'final_std': final_std,
        'final_cv': final_cv,
        'final_stability': final_stability,
        'final_load': dict(current_load),
        'baseline_load': dict(baseline['load']),
    }


# ==================== 可视化 ====================
def plot_robustness_figures(baseline, robust_results):
    """绘制鲁棒性实验图：稳定率柱状图（含误差棒）与箱线图"""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    labels = [r['label'] for r in robust_results]
    means = [r['stability_mean'] * 100 for r in robust_results]
    stds = [r['stability_std'] * 100 for r in robust_results]

    # 图1：平均分配稳定率柱状图（误差棒表示蒙特卡洛标准差）
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, means, yerr=stds, capsize=4, color='#2E86AB',
                  edgecolor='black', alpha=0.85)
    ax.set_ylim(0, 110)
    ax.set_ylabel('平均分配稳定率 (%)')
    ax.set_xlabel('扰动场景')
    ax.set_title('鲁棒性实验：各场景分配稳定率（2000 次蒙特卡洛）')
    ax.grid(axis='y', alpha=0.3)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 1,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'R1_robust_stability_bars.png'), dpi=150)
    plt.close(fig)

    # 图2：稳定率分布箱线图
    fig, ax = plt.subplots(figsize=(10, 6))
    box_data = [r['stability_list'] for r in robust_results]
    ax.boxplot(box_data, labels=labels, patch_artist=True,
               boxprops=dict(facecolor='#A23B72', alpha=0.6))
    ax.set_ylabel('分配稳定率')
    ax.set_xlabel('扰动场景')
    ax.set_title('鲁棒性实验：稳定率分布箱线图（2000 次蒙特卡洛）')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'R2_robust_stability_boxplot.png'), dpi=150)
    plt.close(fig)

    # 图5：稳定率分布直方图（2×3 子图），观察各场景的离散形态
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for idx, r in enumerate(robust_results):
        ax = axes[idx]
        vals = [v * 100 for v in r['stability_list']]
        ax.hist(vals, bins=40, color='#2E86AB', edgecolor='black', alpha=0.75)
        ax.axvline(r['stability_mean'] * 100, color='red', linestyle='--', linewidth=1.5)
        ax.set_title(r['label'])
        ax.set_xlabel('分配稳定率 (%)')
        ax.set_ylabel('频次')
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('鲁棒性实验：各场景稳定率分布直方图（5000 次蒙特卡洛）',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIGURES_DIR, 'R5_robust_stability_histograms.png'), dpi=150)
    plt.close(fig)


def plot_dynamic_figures(dynamic_results):
    """绘制动态调度实验图：逐日调整柱状图与负载均衡前后对比"""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    daily = dynamic_results['daily_log']
    days = [str(d['day'])[-5:] for d in daily]
    affected = [d['affected'] for d in daily]
    static_infeasible = [d['static_infeasible'] for d in daily]
    adjusted = [d['adjusted'] for d in daily]

    # 图3：逐日受影响/静态不可行/需调整分组柱状图
    fig, ax = plt.subplots(figsize=(11, 6))
    x = list(range(len(days)))
    width = 0.25
    ax.bar([i - width for i in x], affected, width, label='受影响需求',
           color='#2E86AB', edgecolor='black')
    ax.bar(x, static_infeasible, width, label='静态不可行', color='#C73E1D',
           edgecolor='black')
    ax.bar([i + width for i in x], adjusted, width, label='需调整', color='#F18F01',
           edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(days)
    ax.set_xlabel('日期')
    ax.set_ylabel('需求数量')
    ax.set_title('动态调度：逐日突发需求调整情况')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'R3_dynamic_daily_adjustments.png'), dpi=150)
    plt.close(fig)

    # 图4：负载均衡前后对比折线图
    dates = sorted(dynamic_results['baseline_load'].keys())
    base_load = [dynamic_results['baseline_load'][d] for d in dates]
    final_load = [dynamic_results['final_load'][d] for d in dates]
    day_labels = [str(d)[-5:] for d in dates]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(day_labels, base_load, 'o-', label='基准（静态一次性）',
            color='#2E86AB', linewidth=2)
    ax.plot(day_labels, final_load, 's--', label='动态滚动重优化后',
            color='#C73E1D', linewidth=2)
    ax.set_xlabel('日期')
    ax.set_ylabel('每日需求负载（人）')
    ax.set_title('动态调度前后每日负载对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'R4_dynamic_load_comparison.png'), dpi=150)
    plt.close(fig)


# ==================== 结果输出 ====================
def write_results(path, baseline, robust_results, dynamic_results):
    """将实验结果写入 txt 文件"""
    lines = []
    lines.append('=' * 70)
    lines.append('模型推广实验：鲁棒优化框架与动态调度机制')
    lines.append('=' * 70)
    lines.append('生成时间: 2026-08-14')
    lines.append('数据: peopleQ3.csv (4000 条需求, 7 天规划期)')
    lines.append('分配器: Q3DemandDayAllocator (与问题三求解器一致)')
    lines.append(f'蒙特卡洛次数: 每个场景 {MC_RUNS} 次')
    lines.append('')

    lines.append('-' * 70)
    lines.append('一、确定性模型基准')
    lines.append('-' * 70)
    lines.append(f'平均可行日期数: {baseline["avg_feasible_days"]:.2f} 天')
    lines.append(f'每日负载: { {str(k): v for k, v in baseline["load"].items()} }')
    lines.append(f'每日负载标准差: {baseline["std"]:.1f}')
    lines.append(f'每日负载变异系数: {baseline["cv"]:.4f}')
    lines.append('')

    lines.append('-' * 70)
    lines.append('二、鲁棒性实验（时间窗扰动下的分配稳定性）')
    lines.append('-' * 70)
    lines.append(f'每个场景 {MC_RUNS} 次蒙特卡洛扰动，指标取平均值')
    lines.append('')
    header = (
        f'{"场景":<16}{"稳定率均值":>12}{"稳定率std":>12}'
        f'{"负载标准差":>12}{"平均可行天数":>14}{"不可行需求":>12}'
    )
    lines.append(header)
    lines.append('-' * 74)
    for r in robust_results:
        lines.append(
            f'{r["label"]:<16}{r["stability_mean"] * 100:>11.1f}%'
            f'{r["stability_std"] * 100:>12.2f}%'
            f'{r["std_mean"]:>12.1f}{r["avg_feasible_mean"]:>14.2f}'
            f'{r["infeasible_mean"]:>12.1f}'
        )
    lines.append('')
    lines.append('稳定率分布明细（按样本升序统计）:')
    lines.append(
        f'{"场景":<16}{"最小值":>10}{"25分位":>10}{"中位数":>10}'
        f'{"75分位":>10}{"最大值":>10}{"偏度":>8}{"峰度":>8}'
    )
    lines.append('-' * 74)
    for r in robust_results:
        lines.append(
            f'{r["label"]:<16}{r["stability_min"] * 100:>9.1f}%'
            f'{r["stability_q25"] * 100:>10.1f}%'
            f'{r["stability_median"] * 100:>10.1f}%'
            f'{r["stability_q75"] * 100:>10.1f}%'
            f'{r["stability_max"] * 100:>10.1f}%'
            f'{r["stability_skew"]:>8.2f}'
            f'{r["stability_kurt"]:>8.2f}'
        )
    lines.append('')
    lines.append('注: 分配稳定率 = 扰动后分配到相同日期的需求比例；')
    lines.append('    不可行需求数 = 扰动后无任何可行日期的需求数量（均值）。')
    lines.append('')

    lines.append('-' * 70)
    lines.append('三、动态调度实验（滚动时域重优化）')
    lines.append('-' * 70)
    lines.append(f'每日扰动比例: {DYNAMIC_PERTURB_RATIO * 100:.0f}% 的未执行需求')
    lines.append('扰动方式: 任务紧急化（最晚到达时间提前 18~36 小时）')
    lines.append('')
    lines.append('逐日滚动日志:')
    lines.append(f'{"日期":<12}{"受影响需求":>12}{"静态不可行":>14}{"需调整":>10}')
    lines.append('-' * 48)
    for log in dynamic_results['daily_log']:
        lines.append(
            f'{str(log["day"]):<12}{log["affected"]:>12}'
            f'{log["static_infeasible"]:>14}{log["adjusted"]:>10}'
        )
    lines.append('')
    lines.append(f'累计受影响需求: {dynamic_results["total_affected"]}')
    lines.append(
        f'静态不调整的不可行率: '
        f'{dynamic_results["static_infeasible_rate"] * 100:.1f}%'
    )
    lines.append(f'滚动重分配调整率: {dynamic_results["adjust_rate"] * 100:.1f}%')
    lines.append(
        f'最终分配相对基准的稳定率: '
        f'{dynamic_results["final_stability"] * 100:.1f}%'
    )
    lines.append(f'最终每日负载标准差: {dynamic_results["final_std"]:.1f}')
    lines.append(f'最终每日负载变异系数: {dynamic_results["final_cv"]:.4f}')
    lines.append(
        f'最终每日负载: '
        f'{ {str(k): v for k, v in dynamic_results["final_load"].items()} }'
    )
    lines.append('')

    lines.append('=' * 70)

    content = '\n'.join(lines) + '\n'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(content)


# ==================== 主函数 ====================
def main():
    """运行鲁棒性与动态调度实验、输出结果并绘图"""
    print('加载需求数据...')
    loader = Q3DataLoader()
    demands = loader.load_demands_q3()
    print(f'加载 {len(demands)} 条需求')

    dates = PLANNING_DATES

    scenarios = [
        {'label': '相对 ±10%', 'mode': 'relative', 'magnitude': 0.10},
        {'label': '相对 ±20%', 'mode': 'relative', 'magnitude': 0.20},
        {'label': '绝对 ±60min', 'mode': 'absolute', 'magnitude': 60},
        {'label': '绝对 ±120min', 'mode': 'absolute', 'magnitude': 120},
        {'label': '紧急化 ≤12h', 'mode': 'tighten', 'magnitude': 720},
        {'label': '紧急化 ≤24h', 'mode': 'tighten', 'magnitude': 1440},
    ]
    print(f'运行鲁棒性实验（{len(scenarios)} 个场景 × {MC_RUNS} 次蒙特卡洛，多进程并行）...')
    baseline = baseline_statistics(demands, dates)
    robust_results = run_robustness_experiment(
        demands, dates, scenarios, MC_RUNS, SEED
    )

    print('运行动态调度实验（7 天滚动）...')
    dynamic_results = run_dynamic_experiment(
        demands, dates, DYNAMIC_PERTURB_RATIO, SEED
    )

    out_path = os.path.normpath(
        os.path.join(_PROJECT_DIR, 'output', 'Problem3', 'robust_dynamic_results.txt')
    )
    write_results(out_path, baseline, robust_results, dynamic_results)

    print('绘制图表...')
    plot_robustness_figures(baseline, robust_results)
    plot_dynamic_figures(dynamic_results)

    print(f'\n结果已写入: {out_path}')
    print(f'图表已写入: {FIGURES_DIR}')


if __name__ == '__main__':
    # Windows 多进程必须由入口保护
    mp.freeze_support()
    main()
