# -*- coding: utf-8 -*-
"""
文件: generate_q3_figures.py
功能: 生成问题3的5张图表 (F1/F2/F3/F4/F6)
项目: 2026策联杯 B题
输出: figures/Problem3/ 目录下的 PNG 文件
"""

import os
import sys
import csv
from datetime import datetime
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter, date2num
import numpy as np

# ==================== 路径与字体配置 ====================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
_FIG_DIR = os.path.join(_PROJECT_DIR, 'figures', 'Problem3')
os.makedirs(_FIG_DIR, exist_ok=True)

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# 统一配色
COLOR_PRIMARY = '#2C5F8D'
COLOR_ACCENT = '#E8833A'
COLOR_GREEN = '#5B9F5B'
COLOR_RED = '#C0504D'
COLOR_GRAY = '#A0A0A0'
COLORS_MULTI = ['#2C5F8D', '#E8833A', '#5B9F5B', '#C0504D', '#8064A7']


# ==================== F1: 每日指标对比图 ====================
def generate_f1():
    """生成每日指标对比图：7天×4子图（架次、时间、利用率、人/架次）"""

    dates = ['08-03', '08-04', '08-05', '08-06', '08-07', '08-08', '08-09']
    flights = [20, 23, 20, 18, 24, 24, 21]
    time_min = [3446, 4327, 3607, 3014, 4530, 4403, 3657]
    util = [49.68, 25.56, 27.41, 26.48, 27.70, 19.44, 18.85]
    per_flight = [27.6, 26.2, 25.0, 28.1, 27.3, 23.8, 26.2]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('问题三：各日求解指标对比', fontsize=14, fontweight='bold')

    x = np.arange(len(dates))

    # 子图1: 架次数
    ax = axes[0, 0]
    bars = ax.bar(x, flights, color=COLOR_PRIMARY, alpha=0.85, width=0.6)
    ax.set_ylabel('架次数', fontsize=11)
    ax.set_title('(a) 每日架次数', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(dates, fontsize=9)
    for bar, val in zip(bars, flights):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(val), ha='center', va='bottom', fontsize=9)

    # 子图2: 总时间
    ax = axes[0, 1]
    bars = ax.bar(x, time_min, color=COLOR_ACCENT, alpha=0.85, width=0.6)
    ax.set_ylabel('总飞机使用时间 (min)', fontsize=11)
    ax.set_title('(b) 每日总飞机使用时间', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(dates, fontsize=9)
    for bar, val in zip(bars, time_min):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                str(val), ha='center', va='bottom', fontsize=9)

    # 子图3: 利用率
    ax = axes[1, 0]
    bars = ax.bar(x, util, color=COLOR_GREEN, alpha=0.85, width=0.6)
    ax.set_ylabel('座位利用率 (%)', fontsize=11)
    ax.set_title('(c) 每日座位利用率', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(dates, fontsize=9)
    ax.set_ylim(0, 60)
    for bar, val in zip(bars, util):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    # 子图4: 人/架次
    ax = axes[1, 1]
    bars = ax.bar(x, per_flight, color=COLOR_RED, alpha=0.85, width=0.6)
    ax.set_ylabel('人均架次承载 (人/架次)', fontsize=11)
    ax.set_title('(d) 每日人均架次承载', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(dates, fontsize=9)
    for bar, val in zip(bars, per_flight):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = os.path.join(_FIG_DIR, 'q3_daily_metrics.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'[F1] 已保存: {path}')


# ==================== F2: Pareto前沿散点图 ====================
def generate_f2():
    """生成Pareto前沿散点图：7组权重方案"""

    # 从 pareto_sweep_results.csv 读取数据
    csv_path = os.path.join(_PROJECT_DIR, 'output', 'Problem3', 'pareto_sweep_results.csv')
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'label': row['方案'],
                'time': float(row['总时间(min)']),
                'util': float(row['利用率(%)']),
                'flights': int(row['架次']),
                'fuel': float(row['燃油(kg)']),
                'pareto': row['Pareto最优'] == '是',
            })

    fig, ax = plt.subplots(figsize=(9, 7))

    for d in data:
        if d['pareto']:
            ax.scatter(d['time'], d['util'], c=COLOR_RED, s=200, zorder=5,
                       marker='*', edgecolors='black', linewidths=1.2)
            ax.annotate(f"{d['label']}\n(最优)", (d['time'], d['util']),
                        textcoords="offset points", xytext=(15, -15),
                        fontsize=10, fontweight='bold', color=COLOR_RED,
                        arrowprops=dict(arrowstyle='->', color=COLOR_RED, lw=1.5))
        else:
            ax.scatter(d['time'], d['util'], c=COLOR_PRIMARY, s=120, zorder=4,
                       edgecolors='gray', linewidths=0.8)
            ax.annotate(d['label'], (d['time'], d['util']),
                        textcoords="offset points", xytext=(8, 5),
                        fontsize=8, color='dimgray')

    # 画Pareto前沿连接线（只有1个点，用虚线标注区域）
    pareto_points = [(d['time'], d['util']) for d in data if d['pareto']]
    if pareto_points:
        px, py = pareto_points[0]
        ax.axvline(x=px, color=COLOR_RED, linestyle='--', alpha=0.3, lw=1)
        ax.axhline(y=py, color=COLOR_RED, linestyle='--', alpha=0.3, lw=1)

    ax.set_xlabel('总飞机使用时间 (min)', fontsize=12)
    ax.set_ylabel('座位利用率 (%)', fontsize=12)
    ax.set_title('问题三：Pareto前沿灵敏度分析 (2026-08-03单日)', fontsize=13, fontweight='bold')

    # 图例
    star_patch = mpatches.Patch(color=COLOR_RED, label='Pareto最优解')
    dot_patch = mpatches.Patch(color=COLOR_PRIMARY, label='被支配解')
    ax.legend(handles=[star_patch, dot_patch], loc='lower right', fontsize=10)

    ax.grid(True, alpha=0.3)
    ax.set_xlim(6750, 7250)
    ax.set_ylim(39, 44)

    plt.tight_layout()
    path = os.path.join(_FIG_DIR, 'q3_pareto_frontier.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'[F2] 已保存: {path}')


# ==================== F3: 单日排班甘特图 ====================
def generate_f3():
    """生成单日排班甘特图：08-03首日 A01机场"""

    routes_path = os.path.join(_PROJECT_DIR, 'output', 'Problem3', 'q3-routes.csv')

    # 解析 routes.csv，提取每个架次的起飞和返回时间
    flights_by_aircraft = defaultdict(list)
    with open(routes_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            aircraft_id = row['aircraft_id']
            # 只取08-03的A01机场飞机
            if not aircraft_id.startswith('A01'):
                continue
            dep = row['departure_time']
            arr = row['arrival_time']
            if not dep and not arr:
                continue
            flights_by_aircraft[aircraft_id].append({
                'flight_no': int(row['flight_no']),
                'stop_order': int(row['stop_order']),
                'facility': row['facility_id'],
                'dep': dep,
                'arr': arr,
            })

    # 对每架飞机，按flight_no和stop_order分组，提取每个架次的起止时间
    aircraft_schedules = {}
    for ac_id, stops in flights_by_aircraft.items():
        # 按flight_no分组
        flight_groups = defaultdict(list)
        for s in stops:
            flight_groups[s['flight_no']].append(s)

        schedule = []
        for fno, group in sorted(flight_groups.items()):
            group.sort(key=lambda x: x['stop_order'])
            # 起飞时间 = stop_order=0 的 departure_time
            dep_time = None
            arr_time = None
            facilities = []
            for s in group:
                if s['stop_order'] == 0 and s['dep']:
                    dep_time = datetime.strptime(s['dep'], '%Y-%m-%d %H:%M')
                if s['arr']:
                    arr_time = datetime.strptime(s['arr'], '%Y-%m-%d %H:%M')
                if not s['facility'].startswith('A0'):
                    facilities.append(s['facility'])

            # 只保留08-03的架次
            if dep_time and dep_time.strftime('%Y-%m-%d') == '2026-08-03':
                schedule.append({
                    'flight_no': fno,
                    'start': dep_time,
                    'end': arr_time,
                    'facilities': facilities,
                })
        if schedule:
            aircraft_schedules[ac_id] = schedule

    # 按机型排序
    sorted_aircraft = sorted(aircraft_schedules.keys(),
                             key=lambda x: (x.split('-')[1], x.split('-')[2]))

    fig, ax = plt.subplots(figsize=(14, max(4, len(sorted_aircraft) * 0.5)))

    bar_height = 0.6
    colors_by_type = {'T1': '#2C5F8D', 'T2': '#E8833A', 'T3': '#5B9F5B'}

    for i, ac_id in enumerate(sorted_aircraft):
        atype = ac_id.split('-')[1]
        color = colors_by_type.get(atype, COLOR_GRAY)
        for flight in aircraft_schedules[ac_id]:
            start_num = date2num(flight['start'])
            end_num = date2num(flight['end'])
            width = end_num - start_num
            ax.barh(i, width, left=start_num, height=bar_height,
                    color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
            # 在色块中间标注架次号
            mid = start_num + width / 2
            ax.text(mid, i, f"F{flight['flight_no']}", ha='center', va='center',
                    fontsize=7, color='white', fontweight='bold')

    # Y轴标签
    ax.set_yticks(range(len(sorted_aircraft)))
    ax.set_yticklabels(sorted_aircraft, fontsize=9)

    # X轴时间格式
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax.set_xlim(date2num(datetime(2026, 8, 3, 5, 30)),
                date2num(datetime(2026, 8, 3, 20, 30)))

    ax.set_xlabel('时间 (2026-08-03)', fontsize=12)
    ax.set_ylabel('飞机编号', fontsize=12)
    ax.set_title('问题三：A01机场单日排班甘特图 (2026-08-03)', fontsize=13, fontweight='bold')

    # 图例
    patches = [mpatches.Patch(color=c, label=f'{t} ({t_info})')
               for t, (c, t_info) in {
                   'T1': ('#2C5F8D', '12座'),
                   'T2': ('#E8833A', '16座'),
                   'T3': ('#5B9F5B', '19座'),
               }.items()]
    ax.legend(handles=patches, loc='upper right', fontsize=9)

    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(_FIG_DIR, 'q3_gantt_a01_0803.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'[F3] 已保存: {path}')


# ==================== F4: Q1/Q2/Q3指标对比雷达图 ====================
def generate_f4():
    """生成Q1/Q2/Q3五项指标对比雷达图（归一化）"""

    # 原始数据
    metrics = {
        '问题一': {
            '总时间(h)': 249.85,
            '在途时间(h)': 2141.58,
            '架次数': 90,
            '燃油(t)': 121.181,
            '利用率(%)': 48.96,
        },
        '问题二': {
            '总时间(h)': 294.48,
            '在途时间(h)': 4427.97,
            '架次数': 95,
            '燃油(t)': 137.236,
            '利用率(%)': 89.98,
        },
        '问题三': {
            '总时间(h)': 449.73,
            '在途时间(h)': 4460.18,
            '架次数': 150,
            '燃油(t)': 209.528,
            '利用率(%)': 63.78,
        },
    }

    labels = list(metrics['问题一'].keys())
    n = len(labels)

    # 归一化（每个指标除以最大值）
    max_vals = {k: max(metrics[q][k] for q in metrics) for k in labels}
    norm = {}
    for q in metrics:
        norm[q] = [metrics[q][k] / max_vals[k] for k in labels]

    # 雷达图
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = [COLOR_PRIMARY, COLOR_ACCENT, COLOR_GREEN]
    for idx, (q, vals) in enumerate(norm.items()):
        vals_closed = vals + vals[:1]
        ax.plot(angles, vals_closed, 'o-', linewidth=2, color=colors[idx], label=q)
        ax.fill(angles, vals_closed, alpha=0.15, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8)
    ax.set_title('问题一/二/三 指标对比（归一化）', fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.tight_layout()
    path = os.path.join(_FIG_DIR, 'q3_q1q2q3_comparison.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'[F4] 已保存: {path}')


# ==================== F6: C7/C8对比实验图 ====================
def generate_f6():
    """生成C7/C8对比实验图：归一化对比（基线=100%），6个子图"""

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle('问题三：创新点对比实验 (C7时间窗聚类 / C8反应式列生成)',
                 fontsize=13, fontweight='bold')

    # C7 数据
    c7_metrics = ['架次', '总时间(min)', '利用率(%)']
    c7_baseline = [32, 5927, 49.68]
    c7_clustering = [40, 7282, 41.37]

    # C8 数据
    c8_metrics = ['架次', '在途时间(min)', '利用率(%)']
    c8_initial = [38, 43564, 43.12]
    c8_enhanced = [40, 41666, 42.75]

    # --- 上排: C7 归一化对比 (基线=100%) ---
    for j, (metric, base_val, exp_val) in enumerate(zip(c7_metrics, c7_baseline, c7_clustering)):
        ax = axes[0, j]
        norm_exp = exp_val / base_val * 100
        bars = ax.bar(['基线', 'C7聚类'], [100, norm_exp],
                       color=[COLOR_PRIMARY, COLOR_RED], alpha=0.85, width=0.5)
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_ylabel('归一化值 (基线=100)', fontsize=9)
        ax.set_title(f'C7-{metric}', fontsize=10)
        # 标注原始数值
        ax.text(0, 102, f'{base_val}', ha='center', va='bottom', fontsize=8, color=COLOR_PRIMARY)
        ax.text(1, norm_exp + 2, f'{exp_val}', ha='center', va='bottom', fontsize=8, color=COLOR_RED)
        # 变化幅度
        delta = norm_exp - 100
        color = COLOR_RED if delta > 0 else COLOR_GREEN
        if metric == '利用率(%)':
            color = COLOR_RED if delta < 0 else COLOR_GREEN
        ax.text(1, norm_exp / 2, f'{delta:+.1f}%', ha='center', va='center',
                fontsize=11, fontweight='bold', color=color)
        ax.set_ylim(0, max(130, norm_exp + 15))

    # --- 下排: C8 归一化对比 (基线=100%) ---
    for j, (metric, base_val, exp_val) in enumerate(zip(c8_metrics, c8_initial, c8_enhanced)):
        ax = axes[1, j]
        norm_exp = exp_val / base_val * 100
        bars = ax.bar(['初始池', 'C8增强'], [100, norm_exp],
                       color=[COLOR_PRIMARY, COLOR_GREEN], alpha=0.85, width=0.5)
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_ylabel('归一化值 (基线=100)', fontsize=9)
        ax.set_title(f'C8-{metric}', fontsize=10)
        # 标注原始数值
        ax.text(0, 102, f'{base_val}', ha='center', va='bottom', fontsize=8, color=COLOR_PRIMARY)
        ax.text(1, norm_exp + 2, f'{exp_val}', ha='center', va='bottom', fontsize=8, color=COLOR_GREEN)
        # 变化幅度
        delta = norm_exp - 100
        color = COLOR_RED if delta > 0 else COLOR_GREEN
        if metric == '在途时间(min)':
            color = COLOR_GREEN if delta < 0 else COLOR_RED
        elif metric == '架次':
            color = COLOR_RED if delta > 0 else COLOR_GREEN
        ax.text(1, norm_exp / 2, f'{delta:+.1f}%', ha='center', va='center',
                fontsize=11, fontweight='bold', color=color)
        ax.set_ylim(0, max(115, norm_exp + 10))

    plt.tight_layout()
    path = os.path.join(_FIG_DIR, 'q3_c7_c8_comparison.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'[F6] 已保存: {path}')


# ==================== 主函数 ====================
def main():
    """生成全部5张图表"""
    print('=' * 60)
    print('生成问题3图表 (F1/F2/F3/F4/F6)')
    print('=' * 60)

    generate_f1()
    generate_f2()
    generate_f3()
    generate_f4()
    generate_f6()

    print(f'\n全部图表已保存至: {_FIG_DIR}')
    print('(F5 拉格朗日收敛曲线需单独运行 lagrangian_bound.py)')


if __name__ == '__main__':
    main()

