# -*- coding: utf-8 -*-
"""
文件: gen_q1q2_solution_figures.py
功能: 从 q1/q2 的 routes.csv 和 assignments.csv 读取真实求解数据，
      生成两张科学图表用于论文问题一、问题二求解结果小节。
      图1: 问题一架次特征分布三联图（机型占比 + 停靠次数 + 加油需求）
      图2: 问题二联合运输结构对比图（机型架次+复用倍数 + 三类需求运量结构）
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 中文字体设置
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 11

# 路径配置
BASE = r'd:\数模\26策联杯\8.13MM_STJU'
OUT_DIR = os.path.join(BASE, 'figures', 'Problem1')
OUT_DIR2 = os.path.join(BASE, 'figures', 'Problem2')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_DIR2, exist_ok=True)

# 机型颜色
COLOR_T1 = '#4ECDC4'
COLOR_T2 = '#45B7D1'
COLOR_T3 = '#FF6B6B'
TYPE_COLORS = {'T1': COLOR_T1, 'T2': COLOR_T2, 'T3': COLOR_T3}


def load_routes(path):
    """加载routes.csv，返回按架次分组的数据"""
    df = pd.read_csv(path)
    # 每个架次的机场是 stop_order=0 和 最后一个 stop_order
    grouped = df.groupby(['aircraft_type', 'flight_no'])
    results = []
    for (atype, fno), grp in grouped:
        grp_sorted = grp.sort_values('stop_order')
        facilities = grp_sorted['facility_id'].tolist()
        # 去掉起降机场（A开头），统计海上设施停靠次数
        sea_stops = [f for f in facilities if not f.startswith('A')]
        n_stops = len(sea_stops)
        refuel_count = grp_sorted['refuel'].sum()
        results.append({
            'aircraft_type': atype,
            'flight_no': fno,
            'n_sea_stops': n_stops,
            'refuel_count': refuel_count,
            'n_records': len(grp_sorted),
        })
    return pd.DataFrame(results)


def load_assignments(path):
    """加载assignments.csv"""
    return pd.read_csv(path)


def load_people_q2(path):
    """加载peopleQ2.csv，标记需求类型"""
    df = pd.read_csv(path)
    def classify(row):
        o, d = row['origin_id'], row['destination_id']
        if o == 'LAND' and d != 'LAND':
            return '出海'
        elif o != 'LAND' and d == 'LAND':
            return '海返'
        else:
            return '穿梭'
    df['demand_type'] = df.apply(classify, axis=1)
    return df


# ================================================================
# 图1: 问题一架次特征分布三联图
# ================================================================
def plot_q1_overview():
    """绘制问题一架次特征分布三联图"""
    routes = load_routes(os.path.join(BASE, 'output', 'Problem1', 'q1-routes.csv'))
    assigns = load_assignments(os.path.join(BASE, 'output', 'Problem1', 'q1-assignments.csv'))

    # 统计机型使用
    type_counts = routes['aircraft_type'].value_counts().sort_index()
    # 统计停靠次数分布
    stop_counts = routes['n_sea_stops'].value_counts().sort_index()
    # 统计加油需求
    refuel_routes = routes[routes['refuel_count'] > 0]
    n_refuel = len(refuel_routes)
    n_no_refuel = len(routes) - n_refuel

    # 统计每架次运量
    flight_load = assigns.groupby(['aircraft_type', 'flight_no']).size().reset_index(name='load')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # --- 子图1: 机型使用占比环形图 ---
    ax1 = axes[0]
    labels = [f'{t}' for t in type_counts.index]
    sizes = type_counts.values
    colors = [TYPE_COLORS.get(t, '#999') for t in type_counts.index]
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, pctdistance=0.78,
        wedgeprops=dict(width=0.42, edgecolor='white', linewidth=2)
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight('bold')
    ax1.set_title('(a) 机型使用占比', fontsize=12, fontweight='bold', pad=10)
    # 中心标注总架次
    ax1.text(0, 0, f'{len(routes)}\n架次', ha='center', va='center',
             fontsize=13, fontweight='bold', color='#333')

    # --- 子图2: 停靠次数分布柱状图 ---
    ax2 = axes[1]
    x_labels = [f'{int(s)}站' for s in stop_counts.index]
    bars = ax2.bar(x_labels, stop_counts.values, color='#5B9BD5', edgecolor='white', linewidth=1.2)
    # 在柱上标注数量和百分比
    total = stop_counts.sum()
    for bar, val in zip(bars, stop_counts.values):
        pct = val / total * 100
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
    ax2.set_ylabel('架次数', fontsize=11)
    ax2.set_title('(b) 海上设施停靠次数分布', fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylim(0, max(stop_counts.values) * 1.25)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # --- 子图3: 加油需求占比环形图 ---
    ax3 = axes[2]
    labels3 = ['无需加油', '需中途加油']
    sizes3 = [n_no_refuel, n_refuel]
    colors3 = ['#52C41A', '#FA8C16']
    wedges3, texts3, autotexts3 = ax3.pie(
        sizes3, labels=labels3, colors=colors3, autopct='%1.1f%%',
        startangle=90, pctdistance=0.78,
        wedgeprops=dict(width=0.42, edgecolor='white', linewidth=2)
    )
    for t in autotexts3:
        t.set_fontsize(10)
        t.set_fontweight('bold')
    ax3.set_title('(c) 加油需求分布', fontsize=12, fontweight='bold', pad=10)
    ax3.text(0, 0, f'{n_refuel}\n需加油', ha='center', va='center',
             fontsize=12, fontweight='bold', color='#333')

    plt.tight_layout(pad=1.2)
    out_path = os.path.join(OUT_DIR, 'q1_solution_overview.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'图1已保存: {out_path}')

    # 打印统计信息供核对
    print(f'\n问题一统计:')
    print(f'  机型分布: {dict(type_counts)}')
    print(f'  停靠次数: {dict(stop_counts)}')
    print(f'  加油: 需{n_refuel}架次, 无需{n_no_refuel}架次')


# ================================================================
# 图2: 问题二联合运输结构对比图
# ================================================================
def plot_q2_overview():
    """绘制问题二联合运输结构对比图"""
    routes = load_routes(os.path.join(BASE, 'output', 'Problem2', 'q2-routes.csv'))
    assigns = load_assignments(os.path.join(BASE, 'output', 'Problem2', 'q2-assignments.csv'))
    people = load_people_q2(os.path.join(
        BASE, '题目和思路', '2026年度“策联杯”数学建模精英联赛-B题-附件', 'peopleQ2.csv'))

    # 统计机型使用
    type_counts = routes['aircraft_type'].value_counts().sort_index()
    # 统计每架次运量
    flight_load = assigns.groupby(['aircraft_type', 'flight_no']).size().reset_index(name='load')
    # 各机型平均载客量与座位数
    seat_map = {'T1': 12, 'T2': 16, 'T3': 19}
    type_load_stats = flight_load.groupby('aircraft_type')['load'].agg(['mean', 'max']).reindex(
        ['T1', 'T2', 'T3']).dropna(how='all')

    # 三类需求运量结构（按person_id关联需求类型）
    assigns_merged = assigns.merge(people[['person_id', 'demand_type']], on='person_id', how='left')
    demand_by_type = assigns_merged['demand_type'].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # 问题二专用配色：T3 改用黄色
    Q2_TYPE_COLORS = {'T1': COLOR_T1, 'T2': COLOR_T2, 'T3': '#FFB300'}

    # --- 子图1: 各机型架次占比 + 平均载客量对比 ---
    ax1 = axes[0]
    types_order = ['T1', 'T2', 'T3']
    types_present = [t for t in types_order if t in type_counts.index]
    x = np.arange(len(types_present))
    width = 0.5

    # 左轴: 架次数柱状图（居中对齐 x 刻度）
    flight_nums = [type_counts.get(t, 0) for t in types_present]
    bars1 = ax1.bar(x, flight_nums, width, color=[Q2_TYPE_COLORS[t] for t in types_present],
                    edgecolor='white', linewidth=1.5, label='架次数')
    ax1.set_ylabel('架次数', fontsize=11, color='#333')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{t}\n({seat_map[t]}座)' for t in types_present])
    ax1.set_ylim(0, max(flight_nums) * 1.35)
    for bar, val in zip(bars1, flight_nums):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 右轴: 平均载客量折线
    ax1r = ax1.twinx()
    avg_loads = [type_load_stats.loc[t, 'mean'] if t in type_load_stats.index else 0 for t in types_present]
    ax1r.plot(x, avg_loads, 'o-', color='#E74C3C', linewidth=2, markersize=7, label='平均载客量')
    ax1r.set_ylabel('平均载客量（人/架次）', fontsize=11, color='#E74C3C')
    ax1r.set_ylim(0, max(avg_loads) * 1.4 if avg_loads else 10)
    ax1r.tick_params(axis='y', labelcolor='#E74C3C')
    # 标注复用倍数
    for i, (t, avg) in enumerate(zip(types_present, avg_loads)):
        if avg > 0:
            reuse = avg / seat_map[t]
            ax1r.annotate(f'{reuse:.1f}x', (x[i], avg), textcoords='offset points',
                          xytext=(8, 8), fontsize=9, color='#E74C3C', fontweight='bold')

    ax1.set_title('(a) 各机型架次占比与载客复用倍数', fontsize=12, fontweight='bold', pad=10)
    ax1.spines['top'].set_visible(False)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    # --- 子图2: 三类需求运量结构环形图 ---
    ax2 = axes[1]
    demand_labels = demand_by_type.index.tolist()
    demand_sizes = demand_by_type.values
    demand_colors = {'出海': '#3B82F6', '海返': '#10B981', '穿梭': '#F59E0B'}
    colors2 = [demand_colors.get(d, '#999') for d in demand_labels]
    wedges2, texts2, autotexts2 = ax2.pie(
        demand_sizes, labels=demand_labels, colors=colors2, autopct='%1.1f%%',
        startangle=90, pctdistance=0.78,
        wedgeprops=dict(width=0.42, edgecolor='white', linewidth=2)
    )
    for t in autotexts2:
        t.set_fontsize(10)
        t.set_fontweight('bold')
    ax2.set_title('(b) 三类需求运量结构', fontsize=12, fontweight='bold', pad=10)
    total_people = demand_by_type.sum()
    ax2.text(0, 0, f'{total_people}\n人次', ha='center', va='center',
             fontsize=13, fontweight='bold', color='#333')

    plt.tight_layout(pad=1.2)
    out_path = os.path.join(OUT_DIR2, 'q2_solution_overview.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'\n图2已保存: {out_path}')

    # 打印统计信息
    print(f'\n问题二统计:')
    print(f'  机型分布: {dict(type_counts)}')
    print(f'  各机型平均载客量:')
    for t in types_present:
        avg = type_load_stats.loc[t, 'mean'] if t in type_load_stats.index else 0
        print(f'    {t}({seat_map[t]}座): 平均{avg:.1f}人, 复用{avg / seat_map[t]:.2f}x')
    print(f'  三类需求: {dict(demand_by_type)}')


if __name__ == '__main__':
    plot_q1_overview()
    plot_q2_overview()
