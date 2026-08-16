# -*- coding: utf-8 -*-
"""
文件: generate_f5_lagrangian.py
功能: 运行拉格朗日松弛求解器并生成收敛曲线图 (F5)
项目: 2026策联杯 B题

使用方法: E:\\Python\\python.exe generate_f5_lagrangian.py
预计耗时: 5-8分钟 (需要COPT求解器)
"""

import os
import sys
import csv

# ==================== 模块路径 ====================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VIBECODING_DIR = os.path.dirname(_THIS_DIR)
if _VIBECODING_DIR not in sys.path:
    sys.path.insert(0, _VIBECODING_DIR)

# problem3 子包路径（导入 problem3_solution, lagrangian_bound）
_PROBLEM3_DIR = os.path.join(_VIBECODING_DIR, 'problem3')
if _PROBLEM3_DIR not in sys.path:
    sys.path.insert(0, _PROBLEM3_DIR)

_PROJECT_DIR = os.path.dirname(_VIBECODING_DIR)
_FIG_DIR = os.path.join(_PROJECT_DIR, 'figures', 'Problem3')
_OUTPUT_DIR = os.path.join(_PROJECT_DIR, 'output', 'Problem3')
os.makedirs(_FIG_DIR, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import date

# 字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# 导入项目模块
from problem3_solution import (
    Q3DataLoader, Q3DemandDayAllocator, Q3RouteGenerator,
    PLANNING_DATES,
)
from lagrangian_bound import LagrangianBoundAnalyzer


def main():
    """运行拉格朗日求解器并生成收敛曲线图"""

    print('=' * 60)
    print('生成 F5: 拉格朗日下界收敛曲线')
    print('=' * 60)

    # ========== 1. 加载数据 ==========
    print('\n[1] 加载数据...')
    loader = Q3DataLoader()
    distances = loader.load_distances()
    all_demands = loader.load_demands_q3()
    print(f'  总需求: {len(all_demands)}')

    non_temp = [d for d in all_demands if d['task_type'] != 'temporary']
    print(f'  非临时需求(阶段一): {len(non_temp)}')

    # ========== 2. 需求-日期分配 ==========
    print('\n[2] 需求-日期分配...')
    allocation, daily_load = Q3DemandDayAllocator.allocate(non_temp, PLANNING_DATES)
    for d in non_temp:
        d['_assigned_day'] = allocation[d['person_id']]

    # ========== 3. 选取目标日期 ==========
    target_day = date(2026, 8, 3)
    day_demands = [d for d in non_temp if allocation[d['person_id']] == target_day]
    print(f'\n[3] 目标日期: {target_day} ({len(day_demands)}人)')

    # ========== 4. 生成候选路径 ==========
    print('\n[4] 生成候选路径...')
    generator = Q3RouteGenerator(distances)
    routes_info = generator.generate_daily_routes(day_demands, target_day, route_id_start=1)
    print(f'  候选路径: {len(routes_info)}')

    if not routes_info:
        print('  无候选路径，退出')
        return

    # ========== 5. 构建分析器 ==========
    print('\n[5] 构建下界分析器...')
    analyzer = LagrangianBoundAnalyzer(routes_info, day_demands, target_day, distances)

    # ========== 6. 求解MIP (上界) ==========
    print('\n[6] 求解MIP(上界)...')
    mip_obj, copt_bound = analyzer.solve_mip(time_limit=7200, rel_gap=0.01)
    if mip_obj is None:
        print('  MIP求解失败，退出')
        return
    print(f'  MIP目标值: {mip_obj:.4f}')
    if copt_bound is not None:
        print(f'  COPT下界: {copt_bound:.4f}')

    # ========== 7. 计算LP松弛下界 ==========
    print('\n[7] 计算LP松弛下界...')
    lp_bound = analyzer.solve_lp_relaxation(time_limit=7200)

    # ========== 8. 拉格朗日松弛 (减少参数以加速) ==========
    print('\n[8] 计算拉格朗日松弛下界 (max_iters=20, time/iter=7200s)...')
    lag_bound, lag_history = analyzer.solve_lagrangian(
        ub=mip_obj,
        max_iters=20,
        time_limit_per_iter=7200,
    )

    # ========== 9. 保存历史到CSV ==========
    csv_path = os.path.join(_OUTPUT_DIR, 'lagrangian_history.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'iter', 'lag_val', 'lag_bound', 'best_bound',
            'subgrad_norm', 'n_violated', 'n_constraints',
            'beta', 'time', 'status',
        ])
        writer.writeheader()
        for h in lag_history:
            row = dict(h)
            if row['best_bound'] is None or row['best_bound'] == -float('inf'):
                row['best_bound'] = ''
            if row['lag_bound'] is None:
                row['lag_bound'] = ''
            writer.writerow(row)
    print(f'\n  历史数据已保存: {csv_path}')

    # ========== 10. 绘制收敛曲线 ==========
    print('\n[10] 绘制收敛曲线...')

    fig, ax = plt.subplots(figsize=(10, 6))

    iters = [h['iter'] for h in lag_history]
    lag_vals = [h['lag_val'] for h in lag_history]

    # 有效下界 (lag_bound可能为None，需过滤)
    lb_iters = []
    lb_vals = []
    for h in lag_history:
        if h['lag_bound'] is not None:
            lb_iters.append(h['iter'])
            lb_vals.append(h['lag_bound'])

    # 最佳下界
    bb_iters = []
    bb_vals = []
    for h in lag_history:
        if h['best_bound'] is not None and h['best_bound'] != -float('inf'):
            bb_iters.append(h['iter'])
            bb_vals.append(h['best_bound'])

    # 画曲线
    ax.plot(iters, lag_vals, 'o-', color='#E8833A', linewidth=1.5, markersize=5,
            label='L(λ) 估计值', alpha=0.8)
    if lb_iters:
        ax.plot(lb_iters, lb_vals, 's--', color='#2C5F8D', linewidth=1.5, markersize=5,
                label='有效拉格朗日下界', alpha=0.8)
    if bb_iters:
        ax.plot(bb_iters, bb_vals, '^-', color='#5B9F5B', linewidth=2, markersize=6,
                label='最佳下界', zorder=5)

    # 画水平参考线
    ax.axhline(y=mip_obj, color='#C0504D', linestyle='-', linewidth=1.5, alpha=0.7,
               label=f'MIP上界 ({mip_obj:.1f})')
    if copt_bound is not None:
        ax.axhline(y=copt_bound, color='#8064A7', linestyle='--', linewidth=1.5, alpha=0.7,
                   label=f'COPT内置下界 ({copt_bound:.1f})')
    if lp_bound is not None:
        ax.axhline(y=lp_bound, color='#A0A0A0', linestyle=':', linewidth=1.5, alpha=0.7,
                   label=f'LP松弛下界 ({lp_bound:.1f})')

    ax.set_xlabel('迭代次数', fontsize=12)
    ax.set_ylabel('目标值', fontsize=12)
    ax.set_title('问题三：拉格朗日松弛下界收敛曲线 (2026-08-03单日)', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # 添加间隙标注
    if lag_bound is not None and mip_obj is not None:
        gap = (mip_obj - lag_bound) / abs(mip_obj) * 100 if mip_obj != 0 else 0
        ax.text(0.02, 0.95, f'拉格朗日间隙: {gap:.1f}%',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    fig_path = os.path.join(_FIG_DIR, 'q3_lagrangian_convergence.png')
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f'  图表已保存: {fig_path}')

    print(f'\n完成! MIP={mip_obj:.2f}, Lagrangian={lag_bound:.2f}')


if __name__ == '__main__':
    main()
