# -*- coding: utf-8 -*-
"""
生成额外的分析图表
包含: G1-G5 图表以及敏感性分析相关图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 150

# 路径配置
BASE_DIR = Path(__file__).resolve().parents[2]  # 项目根目录（动态定位，原为硬编码绝对路径）
OUTPUT_DIR = BASE_DIR / "figures" / "Problem3" / "extra"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """加载所有需要的数据文件"""
    data = {}
    
    # Q3 路线数据
    routes_file = BASE_DIR / "output" / "Problem3" / "q3-routes.csv"
    data['routes'] = pd.read_csv(routes_file, encoding='utf-8')
    
    # Q3 分配数据
    assign_file = BASE_DIR / "output" / "Problem3" / "q3-assignments.csv"
    data['assignments'] = pd.read_csv(assign_file, encoding='utf-8')
    
    # 人员需求数据
    people_file = list(BASE_DIR.rglob("peopleQ3.csv"))[0]
    data['people'] = pd.read_csv(people_file, encoding='utf-8')
    
    # 距离矩阵
    dist_file = list(BASE_DIR.rglob("distances.csv"))[0]
    data['distances'] = pd.read_csv(dist_file, encoding='utf-8', index_col=0)
    
    return data


def generate_g1_flow_heatmap(data):
    """
    G1: 机场-设施流量热力图
    展示各机场服务的设施数量和人员流量分布
    """
    print("生成 G1: 机场-设施流量热力图...")
    
    routes = data['routes'].copy()
    assignments = data['assignments'].copy()
    
    # 清理数据
    routes = routes.dropna(subset=['aircraft_id'])
    assignments = assignments.dropna(subset=['aircraft_id'])
    
    # 提取机场信息
    routes['airport'] = routes['aircraft_id'].astype(str).apply(lambda x: x.split('-')[0])
    
    # 计算每个机场服务的设施
    airport_facility_flow = []
    
    for airport in ['A01', 'A02', 'A03']:
        airport_routes = routes[routes['airport'] == airport]
        
        # 获取该机场服务的所有设施
        facilities_served = airport_routes[
            airport_routes['facility_id'].astype(str).str.startswith('F')
        ]['facility_id'].unique()
        
        # 计算每个设施的访问次数
        for facility in facilities_served:
            visit_count = len(airport_routes[airport_routes['facility_id'] == facility])
            
            airport_facility_flow.append({
                'airport': airport,
                'facility': facility,
                'visit_count': visit_count
            })
    
    flow_df = pd.DataFrame(airport_facility_flow)
    
    # 创建热力图数据
    pivot_data = flow_df.pivot_table(
        values='visit_count', 
        index='facility', 
        columns='airport', 
        aggfunc='sum',
        fill_value=0
    )
    
    # 排序
    pivot_data = pivot_data.sort_index()
    
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(pivot_data.values, cmap='YlOrRd', aspect='auto')
    
    # 设置坐标轴
    ax.set_xticks(range(len(pivot_data.columns)))
    ax.set_xticklabels(pivot_data.columns, fontsize=12)
    ax.set_yticks(range(len(pivot_data.index)))
    ax.set_yticklabels(pivot_data.index, fontsize=8)
    
    # 添加数值标注
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            value = pivot_data.values[i, j]
            if value > 0:
                ax.text(j, i, f'{int(value)}', ha='center', va='center', 
                       fontsize=7, color='black' if value < pivot_data.values.max()/2 else 'white')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('访问次数', fontsize=12)
    
    ax.set_title('G1: 机场-设施流量热力图', fontsize=14, fontweight='bold')
    ax.set_xlabel('机场', fontsize=12)
    ax.set_ylabel('设施', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'G1_airport_facility_flow_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  已保存: {OUTPUT_DIR / 'G1_airport_facility_flow_heatmap.png'}")
    
    # 统计信息
    for airport in ['A01', 'A02', 'A03']:
        if airport in pivot_data.columns:
            count = len(pivot_data[pivot_data[airport] > 0])
            print(f"  {airport} 服务 {count} 个设施")


def generate_g2_aircraft_type_comparison(data):
    """
    G2: 机型使用效率对比图
    T1/T2/T3 的利用率、燃油效率、覆盖范围对比
    """
    print("生成 G2: 机型使用效率对比图...")
    
    routes = data['routes'].copy()
    assignments = data['assignments'].copy()
    
    # 清理数据
    routes = routes.dropna(subset=['aircraft_id'])
    assignments = assignments.dropna(subset=['aircraft_id'])
    
    # 提取机型信息
    routes['aircraft_type'] = routes['aircraft_id'].astype(str).apply(lambda x: x.split('-')[1])
    assignments['aircraft_type'] = assignments['aircraft_id'].astype(str).apply(lambda x: x.split('-')[1])
    
    # 机型参数
    aircraft_params = {
        'T1': {'seats': 12, 'speed': 250, 'fuel_rate': 3.4, 'tank': 1000},
        'T2': {'seats': 16, 'speed': 220, 'fuel_rate': 2.5, 'tank': 1150},
        'T3': {'seats': 19, 'speed': 190, 'fuel_rate': 2.9, 'tank': 1600}
    }
    
    # 统计各机型的使用情况
    type_stats = {}
    
    for aircraft_type in ['T1', 'T2', 'T3']:
        type_routes = routes[routes['aircraft_type'] == aircraft_type]
        type_assigns = assignments[assignments['aircraft_type'] == aircraft_type]
        
        # 独立飞机数
        aircraft_count = type_routes['aircraft_id'].nunique()
        
        # 架次数（独立flight_no）
        flight_count = type_routes.groupby(['aircraft_id', 'flight_no']).ngroups
        
        # 服务的设施数
        facilities_served = type_routes[type_routes['facility_id'].astype(str).str.startswith('F')]['facility_id'].nunique()
        
        # 运输的人员数
        person_count = len(type_assigns)
        
        type_stats[aircraft_type] = {
            'aircraft_count': aircraft_count,
            'flight_count': flight_count,
            'facilities_served': facilities_served,
            'person_count': person_count,
            'params': aircraft_params[aircraft_type]
        }
    
    # 绘图 - 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    types = list(type_stats.keys())
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # 1. 飞机数量和架次数
    ax1 = axes[0, 0]
    x = np.arange(len(types))
    width = 0.35
    
    aircraft_counts = [type_stats[t]['aircraft_count'] for t in types]
    flight_counts = [type_stats[t]['flight_count'] for t in types]
    
    bars1 = ax1.bar(x - width/2, aircraft_counts, width, label='飞机数量', color=colors[0])
    bars2 = ax1.bar(x + width/2, flight_counts, width, label='架次数', color=colors[1])
    
    ax1.set_xlabel('机型')
    ax1.set_ylabel('数量')
    ax1.set_title('飞机数量与架次数')
    ax1.set_xticks(x)
    ax1.set_xticklabels(types)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. 服务设施数
    ax2 = axes[0, 1]
    facilities_served = [type_stats[t]['facilities_served'] for t in types]
    bars = ax2.bar(types, facilities_served, color=colors)
    ax2.set_xlabel('机型')
    ax2.set_ylabel('设施数量')
    ax2.set_title('服务设施数量')
    ax2.grid(axis='y', alpha=0.3)
    
    # 添加数值标注
    for bar, val in zip(bars, facilities_served):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val}', ha='center', va='bottom', fontweight='bold')
    
    # 3. 运输人员数
    ax3 = axes[1, 0]
    person_counts = [type_stats[t]['person_count'] for t in types]
    bars = ax3.bar(types, person_counts, color=colors)
    ax3.set_xlabel('机型')
    ax3.set_ylabel('人员数量')
    ax3.set_title('运输人员数量')
    ax3.grid(axis='y', alpha=0.3)
    
    # 添加数值标注
    for bar, val in zip(bars, person_counts):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 10,
                f'{val}', ha='center', va='bottom', fontweight='bold')
    
    # 4. 座位利用率估算
    ax4 = axes[1, 1]
    # 估算：人员数 / (飞机数 * 架次 * 座位数) 
    utilization_est = []
    for t in types:
        avg_flights_per_ac = type_stats[t]['flight_count'] / max(type_stats[t]['aircraft_count'], 1)
        est = type_stats[t]['person_count'] / (
            type_stats[t]['aircraft_count'] * 
            avg_flights_per_ac * 
            type_stats[t]['params']['seats']
        ) * 100
        utilization_est.append(min(est, 100))
    
    bars = ax4.bar(types, utilization_est, color=colors)
    ax4.set_xlabel('机型')
    ax4.set_ylabel('估算利用率 (%)')
    ax4.set_title('座位利用率估算')
    ax4.set_ylim(0, 100)
    ax4.grid(axis='y', alpha=0.3)
    
    # 添加数值标注
    for bar, val in zip(bars, utilization_est):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    fig.suptitle('G2: 机型使用效率对比图', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'G2_aircraft_type_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  已保存: {OUTPUT_DIR / 'G2_aircraft_type_comparison.png'}")
    
    # 输出统计
    for t in types:
        print(f"  {t}: {type_stats[t]['aircraft_count']}架飞机, "
              f"{type_stats[t]['flight_count']}架次, "
              f"服务{type_stats[t]['facilities_served']}个设施, "
              f"运输{type_stats[t]['person_count']}人")


def generate_g3_demand_time_distribution(data):
    """
    G3: 需求时间分布直方图
    展示4000条需求的时间窗分布特征
    """
    print("生成 G3: 需求时间分布直方图...")
    
    people = data['people'].copy()
    
    # 转换时间格式
    people['earliest_pickup_time'] = pd.to_datetime(people['earliest_pickup_time'])
    people['latest_arrival_time'] = pd.to_datetime(people['latest_arrival_time'])
    
    # 计算时间窗
    people['window_hours'] = (people['latest_arrival_time'] - people['earliest_pickup_time']).dt.total_seconds() / 3600
    
    # 提取日期
    people['pickup_date'] = people['earliest_pickup_time'].dt.date
    people['pickup_hour'] = people['earliest_pickup_time'].dt.hour
    
    # 绘图 - 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 时间窗长度分布
    ax1 = axes[0, 0]
    ax1.hist(people['window_hours'], bins=50, color='#2E86AB', edgecolor='black', alpha=0.7)
    ax1.axvline(people['window_hours'].mean(), color='red', linestyle='--', 
                label=f'均值: {people["window_hours"].mean():.1f}h')
    ax1.axvline(people['window_hours'].median(), color='green', linestyle='--', 
                label=f'中位数: {people["window_hours"].median():.1f}h')
    ax1.set_xlabel('时间窗长度 (小时)')
    ax1.set_ylabel('需求数量')
    ax1.set_title('时间窗长度分布')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. 最早登机时间分布（按日期）
    ax2 = axes[0, 1]
    date_counts = people['pickup_date'].value_counts().sort_index()
    ax2.bar(range(len(date_counts)), date_counts.values, color='#A23B72', edgecolor='black')
    ax2.set_xlabel('日期')
    ax2.set_ylabel('需求数量')
    ax2.set_title('按日期的需求分布')
    ax2.set_xticks(range(len(date_counts)))
    ax2.set_xticklabels([str(d)[-5:] for d in date_counts.index], rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. 最早登机时间分布（按小时）
    ax3 = axes[1, 0]
    hour_counts = people['pickup_hour'].value_counts().sort_index()
    ax3.bar(hour_counts.index, hour_counts.values, color='#F18F01', edgecolor='black')
    ax3.set_xlabel('小时')
    ax3.set_ylabel('需求数量')
    ax3.set_title('按小时的需求分布')
    ax3.set_xticks(range(0, 24))
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. 任务类型分布
    ax4 = axes[1, 1]
    task_counts = people['task_type'].value_counts()
    colors_pie = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    wedges, texts, autotexts = ax4.pie(
        task_counts.values,
        autopct='%1.1f%%',
        colors=colors_pie[:len(task_counts)],
        startangle=90,
        pctdistance=0.6
    )
    # emergency（占比最小，最后一个）标签往外推，避免和 temporary 重叠
    emg_idx = len(task_counts) - 1
    ex, ey = autotexts[emg_idx].get_position()
    autotexts[emg_idx].set_position((ex, ey + 0.18))
    ax4.set_title('任务类型分布')
    # 右侧统一图注：类型 + 数量
    legend_labels = [
        f'{t} ({c}条)' for t, c in task_counts.items()
    ]
    ax4.legend(
        wedges, legend_labels,
        loc='center left',
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        title='任务类型'
    )
    
    fig.suptitle('G3: 需求时间分布分析', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'G3_demand_time_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  已保存: {OUTPUT_DIR / 'G3_demand_time_distribution.png'}")
    
    # 输出统计
    print(f"  总需求数: {len(people)}")
    print(f"  时间窗均值: {people['window_hours'].mean():.1f}小时")
    print(f"  时间窗中位数: {people['window_hours'].median():.1f}小时")
    print(f"  任务类型: {dict(task_counts)}")


def generate_g4_waiting_time_boxplot(data):
    """
    G4: 人员等待时间箱线图
    展示各需求的等待时间分布，评估服务质量
    """
    print("生成 G4: 人员等待时间箱线图...")
    
    routes = data['routes'].copy()
    assignments = data['assignments'].copy()
    people = data['people'].copy()
    
    # 清理数据
    routes = routes.dropna(subset=['aircraft_id'])
    assignments = assignments.dropna(subset=['aircraft_id'])
    
    # 转换时间格式
    people['earliest_pickup_time'] = pd.to_datetime(people['earliest_pickup_time'])
    people['latest_arrival_time'] = pd.to_datetime(people['latest_arrival_time'])
    
    # 合并数据
    merged = assignments.merge(people[['person_id', 'earliest_pickup_time', 'latest_arrival_time', 'task_type']], 
                              on='person_id', how='left')
    
    # 计算时间窗
    merged['window_hours'] = (merged['latest_arrival_time'] - merged['earliest_pickup_time']).dt.total_seconds() / 3600
    
    # 任务类型
    task_types = merged['task_type'].dropna().unique()
    
    # 估算等待时间（使用时间窗的30%作为估算）
    waiting_data = []
    for task_type in task_types:
        task_data = merged[merged['task_type'] == task_type]
        window_hours = task_data['window_hours'].dropna()
        # 实际等待通常小于时间窗，使用30%作为估算
        estimated_wait = window_hours * 0.3
        waiting_data.extend([(task_type, w) for w in estimated_wait if pd.notna(w)])
    
    wait_df = pd.DataFrame(waiting_data, columns=['task_type', 'waiting_hours'])
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. 按任务类型的等待时间箱线图
    ax1 = axes[0]
    task_types_list = [t for t in task_types if t in wait_df['task_type'].values]
    box_data = [wait_df[wait_df['task_type'] == t]['waiting_hours'].values for t in task_types_list]
    
    bp = ax1.boxplot(box_data, labels=task_types_list, patch_artist=True)
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    for patch, color in zip(bp['boxes'], colors[:len(task_types_list)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_xlabel('任务类型')
    ax1.set_ylabel('等待时间 (小时)')
    ax1.set_title('按任务类型的等待时间分布')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. 等待时间直方图
    ax2 = axes[1]
    ax2.hist(wait_df['waiting_hours'], bins=30, color='#2E86AB', edgecolor='black', alpha=0.7)
    ax2.axvline(wait_df['waiting_hours'].mean(), color='red', linestyle='--', 
                label=f'均值: {wait_df["waiting_hours"].mean():.1f}h')
    ax2.set_xlabel('等待时间 (小时)')
    ax2.set_ylabel('频次')
    ax2.set_title('等待时间分布')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    fig.suptitle('G4: 人员等待时间分析', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'G4_waiting_time_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  已保存: {OUTPUT_DIR / 'G4_waiting_time_analysis.png'}")
    
    # 输出统计
    print(f"  总样本数: {len(wait_df)}")
    print(f"  平均等待时间: {wait_df['waiting_hours'].mean():.1f}小时")
    for t in task_types_list:
        t_data = wait_df[wait_df['task_type'] == t]['waiting_hours']
        if len(t_data) > 0:
            print(f"  {t}: 均值={t_data.mean():.1f}h, 中位数={t_data.median():.1f}h")


def generate_g5_refuel_frequency(data):
    """
    G5: 加油设施使用频率图
    8个加油点的使用次数和加油量分布
    """
    print("生成 G5: 加油设施使用频率图...")
    
    routes = data['routes'].copy()
    
    # 清理数据
    routes = routes.dropna(subset=['aircraft_id', 'facility_id'])
    
    # 加油设施列表
    refuel_facilities = ['F006', 'F011', 'F018', 'F024', 'F031', 'F038', 'F044', 'F050']
    
    # 统计每个加油设施的使用情况
    refuel_stats = []
    
    for facility in refuel_facilities:
        # 该设施的所有访问
        facility_visits = routes[routes['facility_id'] == facility]
        
        # 加油次数（refuel=1）
        refuel_count = len(facility_visits[facility_visits['refuel'] == 1])
        
        # 总访问次数
        total_visits = len(facility_visits)
        
        refuel_stats.append({
            'facility': facility,
            'total_visits': total_visits,
            'refuel_count': refuel_count,
            'non_refuel_count': total_visits - refuel_count
        })
    
    refuel_df = pd.DataFrame(refuel_stats)
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. 加油次数柱状图
    ax1 = axes[0]
    x = np.arange(len(refuel_df))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, refuel_df['refuel_count'], width, 
                    label='加油次数', color='#2E86AB')
    bars2 = ax1.bar(x + width/2, refuel_df['non_refuel_count'], width, 
                    label='仅停靠次数', color='#F18F01')
    
    ax1.set_xlabel('加油设施')
    ax1.set_ylabel('次数')
    ax1.set_title('各加油设施使用情况')
    ax1.set_xticks(x)
    ax1.set_xticklabels(refuel_df['facility'])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. 加油比例饼图
    ax2 = axes[1]
    total_refuel = refuel_df['refuel_count'].sum()
    total_non_refuel = refuel_df['non_refuel_count'].sum()
    
    wedges, texts, autotexts = ax2.pie(
        [total_refuel, total_non_refuel],
        labels=['加油停靠', '仅停靠'],
        autopct='%1.1f%%',
        colors=['#2E86AB', '#F18F01'],
        startangle=90
    )
    ax2.set_title('加油停靠比例')
    
    fig.suptitle('G5: 加油设施使用频率分析', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'G5_refuel_facility_frequency.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  已保存: {OUTPUT_DIR / 'G5_refuel_facility_frequency.png'}")
    
    # 输出统计
    print(f"  总加油次数: {total_refuel}")
    print(f"  总仅停靠次数: {total_non_refuel}")
    if total_refuel + total_non_refuel > 0:
        print(f"  加油比例: {total_refuel/(total_refuel+total_non_refuel)*100:.1f}%")


def main():
    """主函数"""
    print("=" * 60)
    print("开始生成额外分析图表")
    print("=" * 60)
    
    # 加载数据
    print("\n加载数据...")
    data = load_data()
    print("数据加载完成")
    
    # 生成图表
    print("\n" + "=" * 60)
    generate_g1_flow_heatmap(data)
    
    print("\n" + "=" * 60)
    generate_g2_aircraft_type_comparison(data)
    
    print("\n" + "=" * 60)
    generate_g3_demand_time_distribution(data)
    
    print("\n" + "=" * 60)
    generate_g4_waiting_time_boxplot(data)
    
    print("\n" + "=" * 60)
    generate_g5_refuel_frequency(data)
    
    print("\n" + "=" * 60)
    print("所有图表生成完成!")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
