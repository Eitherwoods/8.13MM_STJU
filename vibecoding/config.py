# -*- coding: utf-8 -*-
"""
文件: config.py
功能: 全局配置文件，包含路径、机型参数、加油设施等常量定义
项目: 2026策联杯 B题 - 海上油田人员直升机运载计划编排
"""

import os

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# 附件目录（含全角引号，需动态查找）
_TOPIC_DIR = os.path.join(PROJECT_DIR, '题目和思路')
DATA_DIR = None
for _d in os.listdir(_TOPIC_DIR):
    _full = os.path.join(_TOPIC_DIR, _d)
    if os.path.isdir(_full) and '附件' in _d:
        DATA_DIR = _full
        break
if DATA_DIR is None:
    raise FileNotFoundError("未找到附件文件夹")

FIGURES_DIR = os.path.join(PROJECT_DIR, 'figures', 'Problem1')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output', 'Problem1')

# Problem2 路径
FIGURES_DIR_Q2 = os.path.join(PROJECT_DIR, 'figures', 'Problem2')
OUTPUT_DIR_Q2 = os.path.join(PROJECT_DIR, 'output', 'Problem2')

# ==================== 机型参数 ====================
AIRCRAFT = {
    'T1': {'seats': 12, 'speed': 250, 'fuel_rate': 3.4, 'tank': 1000, 'min_reserve': 150},
    'T2': {'seats': 16, 'speed': 220, 'fuel_rate': 2.5, 'tank': 1150, 'min_reserve': 150},
    'T3': {'seats': 19, 'speed': 190, 'fuel_rate': 2.9, 'tank': 1600, 'min_reserve': 200},
}

# 机型列表（按座位数升序）
AIRCRAFT_TYPES = ['T1', 'T2', 'T3']

# ==================== 设施参数 ====================
AIRPORTS = ['A01', 'A02', 'A03']
REFUEL_FACILITIES = {'F006', 'F011', 'F018', 'F024', 'F031', 'F038', 'F044', 'F050'}

# 每架次最大海上着陆次数
MAX_LANDINGS = 5

# 停靠时间（分钟）
STOP_TIME_NO_REFUEL = 10
STOP_TIME_REFUEL = 20

# ==================== 数值格式化 ====================
def fmt6(x, max_decimals=6):
    """保留6位小数的格式化函数"""
    if x is None:
        return 'None'
    s = f'{x:.{max_decimals}f}'
    s = s.rstrip('0').rstrip('.')
    if s == '' or s == '-':
        s = '0'
    return s


def setup_matplotlib():
    """配置matplotlib中文字体"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    return plt
