# -*- coding: utf-8 -*-
import csv
import os
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

PROJECT_DIR = r'D:\数模\26策联杯\8.13MM_STJU'
FIG_DIR = os.path.join(PROJECT_DIR, 'figures', 'Problem3')

# ==================== Data Loading ====================
routes_path = os.path.join(PROJECT_DIR, 'output', 'Problem3', 'q3-routes.csv')
assignments_path = os.path.join(PROJECT_DIR, 'output', 'Problem3', 'q3-assignments.csv')

flights_raw = []
with open(routes_path, 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        flights_raw.append(row)

assignments = []
with open(assignments_path, 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        assignments.append(row)

# Group by flight
flight_groups = defaultdict(list)
for fo in flights_raw:
    if fo['flight_no']:
        flight_groups[(fo['aircraft_id'], int(fo['flight_no']))].append(fo)

# Extract flight info
flight_info = []
for (ac_id, fno), stops in flight_groups.items():
    stops.sort(key=lambda x: int(x['stop_order']))
    dep = stops[0].get('departure_time', '')
    arr = stops[-1].get('arrival_time', '')
    if dep and arr:
        dep_dt = datetime.strptime(dep, '%Y-%m-%d %H:%M')
        arr_dt = datetime.strptime(arr, '%Y-%m-%d %H:%M')
        duration = (arr_dt - dep_dt).total_seconds() / 60
        flight_info.append({
            'date': dep_dt.strftime('%Y-%m-%d'),
            'aircraft': ac_id,
            'flight': fno,
            'duration': duration,
            'dep_dt': dep_dt,
            'arr_dt': arr_dt,
        })

# Count people per flight
people_per_flight = defaultdict(int)
for a in assignments:
    if a['flight_no']:
        key = (a['aircraft_id'], int(a['flight_no']))
        people_per_flight[key] += 1

# Daily stats
daily = defaultdict(lambda: {'flights': 0, 'time': 0, 'people': 0})
for fi in flight_info:
    date = fi['date']
    key = (fi['aircraft'], fi['flight'])
    daily[date]['flights'] += 1
    daily[date]['time'] += fi['duration']
    daily[date]['people'] += people_per_flight.get(key, 0)

# Sort dates
dates = sorted(daily.keys())
date_labels = [d[5:] for d in dates]  # MM-DD

flights_arr = [daily[d]['flights'] for d in dates]
time_arr = [daily[d]['time'] for d in dates]
people_arr = [daily[d]['people'] for d in dates]
per_flight_arr = [daily[d]['people'] / daily[d]['flights'] if daily[d]['flights'] > 0 else 0 for d in dates]

# ==================== F1: Daily Metrics ====================
COLOR_PRIMARY = '#2C5F8D'
COLOR_ACCENT = '#E8833A'
COLOR_GREEN = '#5B9F5B'
COLOR_RED = '#C0504D'

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('问题三：各日求解指标对比（新方案）', fontsize=14, fontweight='bold')
x = np.arange(len(dates))

# (a) flights
ax = axes[0, 0]
bars = ax.bar(x, flights_arr, color=COLOR_PRIMARY, alpha=0.85, width=0.6)
ax.set_ylabel('架次数', fontsize=11)
ax.set_title('(a) 每日架次数', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(date_labels, fontsize=9)
for bar, val in zip(bars, flights_arr):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            str(val), ha='center', va='bottom', fontsize=9)

# (b) time
ax = axes[0, 1]
bars = ax.bar(x, time_arr, color=COLOR_ACCENT, alpha=0.85, width=0.6)
ax.set_ylabel('总飞机使用时间 (min)', fontsize=11)
ax.set_title('(b) 每日总飞机使用时间', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(date_labels, fontsize=9)
for bar, val in zip(bars, time_arr):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
            str(int(val)), ha='center', va='bottom', fontsize=8)

# (c) per_flight
ax = axes[1, 0]
bars = ax.bar(x, per_flight_arr, color=COLOR_GREEN, alpha=0.85, width=0.6)
ax.set_ylabel('人均架次承载 (人/架次)', fontsize=11)
ax.set_title('(c) 每日人均架次承载', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(date_labels, fontsize=9)
for bar, val in zip(bars, per_flight_arr):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f'{val:.1f}', ha='center', va='bottom', fontsize=9)

# (d) people
ax = axes[1, 1]
bars = ax.bar(x, people_arr, color=COLOR_RED, alpha=0.85, width=0.6)
ax.set_ylabel('承运人数', fontsize=11)
ax.set_title('(d) 每日承运人数', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(date_labels, fontsize=9)
for bar, val in zip(bars, people_arr):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            str(val), ha='center', va='bottom', fontsize=9)

plt.tight_layout()
path1 = os.path.join(FIG_DIR, 'q3_daily_metrics_new.png')
plt.savefig(path1, bbox_inches='tight')
plt.close()
print('F1 saved:', path1)

# ==================== F3: Gantt Chart ====================
aircraft_schedules = defaultdict(list)
for fi in flight_info:
    if fi['date'] == '2026-08-03':
        ac_id = fi['aircraft']
        if ac_id.startswith('A01'):
            aircraft_schedules[ac_id].append(fi)

sorted_aircraft = sorted(aircraft_schedules.keys())

fig, ax = plt.subplots(figsize=(14, max(4, len(sorted_aircraft) * 0.5)))
bar_height = 0.6
colors_by_type = {'T1': '#2C5F8D', 'T2': '#E8833A', 'T3': '#5B9F5B'}

for i, ac_id in enumerate(sorted_aircraft):
    atype = ac_id.split('-')[1]
    color = colors_by_type.get(atype, '#A0A0A0')
    for flight in aircraft_schedules[ac_id]:
        from matplotlib.dates import date2num
        start_num = date2num(flight['dep_dt'])
        end_num = date2num(flight['arr_dt'])
        width = end_num - start_num
        ax.barh(i, width, left=start_num, height=bar_height,
                color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
        mid = start_num + width / 2
        key = (flight['aircraft'], flight['flight'])
        pcount = people_per_flight.get(key, 0)
        ax.text(mid, i, f"F{flight['flight']}({pcount})", ha='center', va='center',
                fontsize=7, color='white', fontweight='bold')

ax.set_yticks(range(len(sorted_aircraft)))
ax.set_yticklabels(sorted_aircraft, fontsize=9)
ax.xaxis_date()
from matplotlib.dates import DateFormatter
ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
ax.set_xlim(date2num(datetime(2026, 8, 3, 5, 30)),
            date2num(datetime(2026, 8, 3, 20, 30)))
ax.set_xlabel('时间 (2026-08-03)', fontsize=12)
ax.set_ylabel('飞机编号', fontsize=12)
ax.set_title('问题三：A01机场单日排班甘特图 (2026-08-03，新方案)', fontsize=13, fontweight='bold')

import matplotlib.patches as mpatches
patches = [mpatches.Patch(color=c, label=f'{t} ({s}座)')
           for t, c, s in [('T1', '#2C5F8D', 12), ('T2', '#E8833A', 16), ('T3', '#5B9F5B', 19)]]
ax.legend(handles=patches, loc='upper right', fontsize=9)
ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
path3 = os.path.join(FIG_DIR, 'q3_gantt_a01_0803_new.png')
plt.savefig(path3, bbox_inches='tight')
plt.close()
print('F3 saved:', path3)

# ==================== F4: Radar (updated) ====================
metrics = {
    '问题一': {'总时间(h)': 249.85, '在途时间(h)': 2141.58, '架次数': 90, '燃油(t)': 121.181, '利用率(%)': 48.96},
    '问题二': {'总时间(h)': 294.48, '在途时间(h)': 4427.97, '架次数': 95, '燃油(t)': 137.236, '利用率(%)': 89.98},
    '问题三': {'总时间(h)': 449.73, '在途时间(h)': 4460.18, '架次数': 150, '燃油(t)': 209.528, '利用率(%)': 63.78},
}
labels = list(metrics['问题一'].keys())
n = len(labels)
max_vals = {k: max(metrics[q][k] for q in metrics) for k in labels}
norm = {}
for q in metrics:
    norm[q] = [metrics[q][k] / max_vals[k] for k in labels]
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
ax.set_title('问题一/二/三 指标对比（归一化，新Q3）', fontsize=13, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
plt.tight_layout()
path4 = os.path.join(FIG_DIR, 'q3_q1q2q3_comparison_new.png')
plt.savefig(path4, bbox_inches='tight')
plt.close()
print('F4 saved:', path4)

print('All done!')
