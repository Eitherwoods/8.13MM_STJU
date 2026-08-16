# -*- coding: utf-8 -*-
"""
文件: validate_all.py
功能: 统一验证脚本（V1-V10），对 Q1/Q2/Q3 的分配与路径结果做完整性校验
项目: 2026策联杯 B题 - 海上油田人员直升机运载计划编排

用法:
    python validate_all.py --all      # 验证 Q1/Q2/Q3 全部
    python validate_all.py --q1       # 仅验证 Q1
    python validate_all.py --q2       # 仅验证 Q2
    python validate_all.py --q3       # 仅验证 Q3

验证项:
    V1  需求满足率          V2  Q1 总容量违规
    V3  Q2 航段级容量违规   V4  Q3 航段级容量违规
    V5  Q3 时间窗违规       V6  Q3 日期匹配违规
    V7  OD 一致性           V8  Q2/Q3 方向顺序
    V9  重复分配            V10 航油约束

判定规则:
    每项 PASS / FAIL / WARN；存在 FAIL 时退出码为 1，仅 WARN 时退出码为 0。
    报告同时保存到 deliverables/validation_report_{timestamp}.txt。
"""

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

# ==================== 模块路径 ====================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from config import AIRCRAFT, AIRPORTS, DATA_DIR, PROJECT_DIR  # noqa: E402
from base_models import FuelChecker  # noqa: E402

# ==================== 常量 ====================
OUTPUT_DIR_Q1 = os.path.join(PROJECT_DIR, 'output', 'Problem1')
OUTPUT_DIR_Q2 = os.path.join(PROJECT_DIR, 'output', 'Problem2')
OUTPUT_DIR_Q3 = os.path.join(PROJECT_DIR, 'output', 'Problem3')
REPORT_DIR = os.path.join(PROJECT_DIR, 'output')

OPERATION_START_HOUR = 6
RETURN_DEADLINE_HOUR = 20
PLANNING_DATES = [datetime(2026, 8, d) for d in range(3, 10)]


# ==================== 数据加载 ====================
class DataLoader:
    """从附件目录加载 Q1/Q2/Q3 需求数据"""

    @staticmethod
    def load_demands(q):
        """加载指定问题的需求列表，返回 [{person_id, origin_id, destination_id, ...}]"""
        path = os.path.join(DATA_DIR, 'peopleQ%d.csv' % q)
        demands = []
        with open(path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                item = {
                    'person_id': row['person_id'].strip(),
                    'origin_id': row['origin_id'].strip(),
                    'destination_id': row['destination_id'].strip(),
                }
                if q == 3:
                    item['earliest_pickup'] = datetime.strptime(
                        row['earliest_pickup_time'].strip(), '%Y-%m-%d %H:%M')
                    item['latest_arrival'] = datetime.strptime(
                        row['latest_arrival_time'].strip(), '%Y-%m-%d %H:%M')
                    item['task_type'] = row['task_type'].strip()
                demands.append(item)
        return demands

    @staticmethod
    def load_distances():
        """加载距离矩阵，返回 {fac1: {fac2: distance_km}}"""
        path = os.path.join(DATA_DIR, 'distances.csv')
        distances = defaultdict(dict)
        with open(path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                keys = list(row.keys())
                a = (row['from_id'] if 'from_id' in row
                     else row['Unnamed: 0'] if 'Unnamed: 0' in row
                     else row[keys[0]]).strip()
                for k, v in row.items():
                    if k in ('from_id', 'Unnamed: 0') or k == keys[0]:
                        continue
                    try:
                        distances[a][k.strip()] = float(v)
                    except ValueError:
                        pass
        return distances


class OutputReader:
    """读取 Q1/Q2/Q3 的 assignments.csv 与 routes.csv，重建航班结构"""

    @staticmethod
    def read_assignments(q, output_dir):
        """读取分配文件，返回 (assign, total_pids, header)

        assign: {person_id: {'flight_key': tuple, 'pidx': int, 'didx': int}}
        """
        path = os.path.join(output_dir, 'q%d-assignments.csv' % q)
        assign = {}
        with open(path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row['person_id'].strip()
                if not pid:
                    continue
                fn_s = row['flight_no'].strip()
                if not fn_s:
                    continue  # 未分配人员（q3 输出空行）
                if q == 3:
                    key = (row['aircraft_id'].strip(), int(fn_s))
                else:
                    key = (row['aircraft_type'].strip(), int(fn_s))
                assign[pid] = {
                    'flight_key': key,
                    'pidx': int(row['pickup_stop_order'].strip()),
                    'didx': int(row['delivery_stop_order'].strip()),
                }
        return assign

    @staticmethod
    def read_routes(q, output_dir):
        """读取路径文件，重建航班

        返回: {flight_key: {'path': [...], 'airport': str, 'atype': str,
                            'refuel_at': set, 'schedule': [(arr, dep), ...]}}
        """
        path = os.path.join(output_dir, 'q%d-routes.csv' % q)
        flights = defaultdict(lambda: {'stops': [], 'atype': None, 'airport': None})
        with open(path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if q == 3:
                    key = (row['aircraft_id'].strip(), int(row['flight_no'].strip()))
                    atype = row['aircraft_id'].strip().split('-')[1]
                else:
                    key = (row['aircraft_type'].strip(), int(row['flight_no'].strip()))
                    atype = row['aircraft_type'].strip()
                order = int(row['stop_order'].strip())
                fac = row['facility_id'].strip()
                arr_s = row.get('arrival_time', '').strip()
                dep_s = row.get('departure_time', '').strip()
                arr = datetime.strptime(arr_s, '%Y-%m-%d %H:%M') if arr_s else None
                dep = datetime.strptime(dep_s, '%Y-%m-%d %H:%M') if dep_s else None
                refuel = int(row.get('refuel', '0').strip())
                flights[key]['stops'].append((order, fac, arr, dep, refuel))
                flights[key]['atype'] = atype

        result = {}
        for key, info in flights.items():
            stops = sorted(info['stops'], key=lambda s: s[0])
            path_list = [s[1] for s in stops]
            schedule = [(s[2], s[3]) for s in stops]
            refuel_at = {s[0] for s in stops if s[4] == 1}
            result[key] = {
                'path': path_list,
                'airport': path_list[0] if path_list else None,
                'atype': info['atype'],
                'refuel_at': refuel_at,
                'schedule': schedule,
            }
        return result


# ==================== 验证器 ====================
class Validator:
    """V1-V10 统一验证器，逐项返回 (名称, 判定, 详情)"""

    def __init__(self, q, demands, assign, flights, distances):
        self.q = q
        self.demands = {d['person_id']: d for d in demands}
        self.assign = assign
        self.flights = flights
        self.distances = distances

    @staticmethod
    def _seg_loads(assign, flights):
        """按航班重建各航段在机人数与分配明细，返回 (seg_loads, person_segs)"""
        seg_loads = {}
        person_segs = {}
        for pid, info in assign.items():
            key = info['flight_key']
            if key not in flights:
                person_segs[pid] = None
                continue
            n_seg = max(0, len(flights[key]['path']) - 1)
            pidx, didx = info['pidx'], info['didx']
            if pidx >= didx or didx > n_seg:
                person_segs[pid] = None
                continue
            segs = list(range(pidx, didx))
            person_segs[pid] = segs
            seg_loads.setdefault(key, [0] * n_seg)
            for k in segs:
                seg_loads[key][k] += 1
        return seg_loads, person_segs

    def v1_satisfaction(self):
        """V1 需求满足率：已分配人数 / 总需求"""
        total = len(self.demands)
        matched = sum(1 for pid in self.demands if pid in self.assign)
        ratio = matched / total if total else 1.0
        status = 'PASS' if ratio >= 1.0 else 'WARN'
        detail = '已分配 %d/%d (%.4f)' % (matched, total, ratio)
        return 'V1 需求满足率(Q%d)' % self.q, status, detail

    def v2_q1_total_capacity(self):
        """V2 Q1 总容量违规：按 (airport, atype, flight_no) 统计人数 ≤ 座位数"""
        assert self.q == 1
        pax = defaultdict(int)
        for info in self.assign.values():
            pax[info['flight_key']] += 1
        violations = []
        for key, cnt in sorted(pax.items()):
            fl = self.flights.get(key)
            if fl is None:
                continue
            cap = AIRCRAFT[fl['atype']]['seats']
            if cnt > cap:
                over = (cnt - cap) / cap * 100
                violations.append('%s flight=%s 人数=%d 容量=%d 超载=%.1f%%'
                                  % (fl['airport'], key[1], cnt, cap, over))
        status = 'PASS' if not violations else 'FAIL'
        detail = '违规 %d 条' % len(violations)
        if violations:
            detail += '\n    ' + '\n    '.join(violations[:20])
        return 'V2 Q1总容量违规', status, detail

    def v3_v4_segment_capacity(self, label):
        """V3/V4 航段级容量违规：每航段在机人数 ≤ 座位数"""
        seg_loads, _ = self._seg_loads(self.assign, self.flights)
        violations = []
        for key, loads in sorted(seg_loads.items()):
            fl = self.flights.get(key)
            if fl is None:
                continue
            cap = AIRCRAFT[fl['atype']]['seats']
            for k, v in enumerate(loads):
                if v > cap:
                    over = (v - cap) / cap * 100
                    violations.append('%s %s seg=%d 在机=%d 容量=%d 超载=%.1f%%'
                                      % (fl['airport'], key, k, v, cap, over))
        status = 'PASS' if not violations else 'FAIL'
        detail = '违规 %d 条' % len(violations)
        if violations:
            detail += '\n    ' + '\n    '.join(violations[:20])
        return label, status, detail

    def v5_q3_time_window(self):
        """V5 Q3 时间窗违规：登机时刻 ≥ earliest_pickup 且到达时刻 ≤ latest_arrival"""
        assert self.q == 3
        violations = []
        per_day = defaultdict(int)
        for pid, info in sorted(self.assign.items()):
            d = self.demands.get(pid)
            key = info['flight_key']
            fl = self.flights.get(key)
            if d is None or fl is None:
                continue
            n_stops = len(fl['path'])
            pidx, didx = info['pidx'], info['didx']
            if pidx < 0 or pidx >= n_stops or didx < 0 or didx >= n_stops:
                violations.append('%s 停靠越界 pidx=%d didx=%d' % (pid, pidx, didx))
                continue
            dep_pickup = fl['schedule'][pidx][1]
            arr_delivery = fl['schedule'][didx][0]
            day = fl['schedule'][0][1]
            day_str = day.strftime('%Y-%m-%d') if day else 'N/A'
            if dep_pickup is None or arr_delivery is None:
                continue
            reasons = []
            if dep_pickup < d['earliest_pickup']:
                reasons.append('登机 %s < 最早接人 %s' % (
                    dep_pickup.strftime('%H:%M'),
                    d['earliest_pickup'].strftime('%m-%d %H:%M')))
            if arr_delivery > d['latest_arrival']:
                reasons.append('到达 %s > 最晚到达 %s' % (
                    arr_delivery.strftime('%H:%M'),
                    d['latest_arrival'].strftime('%m-%d %H:%M')))
            if reasons:
                violations.append('%s [%s] %s' % (pid, day_str, '; '.join(reasons)))
                per_day[day_str] += 1
        status = 'PASS' if not violations else 'FAIL'
        day_summary = ' '.join('%s:%d' % (k, v) for k, v in sorted(per_day.items()))
        detail = '违规 %d 条 (%s)' % (len(violations), day_summary or '无')
        if violations:
            detail += '\n    ' + '\n    '.join(violations[:20])
        return 'V5 Q3时间窗违规', status, detail

    @staticmethod
    def _feasible_days(demand):
        """计算需求可行日期集合（与 Q3DemandDayAllocator 口径一致）"""
        days = []
        for d in PLANNING_DATES:
            day_start = datetime(d.year, d.month, d.day, OPERATION_START_HOUR, 0)
            day_end = datetime(d.year, d.month, d.day, RETURN_DEADLINE_HOUR, 0)
            if demand['earliest_pickup'] < day_end and demand['latest_arrival'] > day_start:
                days.append(d.date())
        return days

    def v6_q3_day_match(self):
        """V6 Q3 日期匹配违规：航班日期必须属于需求可行日期集合"""
        assert self.q == 3
        violations = []
        for pid, info in sorted(self.assign.items()):
            d = self.demands.get(pid)
            key = info['flight_key']
            fl = self.flights.get(key)
            if d is None or fl is None:
                continue
            dep = fl['schedule'][0][1]
            if dep is None:
                continue
            flight_day = dep.date()
            feasible = self._feasible_days(d)
            if flight_day not in feasible:
                violations.append('%s 需求可行日=%s 航班日=%s' % (
                    pid,
                    ','.join(x.strftime('%m-%d') for x in feasible),
                    flight_day.strftime('%m-%d')))
        status = 'PASS' if not violations else 'FAIL'
        detail = '违规 %d 条' % len(violations)
        if violations:
            detail += '\n    ' + '\n    '.join(violations[:20])
        return 'V6 Q3日期匹配违规', status, detail

    def v7_od_consistency(self):
        """V7 OD 一致性：需求端点必须被航班路径访问，指定机场需求必须匹配航班机场"""
        violations = []
        checked = 0
        airport_set = set(AIRPORTS) | {'LAND'}
        for pid, info in sorted(self.assign.items()):
            d = self.demands.get(pid)
            key = info['flight_key']
            fl = self.flights.get(key)
            if d is None or fl is None:
                continue
            visited = set(fl['path'])
            o, de = d['origin_id'], d['destination_id']
            ok = True
            if o in airport_set and de.startswith('F'):
                checked += 1
                if de not in visited:
                    ok = False
                if o != 'LAND' and fl['airport'] != o:
                    ok = False
            elif o.startswith('F') and de in airport_set:
                checked += 1
                if o not in visited:
                    ok = False
                if de != 'LAND' and fl['airport'] != de:
                    ok = False
            elif o.startswith('F') and de.startswith('F'):
                checked += 1
                if o not in visited or de not in visited:
                    ok = False
            if not ok:
                violations.append('%s %s->%s 航班 %s 路径=%s' % (
                    pid, o, de, key, '->'.join(fl['path'])))
        status = 'PASS' if not violations else 'FAIL'
        ratio = (checked - len(violations)) / checked if checked else 1.0
        detail = '校验 %d 条, 一致率 %.2f%%, 违规 %d 条' % (
            checked, ratio * 100, len(violations))
        if violations:
            detail += '\n    ' + '\n    '.join(violations[:20])
        return 'V7 OD一致性(Q%d)' % self.q, status, detail

    def v8_direction_order(self):
        """V8 方向顺序：pickup_pos < delivery_pos 必须成立"""
        violations = []
        for pid, info in sorted(self.assign.items()):
            if info['pidx'] >= info['didx']:
                violations.append('%s pidx=%d didx=%d' % (
                    pid, info['pidx'], info['didx']))
        status = 'PASS' if not violations else 'FAIL'
        detail = '违规 %d 条' % len(violations)
        if violations:
            detail += '\n    ' + '\n    '.join(violations[:20])
        return 'V8 方向顺序(Q%d)' % self.q, status, detail

    def v9_duplicate(self):
        """V9 重复分配：同一人只能出现一次"""
        counts = Counter(self.assign.keys())
        dup = [pid for pid, n in counts.items() if n > 1]
        status = 'PASS' if not dup else 'FAIL'
        detail = '重复 %d 人' % len(dup)
        if dup:
            detail += '\n    ' + ', '.join(dup[:20])
        return 'V9 重复分配(Q%d)' % self.q, status, detail

    def v10_fuel(self):
        """V10 航油约束：每条路径必须通过燃油可行性检查（FuelChecker 口径）"""
        violations = []
        for key, fl in sorted(self.flights.items()):
            feasible, _, total_fuel = FuelChecker.check_fuel(
                fl['path'], fl['atype'], self.distances)
            if not feasible:
                tank = AIRCRAFT[fl['atype']]['tank']
                violations.append('%s 总油耗=%.1fkg 满箱=%.0fkg 加油点=%s' % (
                    key, total_fuel, tank, sorted(fl['refuel_at'])))
        status = 'PASS' if not violations else 'FAIL'
        detail = '违规 %d 条' % len(violations)
        if violations:
            detail += '\n    ' + '\n    '.join(violations[:20])
        return 'V10 航油约束(Q%d)' % self.q, status, detail

    def run_all(self):
        """运行该问题的全部验证项，返回 [(名称, 判定, 详情)]"""
        items = [self.v1_satisfaction()]
        if self.q == 1:
            items.append(self.v2_q1_total_capacity())
        elif self.q == 2:
            items.append(self.v3_v4_segment_capacity('V3 Q2航段级容量违规'))
        else:
            items.append(self.v3_v4_segment_capacity('V4 Q3航段级容量违规'))
            items.append(self.v5_q3_time_window())
            items.append(self.v6_q3_day_match())
        items.append(self.v7_od_consistency())
        items.append(self.v8_direction_order())
        items.append(self.v9_duplicate())
        items.append(self.v10_fuel())
        return items


# ==================== 主流程 ====================
def validate_q(q, distances):
    """验证单个问题，返回 (名称, 结果列表)"""
    output_dir = {1: OUTPUT_DIR_Q1, 2: OUTPUT_DIR_Q2, 3: OUTPUT_DIR_Q3}[q]
    demands = DataLoader.load_demands(q)
    assign = OutputReader.read_assignments(q, output_dir)
    flights = OutputReader.read_routes(q, output_dir)
    validator = Validator(q, demands, assign, flights, distances)
    return 'Q%d' % q, validator.run_all()


def main():
    """主函数：解析参数、执行验证、输出报告并返回退出码"""
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    parser = argparse.ArgumentParser(description='Q1/Q2/Q3 统一验证脚本')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--all', action='store_true', help='验证全部问题')
    group.add_argument('--q1', action='store_true', help='仅验证 Q1')
    group.add_argument('--q2', action='store_true', help='仅验证 Q2')
    group.add_argument('--q3', action='store_true', help='仅验证 Q3')
    args = parser.parse_args()

    targets = []
    if args.q1:
        targets = [1]
    elif args.q2:
        targets = [2]
    elif args.q3:
        targets = [3]
    else:
        targets = [1, 2, 3]

    distances = DataLoader.load_distances()
    all_items = []
    for q in targets:
        label, items = validate_q(q, distances)
        print('\n==================== %s 验证 ====================' % label)
        for name, status, detail in items:
            print('[%s] %s | %s' % (status, name, detail.split('\n')[0]))
            if detail.count('\n') > 0:
                print(detail)
            all_items.append((label, name, status, detail))

    n_fail = sum(1 for _, _, status, _ in all_items if status == 'FAIL')
    n_warn = sum(1 for _, _, status, _ in all_items if status == 'WARN')
    print('\n==================== 汇总 ====================')
    print('PASS: %d, WARN: %d, FAIL: %d' % (
        len(all_items) - n_fail - n_warn, n_warn, n_fail))

    os.makedirs(REPORT_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(REPORT_DIR, 'validation_report_%s.txt' % ts)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('统一验证报告 %s\n' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        f.write('=' * 60 + '\n')
        for label, name, status, detail in all_items:
            f.write('[%s] %s | %s\n' % (status, name, detail))
            f.write('-' * 60 + '\n')
        f.write('汇总: PASS=%d WARN=%d FAIL=%d\n' % (
            len(all_items) - n_fail - n_warn, n_warn, n_fail))
    print('报告已保存: %s' % report_path)

    return 1 if n_fail > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
