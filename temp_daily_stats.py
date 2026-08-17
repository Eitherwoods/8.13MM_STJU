# -*- coding: utf-8 -*-
import csv
from collections import defaultdict
from datetime import datetime

routes_path = r'D:\数模\26策联杯\8.13MM_STJU\output\Problem3\q3-routes.csv'
assignments_path = r'D:\数模\26策联杯\8.13MM_STJU\output\Problem3\q3-assignments.csv'

flights = []
with open(routes_path, 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        flights.append(row)

flight_groups = defaultdict(list)
for fo in flights:
    if fo['flight_no']:
        flight_groups[(fo['aircraft_id'], int(fo['flight_no']))].append(fo)

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
        })

assignments = []
with open(assignments_path, 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        assignments.append(row)

people_per_flight = defaultdict(int)
for a in assignments:
    if a['flight_no']:
        key = (a['aircraft_id'], int(a['flight_no']))
        people_per_flight[key] += 1

daily = defaultdict(lambda: {'flights': 0, 'time': 0, 'people': 0})
for fi in flight_info:
    date = fi['date']
    key = (fi['aircraft'], fi['flight'])
    daily[date]['flights'] += 1
    daily[date]['time'] += fi['duration']
    daily[date]['people'] += people_per_flight.get(key, 0)

for date in sorted(daily.keys()):
    d = daily[date]
    pf = d['people'] / d['flights'] if d['flights'] > 0 else 0
    print(date, 'flights=' + str(d['flights']), 'time=' + str(int(d['time'])), 'people=' + str(d['people']), 'per_flight=' + str(round(pf, 1)))
