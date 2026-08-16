# -*- coding: utf-8 -*-
import os

# 读取所有部分
parts = []
for i in range(1, 6):
    with open(r'D:\数模\26策联杯\8.13MM_STJU\body\q3_part' + str(i) + '.tex', 'r', encoding='utf-8') as f:
        parts.append(f.read())

# 合并
new_q3 = '\n'.join(parts)

# 读取main.tex
with open(r'D:\数模\26策联杯\8.13MM_STJU\body\main.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# 找到问题三的起始和结束位置
q3_start = content.find('% ==================== 问题三 ====================')
q3_end = content.find('% ==================== 模型评价与推广 ====================')

if q3_start == -1:
    print('ERROR: Could not find Q3 start marker')
elif q3_end == -1:
    print('ERROR: Could not find Q3 end marker')
else:
    # 替换
    new_content = content[:q3_start] + new_q3 + '\n\n' + content[q3_end:]
    
    # 保存
    with open(r'D:\数模\26策联杯\8.13MM_STJU\body\main.tex', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print('SUCCESS: main.tex updated')
    print(f'Old Q3 length: {q3_end - q3_start} chars')
    print(f'New Q3 length: {len(new_q3)} chars')
