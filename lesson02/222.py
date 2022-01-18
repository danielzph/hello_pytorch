# import numpy as np
import matplotlib.pyplot as plt

#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False


x = [800, 900, 950, 1000, 1050, 1100, 1200]
y = [0.255, 0.225, 0.215, 0.215, 0.2, 0.195, 0.19]
y2 = [0.425, 0.39, 0.37, 0.35, 0.325, 0.32, 0.315]
y3 = [0.4, 0.37, 0.36, 0.34, 0.33, 0.33, 0.31]


plt.figure()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.plot(x, y, 'ro-', label='[6.28,14]')
plt.plot(x, y2, 'gx-', label='[9.42,18]')
plt.plot(x, y3, 'b>-', label='[10.21,20]')
plt.xlabel("Feed speed (mm/min) ",fontsize=15)
plt.ylabel("Grinding depth (mm)",fontsize=15)
plt.title("Effect of feed speed on grinding depth",fontsize=15)

# 设置数字标签
# for a, b in zip(x, y):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

plt.legend(loc=1)
plt.show()
