# import numpy as np
import matplotlib.pyplot as plt

#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False


x = [800, 900, 950, 1000, 1050, 1100, 1200]
y = [3.493, 3.477, 3.523, 3.677, 3.623, 3.705, 3.927]
y2 = [5.782, 5.987, 6.026, 5.954, 5.858, 6.051, 6.480]
y3 = [5.448, 5.685, 5.866, 5.786, 5.947, 6.238, 6.378]


plt.figure()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.plot(x, y, 'ro-', label='[6.28,14]')
plt.plot(x, y2, 'gx-', label='[9.42,18]')
plt.plot(x, y3, 'b>-', label='[10.21,20]')
plt.xlabel("Feed speed (mm/min) ",fontsize=15)
plt.ylabel("Grinding efficiency (mm^3/s)",fontsize=15)
plt.title("Effect of feed speed on grinding efficiency",fontsize=15)

# 设置数字标签
# for a, b in zip(x, y):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

plt.legend(loc=2)
plt.show()
