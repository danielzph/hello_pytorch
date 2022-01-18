# import numpy as np
import matplotlib.pyplot as plt

#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False


x = [12, 14, 16, 18, 20]
y = [0.21, 0.295, 0.31, 0.315, 0.33]
y2 = [0.28, 0.275, 0.305, 0.33, 0.335]
y3 = [0.16, 0.215, 0.22, 0.235, 0.25]


plt.figure()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.plot(x, y, 'ro-', label='[9.42,1200]')
plt.plot(x, y2, 'gx-', label='[7.85,800]')
plt.plot(x, y3, 'b>-', label='[6.28,1000]')
plt.xlabel("Downward displacement (mm) ",fontsize=15)
plt.ylabel("Grinding depth (mm)",fontsize=15)
plt.title("Effect of downward displacement on grinding depth",fontsize=15)

# 设置数字标签
# for a, b in zip(x, y):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

plt.legend(loc=2)
plt.show()
