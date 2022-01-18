# import numpy as np
import matplotlib.pyplot as plt

#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False


x = [12, 14, 16, 18, 20]
y = [4.34, 6.07, 6.38, 6.48, 6.78]
y2 = [3.83, 3.76, 4.17, 4.51, 4.57]
y3 = [2.74, 3.68, 3.76, 4.01, 4.27]


plt.figure()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.plot(x, y, 'ro-', label='[9.42,1200]')
plt.plot(x, y2, 'gx-', label='[7.85,800]')
plt.plot(x, y3, 'b>-', label='[6.28,1000]')
plt.xlabel("Downward displacement (mm) ",fontsize=15)
plt.ylabel("Grinding efficiency (mm^3/s)",fontsize=15)
plt.title("Effect of downward displacement on grinding efficiency",fontsize=15)

# 设置数字标签
# for a, b in zip(x, y):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

plt.legend(loc=2)
plt.show()
