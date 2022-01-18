# import numpy as np
import matplotlib.pyplot as plt

#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False


x = [5.50, 6.28, 7.07, 7.46, 7.85, 8.25, 8.64, 9.42, 10.21]
y = [2.487, 3.677, 3.422, 3.337, 3.422, 4.438, 4.944, 5.618, 5.366]
y2 = [2.863, 3.782, 3.706, 4.240, 4.087, 4.316, 4.316, 5.987, 5.155]
y3 = [3.234, 4.460, 4.739, 4.927, 5.28, 5.302, 5.582, 6.610, 6.238]


plt.figure()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.plot(x, y, 'ro-', label='[14,1000]')
plt.plot(x, y2, 'gx-', label='[18,900]')
plt.plot(x, y3, 'b>-', label='[20,1100]')
plt.xlabel("Linear velocity of abrasive belt (m/s) ",fontsize=15)
plt.ylabel("Grinding efficiency (mm^3/s)",fontsize=15)
plt.title("Effect of belt linear velocity on grinding efficiency",fontsize=15)

# 设置数字标签
# for a, b in zip(x, y):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

plt.legend(loc=2)
plt.show()
