# import numpy as np
import matplotlib.pyplot as plt

#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False


x = [5.50, 6.28, 7.07, 7.46, 7.85, 8.25, 8.64, 9.42, 10.21]
y = [0.145, 0.215, 0.2, 0.195, 0.2, 0.26, 0.29, 0.33, 0.315]
y2 = [0.185, 0.245, 0.24, 0.275, 0.265, 0.28, 0.28, 0.39, 0.335]
y3 = [0.17, 0.235, 0.25, 0.26, 0.275, 0.28, 0.295, 0.35, 0.33]


plt.figure()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.plot(x, y, 'ro-', label='[14,1000]')
plt.plot(x, y2, 'gx-', label='[18,900]')
plt.plot(x, y3, 'b>-', label='[20,1100]')
plt.xlabel("Linear velocity of abrasive belt (m/s) ",fontsize=15)
plt.ylabel("Grinding depth (mm)",fontsize=15)
plt.title("Effect of belt linear velocity on grinding depth",fontsize=15)

# 设置数字标签
# for a, b in zip(x, y):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

plt.legend(loc=2)
plt.show()
