"""
这个demo用来绘制最原始的roofline模型——不带标签, 只有形状。
"""
# 画一条曲线图：横坐标为t，纵坐标为P。t的范围从1到20，当t<=10，P(t)=t；当t>10, P(t)=10.
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

# 定义 t 的范围从1到20
t = np.arange(1, 21)
# 当 t <= 10 时，P(t)=t+5；当 t > 10 时，P(t)=10
P = np.where(t <= 10, t, 10)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(t, P, marker='o', linestyle='-')

# 去除默认的刻度标签
ax.set_xticks([])
ax.set_yticks([])

# 将左右、上下边框隐藏或调整
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# 将 x、y 轴放在数据原点位置（也可以根据需要调整位置）
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))

# 在 x 轴和 y 轴末尾添加箭头
# x轴箭头
ax.annotate("", xy=(1, 0), xytext=(0, 0),
            xycoords=('axes fraction','data'),
            textcoords=('axes fraction','data'),
            arrowprops=dict(arrowstyle="->", color='black', lw=1.5))
# y轴箭头
ax.annotate("", xy=(0, 1), xytext=(0, 0),
            xycoords=('data','axes fraction'),
            textcoords=('data','axes fraction'),
            arrowprops=dict(arrowstyle="->", color='black', lw=1.5))

plt.xlabel('t', fontsize=14)
plt.ylabel('P(t)', fontsize=14)
plt.title('Original Performance', fontsize=16)
plt.grid(False)  # 去除网格

# 设置 y 轴从 0 开始
ax.set_ylim(0, P.max() + 1)

# 标出 t=1 的点，并标注为 P(1)
P1 = P[0]  # 因为 t[0] 对应 t=1
plt.plot(1, P1, marker='o', markersize=6, color='red')
plt.annotate("P(1)", xy=(1, P1), xytext=(0, P1),
             arrowprops=dict(arrowstyle="->", color='red'),
             fontsize=12, color='red')



# plt.legend()
plt.show()
