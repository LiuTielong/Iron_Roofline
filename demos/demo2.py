"""
这个demo是已知两条曲线的数据, 把它们绘制在一张图中。
本例是绘制了motivation图。
"""
import pandas as pd

data_dir = "D:/PHD/HPCA2025/统一的投机采样研究/两套实验数据/128乘128/表1.xlsx"
df = pd.read_excel(data_dir, skiprows=1, header=None)
verify_lengths = df.iloc[:, 0].tolist()     # 第一列是验证长度
gpu_speedup = df.iloc[:, 1].tolist()        # 第二列是GPU加速比
fpga_speedup = df.iloc[:, 2].tolist()       # 第三列是FPGA加速比
aat = df.iloc[:, 3].tolist()                # 平均接受长度

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 20  # 设置全局字体大小
plt.rcParams['font.family'] = 'Times New Roman'

plt.figure(figsize=(10, 6))
# 主坐标轴：绘制 GPU 和 FPGA 加速比
ax = plt.gca()
ax.plot(verify_lengths, gpu_speedup, linestyle='-', color=(180/255, 199/255, 231/255), label='Speedup On GPU', linewidth=3)
ax.plot(verify_lengths, fpga_speedup, linestyle='-', color=(248/255, 203/255, 173/255), label='Speedup On FPGA', linewidth=3)
ax.set_xlabel("Verification Tokens")
ax.set_ylabel("Speedup")
ax.grid(True)

# 在右侧添加一个新的坐标轴，绘制 aat 曲线
ax2 = ax.twinx()
ax2.plot(verify_lengths, aat, linestyle='--', color=(197/255, 224/255, 180/255), label='AAT', linewidth=3)
ax2.set_ylabel("AAT")

# 图例
lines_1, labels_1 = ax.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax.legend(lines_1+lines_2, labels_1+labels_2, 
          loc='upper center', bbox_to_anchor=(0.5, 1.17), 
          ncol=3, frameon=True, 
          columnspacing=1.5, handlelength=2.0, handletextpad=0.5)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # 将下边距调整为适当的比例
plt.savefig("./figures/paper/motivation.pdf")
plt.show()