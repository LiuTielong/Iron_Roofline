"""
这个demo是已知两条曲线的数据, 把它们绘制在一张图中。
本例是绘制了motivation图。
"""

verify_lengths = [1+1,2+1,3+1,4+1,5+1,6+1,8+1,10+1,12+1,14+1,16+1,18+1,20+1,25+1,30+1,35+1,40+1,45+1,50+1,60+1,70+1,80+1,]
gpu_speedup = [0.896,1.258,1.369,1.323,1.474,1.487,1.537,1.553,1.652,1.640,1.674,1.758,1.782,1.761,1.737,1.766,1.769,1.812,1.943,1.910,1.958,1.889,]
fpga_speedup = [ 1.524, 1.903, 1.969, 2.040, 2.144, 2.156, 2.246, 2.244, 2.267, 2.302, 2.313, 2.278, 2.230, 2.172, 1.924, 1.743, 1.555, 1.458, 1.354, 1.181, 1.049, 0.951,]

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 20  # 设置全局字体大小
plt.rcParams['font.family'] = 'Times New Roman'

plt.figure(figsize=(10, 6))
plt.plot(verify_lengths, gpu_speedup, marker='o', linestyle='-', color='r', label='GPU Speedup')
plt.plot(verify_lengths, fpga_speedup, marker='o', linestyle='-', color='b', label='FPGA Speedup')
# 下面标注两条曲线的最高点
plt.annotate(f"GPU: {max(gpu_speedup):.3f}", xy=(verify_lengths[gpu_speedup.index(max(gpu_speedup))], max(gpu_speedup)), 
             xytext=(5, 5), textcoords='offset points', arrowprops=dict(arrowstyle='->', color='r'), color='r')
plt.annotate(f"FPGA: {max(fpga_speedup):.3f}", xy=(verify_lengths[fpga_speedup.index(max(fpga_speedup))], max(fpga_speedup)),
             xytext=(5, -20), textcoords='offset points', arrowprops=dict(arrowstyle='->', color='b'), color='b')

plt.xlabel("Verify Lengths")
plt.ylabel("Speedup")
plt.title("Speedup Comparison between GPU and FPGA")
plt.legend()
plt.grid(True)
plt.show()