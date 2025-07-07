"""
这里是对我的方法2的一个消融实验。
表6.xlsx是三种建树方法的AAT数据。
"""

import pandas as pd
import sys
sys.path.append("./")
sys.path.append("../")
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 20  # 设置全局字体大小
plt.rcParams['font.family'] = 'Times New Roman'
import numpy as np
from modeling.parse_args import parse_args
from modeling.modeling_longspec import longspec_draft_cycles_comp
from modeling.modeling_llama3 import llama3_cycles_comp
from roofline.draw import draw_combined_model

# step 1: 绘制三条AAT曲线的形状
data_dir = "./Data/表6.xlsx"
method_name = ["Tree Search", "Width", "Depth"]
df = pd.read_excel(data_dir, skiprows=1, header=None)
gammas = df.iloc[:, 0].tolist()
verify_lens = [gamma + 1 for gamma in gammas]           # 大模型校验时的输入数据长度
tree_shapes_all = df.iloc[:, 4:7].values.T.tolist()  # 三种建树方法的树形状数据，存为2D列表
AATs_all = df.iloc[:, 1:4].values.T.tolist()             # 三种建树方法的AAT数据，存为2D列表, prefill长度为128的那一列我不要了
colors = [(180/255, 199/255, 231/255), (248/255, 203/255, 173/255), (197/255, 224/255, 180/255)]
for i, aat_curve in enumerate(AATs_all):
    plt.plot(verify_lens, aat_curve, linestyle='-', label=f'{method_name[i]}', linewidth=3, color=colors[i])
plt.xlabel("Verify Lengths")
plt.ylabel("AAT")
# plt.title("AAT Curves for Different Methods")
plt.legend()
plt.grid(True)
plt.subplots_adjust(bottom=0.15)  # 将下边距调整为适当的比例
plt.savefig("Figures/longspec_method2/aat_curves.pdf")
plt.show()

aat1 = AATs_all[0]  # Tree Search
aat2 = AATs_all[1]  # Width
aat3 = AATs_all[2]  # Depth
# 计算每个方法的平均AAT
aat1_mean = np.mean(aat1)
aat2_mean = np.mean(aat2)
aat3_mean = np.mean(aat3)
print(f"Average AAT for Tree Search: {aat1_mean:.2f}")
print(f"Average AAT for Width: {aat2_mean:.2f}")
print(f"Average AAT for Depth: {aat3_mean:.2f}")

# step 2：绘制三条加速比曲线，并且在曲线中标出最大点
base_speed = 35.66          # tokens/s, 自回归生成速度
parser = parse_args()
parser.add_argument("--avg_accepted_tokens",    type=int,   default=4   ,     help="the average number of accepted tokens per iteration."         )
parser.add_argument("--gamma",                  type=int,   default=1,        help="it's similar to total_tokens, (depth+1) in eagle algorithm."  )
parser.add_argument("--tree_shape", nargs="+",  type=int, default=[4, 16, 16, 16, 16], help="the tree shape of the draft token tree." )
args = parser.parse_args()
args.batch_size = 1
args.prompt_len = 1024
for i in range(3):
    tree_shapes = tree_shapes_all[i]
    AATs = AATs_all[i]
    verify_times = []
    draft_times = []
    for total_token, AAT, tree_shape in zip(verify_lens, AATs, tree_shapes):
        aat = int(AAT)
        args.tree_shape = [int(x) for x in tree_shape.split()]
        
        # 小模型的draft阶段
        _, _, _, fused_draft_cycles = longspec_draft_cycles_comp(args, input_len=aat, kv_len=args.prompt_len, method="tree")
        
        # 大模型的verify阶段
        _, _, _, fused_verify_cycles = llama3_cycles_comp(args, input_len=total_token, kv_len=args.prompt_len)
        
        draft_time = fused_draft_cycles / (args.clock_frequency * 1e6)
        verify_time = fused_verify_cycles / (args.clock_frequency * 1e6)
        verify_times.append(verify_time)
        draft_times.append(draft_time)
    draw_combined_model(verify_lengths=verify_lens, verify_times=verify_times, draft_times=draft_times,
                        accepted_lengths=AATs, save_path=f"Figures/longspec_method2/{method_name[i]}.png", batch_size=1, ori_x=None, naive_x=None)
    