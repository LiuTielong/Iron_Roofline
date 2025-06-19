"""
我针对llama-3.1-8B模型, 使用longspec算法, 对多个任务的AAT进行测试。在获得AAT曲线之后, 观察这些曲线的AAT有什么相似之处。
数据来自表5.xlsx. prefill length=1024, generation length=1024.
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


# step 1：先绘制这些曲线的形状
data_dir = "./Data/表5.xlsx"
task_name = ["gov_report", "qmsum", "multi_news", "lcc", "repobench-p"]
df = pd.read_excel(data_dir, skiprows=1, header=None)
gammas = df.iloc[:, 0].tolist()
verify_lens = [gamma + 1 for gamma in gammas]           # 大模型校验时的输入数据长度
tree_shapes = df.iloc[:, 1].tolist()                   # 树形状数据，存为1D列表
AATs_all = df.iloc[:, 2:].values.T.tolist()             # AAT数据，存为2D列表, prefill长度为128的那一列我不要了
for i, aat_curve in enumerate(AATs_all):
    plt.plot(verify_lens, aat_curve, marker='o', label=f'{task_name[i]}')
plt.xlabel("Verify Lengths")
plt.ylabel("AAT")
plt.title("AAT Curves for Different Tasks")
plt.legend()
plt.grid(True)
plt.show()


# step 2: 对这些AAT分别测试加速比最高的点
base_speed = 35.66          # tokens/s, 自回归生成速度
parser = parse_args()
parser.add_argument("--avg_accepted_tokens",    type=int,   default=4   ,     help="the average number of accepted tokens per iteration."         )
parser.add_argument("--gamma",                  type=int,   default=1,        help="it's similar to total_tokens, (depth+1) in eagle algorithm."  )
parser.add_argument("--tree_shape", nargs="+",  type=int, default=[4, 16, 16, 16, 16], help="the tree shape of the draft token tree." )
args = parser.parse_args()
args.batch_size = 1
args.prompt_len = 1024

for AATs in AATs_all:
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
    
    for verify_time, draft_time, aat in zip(verify_times, draft_times, AATs):
        performance = aat / (verify_time + draft_time)
        speedup = performance / base_speed
        print(f"{aat}: {speedup}")
    print("---------------------------------------------------------------")


# step 3: 对这些AAT做一个verify length-wise的几何平均，然后找加速比最高的点
AAT_average = np.mean(AATs_all, axis=0)
for total_token, AAT, tree_shape in zip(verify_lens, AAT_average, tree_shapes):
    aat = int(AAT)
    args.tree_shape = [int(x) for x in tree_shape.split()]
    # 小模型的draft阶段
    _, _, _, fused_draft_cycles = longspec_draft_cycles_comp(args, input_len=aat, kv_len=args.prompt_len, method="tree")
    # 大模型的verify阶段
    _, _, _, fused_verify_cycles = llama3_cycles_comp(args, input_len=total_token, kv_len=args.prompt_len)
    draft_time = fused_draft_cycles / (args.clock_frequency * 1e6)
    verify_time = fused_verify_cycles / (args.clock_frequency * 1e6)
    
    performance = AAT / (verify_time + draft_time)
    speedup = performance / base_speed
    print(f"Verify length: {total_token}, Average AAT: {AAT:.3f}, Speedup: {speedup}")

