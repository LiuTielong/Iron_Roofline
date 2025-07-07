"""
在已知很多模型在不同上下文长度的AAT之后, 绘制一张大图, 记录不同(模型, 上下文长度)的最高加速比。
对于LongSpec, 用来测试的模型包括: Vicuna-7B-v1.5, Vicuna-13B-v1.5, LongChat-7B, LongChat-13B.
使用表8.xlsx中的数据。
"""
import pandas as pd
import sys
sys.path.append("./")
sys.path.append("../")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from roofline.draw import draw_roofline, draw_roofline_discount, draw_acc, draw_combined_model
from modeling.parse_args import parse_args
from modeling.modeling_longspec import longspec_draft_cycles_comp
from modeling.modeling_llama3 import llama3_cycles_comp
from modeling.modeling_eagle3 import eagle3_cycles_comp, eagle3_draft_cycles_comp
plt.rcParams['font.size'] = 15  # 设置全局字体大小
plt.rcParams['font.family'] = 'Times New Roman'

def fitting(x_data, y_data):
    """
    输入一组 (x, y) 数据点，拟合出对数函数 y = A + B * ln(x - C).
    返回拟合参数 A, B, C.

    参数:
        x_data (array-like): x 坐标数组，长度至少为3
        y_data (array-like): y 坐标数组，长度与 x_data 相同

    返回:
        A, B, C (float): 拟合的参数
    """
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    if len(x_data) < 3:
        raise ValueError("至少需要三个数据点来拟合 A, B, C")
    
    # 残差函数
    def residuals(params):
        A, B, C = params
        if np.any(x_data - C <= 0):
            return np.full_like(y_data, 1e6)
        return A + B * np.log(x_data - C) - y_data

    # 初始猜测
    initial_guess = [1.0, 1.0, min(x_data) - 0.5]

    # 设置边界，确保 x - C > 0
    bounds = ([-np.inf, -np.inf, -np.inf],
              [np.inf, np.inf, min(x_data) - 1e-6])

    # 拟合
    result = least_squares(residuals, initial_guess, bounds=bounds)

    if not result.success:
        raise RuntimeError("拟合失败：" + result.message)

    A, B, C = result.x
    return A, B, C



"""1. 读取数据并进行曲线拟合"""
data_dir = "./Data/表8.xlsx"
df_longspec = pd.read_excel(data_dir, skiprows=2, header=None, sheet_name="Sheet1")
df_eagle = pd.read_excel(data_dir, skiprows=2, header=None, sheet_name="Sheet2")
df_tree_shape = pd.read_excel(data_dir, skiprows=2, header=None, sheet_name="Sheet3")
tree_shapes = df_tree_shape.iloc[:, 1].tolist()
gammas = df_longspec.iloc[:, 1].tolist()
gammas = gammas[0:8]    # 对所有模型都是这8个gammas
verify_lens = [gamma + 1 for gamma in gammas]           # 大模型校验时的输入数据长度
aat_longspec = df_longspec.iloc[:, 2:6].values        # LongSpec的AAT数据, 有4列，表示4个模型
aat_eagle = df_eagle.iloc[:, 2:4].values              # Eagle的AAT数据, 有2列，表示2个模型

# 先针对longspec弄
LongSpec_Fit_AAT = [[] for _ in range(4)]  
for i in range(7):
    for j in range(4):
        aat = aat_longspec[i*8:(i+1)*8, j]
        A, B, C = fitting(verify_lens, aat)
        # 然后绘制拟合后的曲线以及原始的数据点，x轴范围从2到85
        x_fit = np.linspace(2, 85, 84)
        y_fit = A + B * np.log(x_fit - C)
        # 加入一点噪声，y_fit增加一个随机噪声
        y_fit += np.random.normal(0, 0.1, size=y_fit.shape)  # 添加噪声
        LongSpec_Fit_AAT[j].append(y_fit)
        # plt.figure(figsize=(8, 5))
        # plt.plot(x_fit, y_fit, label='Fitting Curve', color='blue')
        # plt.scatter(verify_lens, aat, color='red', label='Original Data Points')
        # plt.xlabel("Verify Lengths")
        # plt.ylabel("AAT")
        # plt.legend()
        # plt.show()

# 然后针对eagle弄
Eagle_Fit_AAT = [[] for _ in range(2)]
for i in range(3):
    for j in range(2):
        aat = aat_eagle[i*8:(i+1)*8, j]
        A, B, C = fitting(verify_lens, aat)
        # 然后绘制拟合后的曲线以及原始的数据点，x轴范围从2到85
        x_fit = np.linspace(2, 85, 84)
        y_fit = A + B * np.log(x_fit - C)
        # 加入一点噪声，y_fit增加一个随机噪声
        y_fit += np.random.normal(0, 0.1, size=y_fit.shape)  # 添加噪声
        Eagle_Fit_AAT[j].append(y_fit)


"""2. 求解每种配置的最佳verification tokens和对应的加速比"""
# LongSpec
parser = parse_args()
parser.add_argument("--avg_accepted_tokens",    type=int,   default=4   ,     help="the average number of accepted tokens per iteration."         )
parser.add_argument("--gamma",                  type=int,   default=1,        help="it's similar to total_tokens, (depth+1) in eagle algorithm."  )
parser.add_argument("--tree_shape", nargs="+",  type=int, default=[4, 16, 16, 16, 16], help="the tree shape of the draft token tree." )
args = parser.parse_args()
args.batch_size = 1
args.mm_parallel_m = 128
model_names = ["Vicuna-7B-v1.5", "Vicuna-13B-v1.5", "LongChat-7B", "LongChat-13B"]
context_lengths = [128, 256, 512, 1024, 4096, 8192, 16384]
for j, model_name in enumerate(model_names):
    for i, context_len in enumerate(context_lengths):
        verify_lens = np.linspace(2, 85, 84)
        AATs = LongSpec_Fit_AAT[j][i]
        # print(AATs[0:10])
        verify_times = []
        draft_times = []
        performances = []
        args.prompt_len = context_len
        if model_name == "Vicuna-7B-v1.5":
            # 模型的超参数
            args.hidden_size = 4096
            args.intermediate_size = 11008
            args.num_layers = 32
            args.num_heads = 32
            args.head_dim = 128
            args.vocab_size = 32000
            args.kv_scale = 1.0
        elif model_name == "Vicuna-13B-v1.5":
            args.hidden_size = 5120
            args.intermediate_size = 13824
            args.num_layers = 40
            args.num_heads = 40
            args.head_dim = 128
            args.vocab_size = 32000
            args.kv_scale = 1.0  # 不进行GQA
        elif model_name == "LongChat-7B":
            # 模型的超参数, 修改成vicuna-v1.5-7B的配置
            args.hidden_size = 4096
            args.intermediate_size = 11008
            args.num_layers = 32
            args.num_heads = 32
            args.head_dim = 128
            args.vocab_size = 32000
            args.kv_scale = 1.0  # 不进行GQA
        elif model_name == "LongChat-13B":
            args.hidden_size = 5120
            args.intermediate_size = 13824
            args.num_layers = 40
            args.num_heads = 40
            args.head_dim = 128
            args.vocab_size = 32000
            args.kv_scale = 1.0  # 不进行GQA
        for total_token, AAT, tree_shape in zip(verify_lens, AATs, tree_shapes):
            aat = int(AAT)
            args.tree_shape = [int(x) for x in tree_shape.split()]
            # 小模型的draft阶段
            _, _, _, fused_draft_cycles = longspec_draft_cycles_comp(args, input_len=aat, kv_len=args.prompt_len, method="tree")
            # 大模型的verify阶段
            _, _, _, fused_verify_cycles = llama3_cycles_comp(args, input_len=total_token, kv_len=args.prompt_len)
            draft_time = fused_draft_cycles / (args.clock_frequency * 1e6)
            verify_time = fused_verify_cycles / (args.clock_frequency * 1e6)
            performance = AAT / (verify_time + draft_time)
            performances.append(performance)
            verify_times.append(verify_time)
            draft_times.append(draft_time)
        # print(f"Model: {model_name}, Context Length: {context_len}, Best Verification Token: {verify_lens[np.argmax(performances)]}, Max Speedup: {max(performances):.3f} tokens/s")

# EAGLE-3
parser.add_argument("--total_token",            type=int,   default=60,       help="the maxinum number of new generated tokens that would be verified by LLM.")
parser.add_argument("--top_k",                  type=int,   default=10,       help="the number of generated tokens per SSM forward process."                  )
parser.add_argument("--depth",                  type=int,   default=6,        help="the depth of draft token tree - 2."                                       )
model_names = ["Llama-3.1-8B", "Vicuna-1.3-13B"]
context_lengths = [128, 512, 2048]
draft_depths = df_tree_shape.iloc[:, 2].tolist()
draft_topks = df_tree_shape.iloc[:, 3].tolist()
for j, model_name in enumerate(model_names):
    for i, context_len in enumerate(context_lengths): 
        verify_lens = np.linspace(2, 85, 84)
        AATs = Eagle_Fit_AAT[j][i]
        # print(AATs[0:10])
        verify_times = []
        draft_times = []
        performances = []
        args.prompt_len = context_len
        if model_name == "Llama-3.1-8B":
            # 模型的超参数
            args.hidden_size = 4096
            args.intermediate_size = 14336
            args.num_layers = 32
            args.num_heads = 32
            args.head_dim = 128
            args.vocab_size = 128256
            args.kv_scale = 0.25  # 进行GQA
        elif model_name == "Vicuna-1.3-13B":
            args.hidden_size = 5120
            args.intermediate_size = 13824
            args.num_layers = 40
            args.num_heads = 40
            args.head_dim = 128
            args.vocab_size = 32000
            args.kv_scale = 1.0  # 不进行GQA
        for total_token, AAT, depth, top_k in zip(verify_lens, AATs, draft_depths, draft_topks):
            aat = int(AAT)
            args.depth = depth
            args.top_k = top_k
            # 小模型的draft阶段
            _, _, _, fused_draft_cycles = eagle3_draft_cycles_comp(args, input_len=aat, kv_len=args.prompt_len)
            # 大模型的verify阶段
            _, _, _, fused_verify_cycles = llama3_cycles_comp(args, input_len=total_token, kv_len=args.prompt_len)
            draft_time = fused_draft_cycles / (args.clock_frequency * 1e6)
            verify_time = fused_verify_cycles / (args.clock_frequency * 1e6)
            performance = AAT / (verify_time + draft_time)
            performances.append(performance)
            verify_times.append(verify_time)
            draft_times.append(draft_time)
        # print('[' + ', '.join(f"{v:.3f}" for v in performances) + ']')
        # print(f"Model: {model_name}, Context Length: {context_len}, Best Verification Token: {verify_lens[np.argmax(performances)]}, Max Speedup: {max(performances):.3f}")


"""3. 绘制一张大图记录不同配置的最佳verification tokens和最高加速比"""
import numpy as np
import matplotlib.pyplot as plt

# 3.1 绘制柱状图
# Speedup1：16个数据，每4个为一组（4组）
Speedup1 = [
    2.446, 2.495, 2.630, 2.459,      # Vicuna-7B
    2.761, 2.639, 2.995, 3.002,      # Vicuna-13B
    2.426, 2.562, 2.776, 2.932,      # LongChat-7B
    2.470, 2.493, 2.803, 2.919,      # LongChat-13B
]
Verification_tokens1 = [
    10,	9,	26,	14,
    10,	16,	33,	41,
    11,	16,	40,	44,
    9,	12,	33,	36,
]
group_count1 = 4    # 4组
bar_count1 = 4      # 每组4根柱子
width = 0.2         # 每根柱子的宽度
group_positions1 = np.arange(group_count1)  # 例如：[0, 1, 2, 3]

plt.figure(figsize=(12, 6))
colors1 = [(180/255, 199/255, 231/255), (248/255, 203/255, 173/255), (197/255, 224/255, 180/255), (255/255, 230/255, 153/255)]
labels1 = ['128', '512', '4096', '16384']
colors2 = [(180/255, 199/255, 231/255), (248/255, 203/255, 173/255), (218/255, 197/255, 225/255)]
labels2 = ['128', '512', '2048']

# Speedup2：6个数据，每3个为一组（2组）
Speedup2 = [
    3.25, 2.884, 2.469,   # Llama-3.1-8B
    4.07, 3.904, 3.7,     # Vicuna-1.3-13B
]
Verification_tokens2 = [
    12,	14,	16,
    13,	14,	25,
]
group_count2 = 2    # 2组
bar_count2 = 3      # 每组3根柱子

for i in range(bar_count1):
    # 取每组对应的第i个数据
    group_values = [Speedup1[bar_count1 * j + i] for j in range(group_count1)]
    positions = group_positions1 + i * width
    plt.bar(positions, group_values, width=width, color=colors1[i],
            label=f'{labels1[i]}')

# 设置一个偏移量，使得Speedup2的柱子绘制在Speedup1右侧
offset = group_count1+0.2   # group_count1取最后一组的右侧，再留一点空隙
group_positions2 = np.arange(group_count2) * 0.8 + offset  # 例如：[4, 5] 当group_count1=4

for i in range(bar_count2):
    group_values = [Speedup2[bar_count2 * j + i] for j in range(group_count2)]
    positions = group_positions2 + i * width
    plt.bar(positions, group_values, width=width, color=colors2[i],
            label=f'{labels2[i]}')

# 合并两部分的x轴刻度标签
xtick_positions = np.concatenate((
    group_positions1 + width * (bar_count1 - 1) / 2,
    group_positions2 + width * (bar_count2 - 1) / 2
))
xtick_labels = ["Vicuna-1.5-7B", "Vicuna-1.5-13B", "LongChat-7B", "LongChat-13B",
                "Llama-3.1-8B", "Vicuna-1.3-13B"]

plt.xticks(xtick_positions, xtick_labels)
plt.xlabel("                                                               (LongSpec)                                                                            (EALGE-3)                      ")
plt.ylabel("Speedup")

# 3.2 隔开longspec和eagle-3的柱子
# 计算Speedup1最后一组柱子的最右边位置
rightmost_speedup1 = group_positions1[-1] + (bar_count1 - 1) * width
min_speedup2 = group_positions2[0]
line_x = (rightmost_speedup1 + min_speedup2) / 2
plt.axvline(x=line_x, color='grey', linestyle='--', linewidth=2)

# 3.3 绘制 Verification_tokens 折线
ax1 = plt.gca()
ax1.set_ylim(0, 5)   # 设置主轴 (Speedup) 的 y 范围
ax2 = ax1.twinx()
ax2.set_ylabel("Verification Tokens")
ax2.set_ylim(-30, 50)  # 设置次轴 (Verification Tokens) 的 y 范围

# 绘制 Speedup1 组的 Verification_tokens 折线
for m in range(group_count1):
    # 每组中共有 bar_count1 根柱子
    # x 坐标为：group_positions1[m] + k*width (k=0,...,bar_count1-1)
    xs = group_positions1[m] + np.arange(bar_count1) * width
    # 对应 y 值取 Verification_tokens1 中连续 bar_count1 个数据
    ys = Verification_tokens1[m * bar_count1:(m + 1) * bar_count1]
    ax2.plot(xs, ys, marker='o', markersize=4, linestyle='-', linewidth=2, color=(246/255,92/255,76/255))

# 绘制 Speedup2 组的 Verification_tokens 折线
for m in range(group_count2):
    xs = group_positions2[m] + np.arange(bar_count2) * width
    ys = Verification_tokens2[m * bar_count2:(m + 1) * bar_count2]
    ax2.plot(xs, ys, marker='o', markersize=4, linestyle='-', linewidth=2, color=(246/255,92/255,76/255))

# 去除重复图例
handles, labels = ax1.get_legend_handles_labels()
unique_labels = []
unique_handles = []
for lab, han in zip(labels, handles):
    if lab not in unique_labels:
        unique_labels.append(lab)
        unique_handles.append(han)
# 如果需要调整顺序，可以按照新顺序重新排列：
new_order = [0, 1, 4, 2, 3]  # 根据实际去重后元素的顺序
ordered_labels = [unique_labels[i] for i in new_order]
ordered_handles = [unique_handles[i] for i in new_order]
ax1.legend(ordered_handles, ordered_labels,
           loc='upper center',
           bbox_to_anchor=(0.5, 1.12),
           ncol=5, frameon=True,
           columnspacing=3, handlelength=3.0, handletextpad=1.5)

plt.tight_layout()
plt.savefig("Figures/all_models_speedup.pdf")
plt.show()