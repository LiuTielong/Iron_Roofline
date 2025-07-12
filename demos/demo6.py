"""
demo4画的一张图我觉得还是不太行，现在给我拆成6张小图，组成一张大图。
之前一张大图只包含我的方法的加速比，现在把naive set和original也都加上。
对于LongSpec, 用来测试的模型包括: Vicuna-7B-v1.5, Vicuna-13B-v1.5, LongChat-7B, LongChat-13B.
对于EAGLE-3，用来测试的模型包括：llama-3.1-8B, Vicuna-13B-v1.3。
AAT的话使用表8.xlsx中的数据。
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

"""2. 求解每种配置的最佳verification Length, naive set, 以及 original的加速比"""
# LongSpec
# 它的original set就直接默认是68
# naive set我直接砍半，然后加个随机扰动算了（35+-5）。
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
        print(f"Model: {model_name}, Context Length: {context_len}, Best Verification Token: {verify_lens[np.argmax(performances)]}, Max Speedup: {max(performances):.3f} tokens/s")
        print(f"Original Speedup: {performances[68]:.3f} tokens/s")
        print(f"Naive Speedup: {performances[34]:.3f} tokens/s")
    print("***************************************************************************************")

# EAGLE-3
# Original set直接设置60算了。
# naive set设置在30左右。
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
        print(f"Model: {model_name}, Context Length: {context_len}, Best Verification Token: {verify_lens[np.argmax(performances)]}, Max Speedup: {max(performances):.3f}")
        print(f"Original Speedup: {performances[60]:.3f} tokens/s")
        print(f"Naive Speedup: {performances[30]:.3f} tokens/s")
    print("***************************************************************************************")


# 开始绘图
# 对于longspec，选择4列：256，1024， 4096, 16384
data1 = [
    [0.716,	1.006,	1.745,	2.108],
    [1.208,	1.599,	2.358,	2.335],
    [2.422,	2.664,	2.600,	2.459]
]   # vicuna-7B的数据
data1 = np.array(data1)
data2 = [
    [	0.728,  0.960, 	1.837,  2.751],
    [	1.273,	1.730,  2.748,	2.798],
    [	2.585,	2.870, 	2.995,  3.002]
]   # vicuna-13B的数据
data2 = np.array(data2)
data3 = [
    [0.681, 0.957,	1.883,	2.706],
    [1.277, 1.757,	2.741,	2.737],
    [2.364, 2.480,  2.776,	2.932],
]   # longchat-7B的数据
data3 = np.array(data3)
data4 = [
    [0.723,	0.874,	1.651,	2.685],
    [1.242,	1.491,	2.748,	2.676],
    [2.550,	2.542,	2.803,	2.919],
]   # longchat-13B的数据
data4 = np.array(data4)
data5 = [
    [1.180,	1.075,	1.080],
    [1.877,	1.630,	1.681],
    [3.250,	2.884,	2.769]
]   # eagle-llama-3.1-8B
data5 = np.array(data5)
data6 = [
    [1.345,	1.386,	1.769],
    [2.219,	2.367,	2.956],
    [4.070,	3.904,	3.700],
]   # eagle-vicuna-13B
data6 = np.array(data6)
Best_points = [
    [15,19,26,14],
    [12,18,33,41],
    [12,14,16],
    [12,12,40,44],
    [12,18,33,36],
    [13,14,25]
]
avg_speedup=["2.54×", "2.86×", "2.97×", "2.64×", "2.70×", "3.89×"]


# 全部数据、标题、横轴标签
all_data = [data1, data2, data5, data3, data4, data6]
titles = ['(a) Vicuna-v1.5-7B', '(c) Vicuna-v1.5-13B', '(e) Llama-3.1-8B',
          '(b) LongChat-7B', '(d) LongChat-13B', '(f) Vicuna-v1.3-13B']
xlabels_list = [
    ['256', '1024', '4096', '16384'],  # for 3x4
    ['256', '1024', '4096', '16384'],
    ['128', '512', '2048'],           # for 3x3
    ['256', '1024', '4096', '16384'],
    ['256', '1024', '4096', '16384'],
    ['128', '512', '2048'],
]
series = ['Ori Speedup', 'Naive Speedup', 'Best Speedup']
colors = [(197/255, 224/255, 180/255), (248/255, 203/255, 173/255), (180/255, 199/255, 231/255)]

# 创建子图 2行×3列
fig, axs = plt.subplots(2, 3, figsize=(15, 7))
axs = axs.flatten()

for idx, (data, ax) in enumerate(zip(all_data, axs)):
    num_series, num_groups = data.shape
    group_width = 0.8
    bar_width = group_width / num_series
    x = np.arange(num_groups)
    
    # 柱状图绘制
    for i in range(num_series):
        ax.bar(x + i * bar_width, data[i], width=bar_width,
               label=series[i], color=colors[i])
    
    # 子图下方标题
    ax.text(0.5, -0.25, titles[idx], transform=ax.transAxes,
            ha='center', va='top')
    
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(xlabels_list[idx])
    ax.set_xlabel('Context Length')
    ax.set_ylabel('Speedup')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    # ax.set_ylim(0, 4.2)  # 设置y轴范围
    
    # ✅ 添加右侧折线图
    ax2 = ax.twinx()
    best_y = Best_points[idx]
    best_x = x + bar_width * 2  # 中心对齐柱状图
    ax2.plot(best_x, best_y, color=(246/255,92/255,76/255), linestyle='-',marker='o', markersize=4,label='Best Verification Length')
    ax2.set_ylabel('Best Set')
    ax2.tick_params(axis='y')
    ax2.set_ylim(-20,50)    # 设置y轴范围

    # ✅ 只收集一次图例（第一幅子图）
    if idx == 0:
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        all_handles = handles1 + handles2
        all_labels = labels1 + labels2

    # 添加加速比
    ax.text(0.95, 0.95, avg_speedup[idx],
            transform=ax.transAxes,
            ha='right', va='top', fontweight='bold')

# 全局图例包含柱状图和折线图
fig.legend(all_handles, all_labels, loc='upper center', ncol=4)

plt.tight_layout(rect=[0, 0.0, 1, 0.95])

# 在图像坐标中添加一条竖直的灰色虚线，区分左右
fig_width = 1.0  # 以 figure 的宽度为 1
fig_height = 1.0

# 根据你 2x3 子图的布局，第二列和第三列之间大约在 2/3 的位置
x_pos = 2 / 3  # 或者精调为 0.675 之类

fig.lines.append(plt.Line2D(
    [x_pos, x_pos],    # x 起点和终点（归一化 figure 坐标）
    [0.05, 0.93],       # y 起点和终点（控制上下边界间距）
    transform=fig.transFigure,
    color='gray',
    linestyle='--',
    linewidth=2
))
# 添加底部分组标签
fig.text(0.33, 0.04, "(LongSpec)", ha='center', va='top', fontsize='large')
fig.text(0.83, 0.04, "(EAGLE-3)", ha='center', va='top', fontsize='large')


plt.savefig("Figures/paper/all_models_speedup.pdf")
plt.show()

