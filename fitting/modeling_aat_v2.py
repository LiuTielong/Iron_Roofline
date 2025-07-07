"""
针对longspec的曲线拟合。
"""

import sys
sys.path.append("./")
sys.path.append("../")
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 20  # 设置全局字体大小
plt.rcParams['font.family'] = 'Times New Roman'
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.optimize import fsolve
from matplotlib.ticker import FormatStrFormatter

def draw_all():
    # 现在把所有的AAT曲线都画在一张图上
    data_dir = "./Data/表3-4.xlsx"
    prefill_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    df = pd.read_excel(data_dir, skiprows=2, header=None)
    gammas = df.iloc[:, 0].tolist()
    verify_lens = [gamma + 1 for gamma in gammas]           # 大模型校验时的输入数据长度
    AATs_all = df.iloc[:, 2:].values.T.tolist()             # AAT数据，存为2D列表
    for i, aat_curve in enumerate(AATs_all):
        plt.plot(verify_lens, aat_curve, marker='o', label=f'prefill {prefill_lens[i]}')
    plt.xlabel("Verify Lengths")
    plt.ylabel("AAT")
    plt.title("AAT Curves")
    plt.legend()
    plt.grid(True)
    plt.show()


def fitting3():
    # 非线性拟合某一条曲线
    # 拟合曲线选择对数模型: y = A + B * ln(x - C)
    data_dir = "./Data/表3-4.xlsx"
    df = pd.read_excel(data_dir, skiprows=2, header=None)
    gammas = df.iloc[:, 0].tolist()
    verify_lens = [gamma + 1 for gamma in gammas]           # 大模型校验时的输入数据长度
    aat = df.iloc[:, 9].tolist()                            # AAT数据
    verify_lens = np.array(verify_lens)
    aat = np.array(aat)

    # 接下来进行非线性拟合
    def model_func(x, A, B, C):
        return A + B * np.log(x - C)
    
    # 执行非线性最小二乘拟合
    initial_guess = [1.0, 1.0, 1.0]  # A 和 B 的初始猜测
    params, covariance = curve_fit(model_func, verify_lens, aat, p0=initial_guess)

    # 拟合得到的参数
    A_fit, B_fit, C_fit = params
    print(f"拟合结果：A = {A_fit:.4f}, B = {B_fit:.4f}, C = {C_fit:.4f}")

    # 拟合曲线可视化
    y_fit = model_func(verify_lens, A_fit, B_fit, C_fit)
    plt.plot(verify_lens, y_fit, label="Fitted Curve", color='red')
    plt.plot(verify_lens, aat, marker='o', label="Original Curve", color='blue')
    plt.xlabel("Verify Lengths")
    plt.ylabel("AAT")
    plt.title("AAT Curves")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 计算决定系数r2
    r2 = r2_score(aat, y_fit)
    print(f"决定系数r2: {r2:.4f}")

    rmse = np.sqrt(mean_squared_error(aat, y_fit))  # 均方根误差
    mae = mean_absolute_error(aat, y_fit)  # 平均绝对误差

    print(f"RMSE（均方根误差）: {rmse:.4f}")
    print(f"MAE（平均绝对误差）: {mae:.4f}")
    # x = B_fit/0.02+C_fit
    # print(x)


def draw_fitting():
    """
    这里我们挑选几条曲线, 同时绘制原始的AAT和拟合后的曲线, 以展示我们的拟合方法是靠谱的。
    """
    data_dir = "./Data/表3-4.xlsx"
    df = pd.read_excel(data_dir, skiprows=2, header=None)
    gammas = df.iloc[:, 0].tolist()
    verify_lens = [gamma + 1 for gamma in gammas]           # 大模型校验时的输入数据长度
    AATs_all = df.iloc[:, 2:].values.T.tolist()             # AAT数据，存为2D列表

    # 选择要绘制的曲线索引
    selected_indices = [1, 3, 5, 7]
    Y_fit = []
    # 排列4张小图，每张图都绘制原始的AAT曲线和拟合后的曲线
    for i in selected_indices:
        aat_curve = AATs_all[i]
        
        # 拟合曲线
        def model_func(x, A, B, C):
            return A + B * np.log(x - C)
        
        # 执行非线性最小二乘拟合
        initial_guess = [1.0, 1.0, 1.0]
        params, covariance = curve_fit(model_func, verify_lens, aat_curve, p0=initial_guess)
        A_fit, B_fit, C_fit = params
        y_fit = model_func(verify_lens, A_fit, B_fit, C_fit)
        Y_fit.append(y_fit)  # 保存拟合结果
        
    # 第一种绘制方法
    colors = [(180/255, 199/255, 231/255), (248/255, 203/255, 173/255), (197/255, 224/255, 180/255), (255/255, 230/255, 153/255)]
    context_lengths = [256, 1024, 4096, 16384]
    plt.figure(figsize=(12, 8))
    for i, y_fit in enumerate(Y_fit):
        plt.plot(verify_lens, AATs_all[selected_indices[i]], linestyle="-", label=f'Original Curve ({context_lengths[i]})', color=colors[i], linewidth=2)
        plt.plot(verify_lens, y_fit, linestyle="--", label=f'Fitted Curve ({context_lengths[i]})', color=colors[i], linewidth=2)
    plt.xlabel("Verification Tokens")
    plt.ylabel("AAT")
    # plt.title("Selected AAT Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig("./figures/paper/aat_fitting.pdf", bbox_inches='tight', dpi=300)  # 保存图片
    plt.show()

    # 第二种绘制方法
    # 创建一个 2x2 的子图网格
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    titles = ["(Context Length: 256)", "(Context Length: 1024)",
             "(Context Length: 4096)", "(Context Length: 16384)"]
    # 遍历每个子图，并在每个子图中绘制相应的曲线
    for j, ax in enumerate(axes.flat):
        # j 从 0 到 3，对应 selected_indices 的每组曲线
        ax.plot(verify_lens, AATs_all[selected_indices[j]], linestyle="-", 
                label=f'Original Curve', color=colors[j], linewidth=2)
        ax.plot(verify_lens, Y_fit[j], linestyle="--", 
                label=f'Fitted Curve', color=colors[j], linewidth=2)
        ax.set_xlabel("Verification Tokens")
        ax.set_ylabel("AAT")
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.legend()
        ax.grid(True)
        # 在每个子图的下方添加标题文字
        ax.text(0.5, -0.25, titles[j], transform=ax.transAxes,
                ha='center', va='top', fontsize=22, color='black')

    plt.tight_layout()
    plt.savefig("./figures/paper/aat_fitting_subplots.pdf", bbox_inches='tight', dpi=300)
    plt.show()



def resolve():
    """
    对每一条曲线，给出三个数据点，请求解出A, B, C的值.
    这里 y = A + B * np.log(x - C).
    """

    data_dir = "./Data/表3-4.xlsx"
    df = pd.read_excel(data_dir, skiprows=2, header=None)
    gammas = df.iloc[:, 0].tolist()
    verify_lens = [gamma + 1 for gamma in gammas]           # 大模型校验时的输入数据长度
    aat = df.iloc[:, 9].tolist()                            # AAT数据

    # 3个数据点
    x_data = [verify_lens[0], verify_lens[24], verify_lens[42]]
    y_data = [aat[0], aat[24], aat[42]]
    x_data = np.array(x_data, dtype=float)
    y_data = np.array(y_data, dtype=float)

    # 方程组定义
    def equations(vars):
        A, B, C = vars
        eq1 = A + B * np.log(x_data[0] - C) - y_data[0]
        eq2 = A + B * np.log(x_data[1] - C) - y_data[1]
        eq3 = A + B * np.log(x_data[2] - C) - y_data[2]
        return [eq1, eq2, eq3]

    # 初始猜测
    initial_guess = [1.5, 1.0, 0.5]

    # 求解
    solution = fsolve(equations, initial_guess)
    A, B, C = solution

    print(f"A = {A:.4f}, B = {B:.4f}, C = {C:.4f}")

    # 用求解出来的A, B, C拟合曲线.
    def model_func(x):
        return A + B * np.log(x - C)
    
    verify_lens = np.array(verify_lens)
    aat = np.array(aat)

    y_fit = model_func(verify_lens)

    plt.plot(verify_lens, y_fit, label="Fitted Curve", color='red')
    plt.plot(verify_lens, aat, marker='o', label="Original Curve", color='blue')
    plt.xlabel("Verify Lengths")
    plt.ylabel("AAT")
    plt.title("AAT Curves")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 计算决定系数r2
    r2 = r2_score(aat, y_fit)
    print(f"决定系数r2: {r2:.4f}")




if __name__ == "__main__":
    # draw_all()
    # fitting3()
    draw_fitting()
    # resolve()