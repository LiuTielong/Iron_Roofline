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

def draw_all():
    # 现在把所有的AAT曲线都画在一张图上
    data_dir = "./Data/表3-4.xlsx"
    prefill_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    df = pd.read_excel(data_dir, skiprows=2, header=None)
    gammas = df.iloc[:, 0].tolist()
    verify_lens = [gamma + 1 for gamma in gammas]           # 大模型校验时的输入数据长度
    AATs_all = df.iloc[:, 3:].values.T.tolist()             # AAT数据，存为2D列表, prefill长度为128的那一列我不要了
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
    aat = df.iloc[:, 2].tolist()                            # AAT数据
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
    resolve()