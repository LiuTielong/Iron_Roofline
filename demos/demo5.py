"""
已知一组AAT曲线中的三个点, 如何拟合出对数函数曲线, 并补充其他的点?
这里是对表8.xlsx中sheet2的Vicuna-13B的数据进行拟合和补充。
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 20  # 设置全局字体大小
plt.rcParams['font.family'] = 'Times New Roman'
from scipy.optimize import fsolve
from sklearn.metrics import r2_score
from scipy.optimize import least_squares

# 拟合模型的残差函数
def residuals(params, x, y):
    A, B, C = params
    # 限制 x - C > 0，否则 log 无定义
    if np.any(x - C <= 0):
        return np.full_like(y, 1e6)  # 返回大残差避免非法 log
    return A + B * np.log(x - C) - y

def resolve():
    """
    对每一条曲线，给出三个数据点，请求解出A, B, C的值.
    这里 y = A + B * np.log(x - C).
    """

    data_dir = "./Data/表8.xlsx"
    df = pd.read_excel(data_dir, skiprows=2, header=None, sheet_name="Sheet2")
    gammas = df.iloc[:, 1].tolist()
    gammas = gammas[0:8]
    verify_lens = [gamma + 1 for gamma in gammas]           # 大模型校验时的输入数据长度
    aat = df.iloc[:, 3].tolist()                            # AAT数据
    aat = aat[16:24]                                         # 截取哪一段AAT

    # 3个数据点
    x_data = [verify_lens[0], verify_lens[3], verify_lens[7]]
    y_data = [aat[0], aat[3], aat[7]]
    x_data = np.array(x_data, dtype=float)
    y_data = np.array(y_data, dtype=float)

    # 初始猜测
    initial_guess = [1.5, 1.0, 0.5]

    # 设置边界，确保 C < min(x_data)
    lower_bounds = [-np.inf, -np.inf, -np.inf]
    upper_bounds = [np.inf, np.inf, min(x_data) - 1e-6]

    # 执行拟合
    result = least_squares(residuals, initial_guess, bounds=(lower_bounds, upper_bounds),
                        args=(x_data, y_data))

    # 获取拟合结果
    A_fit, B_fit, C_fit = result.x
    print(f"拟合结果：A = {A_fit:.4f}, B = {B_fit:.4f}, C = {C_fit:.4f}")

    # 用拟合的函数绘图
    x_fit = np.linspace(2, 100, 300)
    y_fit = A_fit + B_fit * np.log(x_fit - C_fit)

    # 绘图展示
    plt.figure(figsize=(8, 5))
    plt.plot(x_fit, y_fit, label='Fitting Curve', color='blue')
    plt.scatter(x_data, y_data, color='red', label='Initial Data Points')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Fitting y = A + B * ln(x - C)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 计算其他5个数据点
    x_other = [verify_lens[1], verify_lens[2], verify_lens[4], verify_lens[5], verify_lens[6]]
    y_other = A_fit + B_fit * np.log(np.array(x_other) - C_fit)
    print("其他5个数据点：")
    for x, y in zip(x_other, y_other):
        print(f"x = {x:.2f}, y = {y:.3f}")



def main():
    resolve()


if __name__ == "__main__":
    main()