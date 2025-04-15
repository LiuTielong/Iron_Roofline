"""
包含绘制Roofline model的函数; 绘制组合模型的函数。
"""
import matplotlib.pyplot as plt
import numpy as np

def draw_roofline(prefill_lengths, times, save_path:str):
    """
    Description:
        绘制原始大模型在硬件平台上的roofline模型。
        横坐标: prefill的token数(也就是投机采样过程中verify的token数).
        纵坐标: 每秒处理的token数(也就是perf)。
    Inputs:
        prefill_lengths: 一个列表。也就是verified token数。
        times: 一个和prefill_lengths一一对应的列表。每个元素是执行prefill_lengths[i]个token的时间。
        save_path: 图片保存位置。
    """
    performance = np.array(prefill_lengths) / np.array(times) # tokens / s

    plt.figure(figsize=(10, 6))
    plt.plot(prefill_lengths, performance, marker='o', linestyle='-', color='r')
    plt.xlabel("Prefill length (tokens)")
    plt.ylabel("Performance (tokens/s)")
    plt.title("Roofline Model")
    plt.grid()
    plt.show()
    # plt.savefig("Figures/roofline_model.png")
    plt.savefig(save_path)
    return 


def draw_roofline_discount(prefill_lengths, verify_times, draft_times, save_path:str):
    """
    Description:
        对于投机采样算法, 我们必须考虑draft的时间。
        这就需要对大模型的roofline模型的performance进行打折。
        缩放因子为: (大模型verify的时间)/ (大模型verify的时间 + 小模型draft的时间)。
        对于EAGLE-2与EAGLE-3算法, 对于任意的verify长度, 我这里draft的时间实际上是固定的。(就是生成一棵树, 七层)。
        对于LongSpec和MagicDec, draft的时间也随着roofline模型的横坐标变化, 但是我们总能求出。
    Inputs:
        prefill_lengths: 一个列表。也就是verified token数。
        verify_times: 一个列表。第i个元素表示大模型同时对prefill_lengths[i]个token进行verify的时间。
        draft_times: 一个列表。第i个元素表示, 当verify长度为: prefill_length[i]时, 小模型进行一次draft需要的时间。
        save_path: 图片保存位置。
    """
    performance = np.array(prefill_lengths) / np.array(verify_times)
    performance *= np.array(verify_times) / (np.array(verify_times) + np.array(draft_times)) # tokens / s

    plt.figure(figsize=(10, 6))
    plt.plot(prefill_lengths, performance, marker='o', linestyle='-', color='r')
    plt.xlabel("Prefill length (tokens)")
    plt.ylabel("Performance (tokens/s)")
    plt.title("Discounted Roofline Model")
    plt.grid()
    plt.show()
    # plt.savefig("Figures/Discounted_roofline_model.png")
    plt.savefig(save_path)
    return


def draw_acc(prefill_lengths, accepted_lengths, save_path:str):
    """
    Description:
        绘制接受数、接受率随verify的token数变化的曲线。
    Inputs:
        prefill_lengths: 一个列表。也就是verified token数。
        accepted_lengths: 一个列表。对应每个prefill_lengths[i]的接受数。
        save_path: 图片保存位置。
    """
    AR = np.array(accepted_lengths) / np.array(prefill_lengths) 
    
    return


def draw_combined_model():
    """
    用打折后的roofline模型乘以接受率曲线, 就能得到组合模型。
    纵坐标是有效performance, 横坐标是verify的token数。
    """

    return