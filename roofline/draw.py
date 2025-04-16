"""
包含绘制Roofline model的函数; 绘制组合模型的函数。
"""
import matplotlib.pyplot as plt
import numpy as np

def draw_roofline(prefill_lengths, times, save_path:str, batch_size:int=1):
    """
    Description:
        绘制原始大模型在硬件平台上的roofline模型。
        横坐标: prefill的token数(也就是投机采样过程中verify的token数).
        纵坐标: 每秒处理的token数(也就是perf)。
    Inputs:
        prefill_lengths: 一个列表。也就是verified token数。
        times: 一个和prefill_lengths一一对应的列表。每个元素是执行prefill_lengths[i]个token的时间。
        save_path: 图片保存位置。
        batch_size: batch size. 默认是1。
    """
    performance = np.array(prefill_lengths) / np.array(times) * batch_size # tokens / s

    plt.figure(figsize=(10, 6))
    plt.plot(prefill_lengths, performance, marker='o', linestyle='-', color='r')
    plt.xlabel("Prefill length (tokens)")
    plt.ylabel("Performance (tokens/s)")
    plt.title("Roofline Model")
    plt.grid()
    # plt.savefig("Figures/roofline_model.png")
    plt.savefig(save_path)
    plt.show()
    return 


def draw_roofline_discount(prefill_lengths, verify_times, draft_times, save_path:str, batch_size:int=1):
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
        batch_size: batch size. 默认是1。
    """
    performance = np.array(prefill_lengths) / np.array(verify_times) * batch_size # tokens / s
    performance *= np.array(verify_times) / (np.array(verify_times) + np.array(draft_times)) # 打折

    plt.figure(figsize=(10, 6))
    plt.plot(prefill_lengths, performance, marker='o', linestyle='-', color='r')
    plt.xlabel("Prefill length (tokens)")
    plt.ylabel("Performance (tokens/s)")
    plt.title("Discounted Roofline Model")
    plt.grid()
    # plt.savefig("Figures/Discounted_roofline_model.png")
    plt.savefig(save_path)
    plt.show()
    return


def draw_acc(prefill_lengths, accepted_lengths, save_path:str):
    """
    Description:
        绘制接受数、接受率随verify的token数变化的曲线。
    Inputs:
        prefill_lengths: 一个列表。也就是verified token数。
        accepted_lengths: 一个列表。对应每个prefill_lengths[i]的接受数。
        save_path: 图片保存位置。
        不需要batch_size.
    """
    acc_rate = np.array(accepted_lengths) / np.array(prefill_lengths)   # acceptance rate.

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Total verified tokens')
    ax1.set_ylabel('Average Accepted tokens', color=color1)
    ax1.plot(prefill_lengths, accepted_lengths, marker='o', linestyle='-', color=color1, label='average accepted tokens')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()  # 创建共享x轴的第二个坐标轴
    color2 = 'tab:red'
    ax2.set_ylabel('Acceptance Rate', color=color2)
    ax2.plot(prefill_lengths, acc_rate, marker='s', linestyle='--', color=color2, label='acceptance rate')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title('TT-AAT and TT-AR')
    fig.tight_layout()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()
    return


def draw_combined_model(prefill_lengths, verify_times, draft_times, accepted_lengths, save_path:str, batch_size:int=1):
    """
    Description:
        用打折后的roofline模型乘以接受率曲线, 就能得到组合模型。
        纵坐标是有效performance, 横坐标是verify的token数。
    Inputs:
        prefill_lengths: 一个列表。也就是verified token数。
        performance: 一个列表。原始roofline模型的performance。
        verify_times: 大模型verify的时间。
        draft_times: 小模型draft的时间。
        accepted_lengths: 一个列表。对应每个prefill_lengths[i]的接受数。
        save_path: 图片保存位置。
    """
    print(verify_times)
    print(accepted_lengths)
    performance_discounted = np.array(prefill_lengths) / (np.array(verify_times) + np.array(draft_times)) * batch_size
    print(performance_discounted)
    acc_rate = np.array(accepted_lengths) / np.array(prefill_lengths) 

    efficiency = np.array(performance_discounted) * acc_rate
    print(efficiency)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(prefill_lengths, efficiency, marker='o', linestyle='-', color='r')
    plt.xlabel("Prefill length (tokens)")
    plt.ylabel("Effective performance (tokens/s)")
    plt.title("Combined Model")
    plt.grid()
    plt.savefig(save_path)
    plt.show()
    return