"""
计算使用eagle-2算法跑llama-3.1-8B模型，能达到的performance (tokens/s)。
这里需要考虑不同的total-token设置已经对应的平均接受数。
实验场景先设计两种：
(1) 短文本下的生成, batch_size=1.
(2) 短文本下的生成, batch_size=128.
没有长文本实验场景的原因是我现在还没有长文本下的平均接受数。
"""

import sys
sys.path.append("./")
sys.path.append("../")

from modeling.modeling_eagle2 import eagle2_cycles_comp
from modeling.parse_args import parse_args
from modeling.modeling_llama3 import llama3_cycles_comp
from modeling.AAT import eagle2_aat
from .draw import draw_roofline, draw_roofline_discounted, draw_acceptance_rate, draw_combined_model

total_tokens = list(eagle2_aat.keys())
AATs = list(eagle2_aat.values())

def run_eagle2():
    # 先配置一些超参数
    parser = parse_args()
    parser.add_argument("--avg_accepted_tokens",    type=int,   default=4   ,     help="the average number of accepted tokens per iteration."                     )
    parser.add_argument("--total_token",            type=int,   default=60,       help="the maxinum number of new generated tokens that would be verified by LLM.")
    parser.add_argument("--top_k",                  type=int,   default=10,       help="the number of generated tokens per SSM forward process."                  )
    parser.add_argument("--depth",                  type=int,   default=6,        help="the depth of draft token tree - 2."                                       )
    args = parser.parse_args()
    
    # 1. 短文本下的生成，batch_size=1.
    run_times = []
    perf1 = []
    args.prompt_len = 128
    args.batch_size = 1
    for total_token, AAT in zip(total_tokens, AATs):
        aat = int(AAT)
        # 1.1 小模型的draft阶段
        _, _, _, fused_draft_cycles = eagle2_cycles_comp(args, input_len=aat, kv_len=args.prompt_len)
        # 1.2 大模型的verify阶段
        _, _, _, fused_verify_cycles = llama3_cycles_comp(args, input_len=total_token, kv_len=args.prompt_len)
        run_time = (fused_draft_cycles + fused_verify_cycles) / (args.clock_frequency * 1e6)  # 单位是秒
        run_times.append(run_time)
        perf1.append(aat / run_time)
    # print(perf1)

    # 2. 短文本下的生成，batch_size=128.
    # batch_size一变大，更加compute bound，投机采样更加没用了。
    run_times = []
    perf2 = []
    args.prompt_len = 128
    args.batch_size = 128
    for total_token, AAT in zip(total_tokens, AATs):
        aat = int(AAT)
        # 2.1 小模型的draft阶段
        _, _, _, fused_draft_cycles = eagle2_cycles_comp(args, input_len=aat, kv_len=args.prompt_len)
        # 2.2 大模型的verify阶段
        _, _, _, fused_verify_cycles = llama3_cycles_comp(args, input_len=total_token, kv_len=args.prompt_len)
        run_time = (fused_draft_cycles + fused_verify_cycles) / (args.clock_frequency * 1e6)  # 单位是秒
        run_times.append(run_time)
        perf2.append(aat * args.batch_size / run_time)
    # print(perf2)

    # 长文本的生成：待实现。
    return perf1, perf2

def draw_figures():
    """
    根据run()函数的结果来绘制基本的roofline模型, 打折后的roofline模型, 接受率曲线, 最终的组合模型。
    """
    perf1, perf2 = run_eagle2()
    # 1. 短文本，batch_size=1

    # 2. 短文本，batch_size=128


if __name__ == "__main__":
    run_eagle2()