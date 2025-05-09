"""
计算使用MagicDec算法跑llama-3.1-8B模型, 能达到的performance (tokens/s)。
这里需要考虑不同的total-token设置已经对应的平均接受数。
实验场景设计：
(1) 长文本下的生成(prefill_length=32000), batch_size=1.
(2) 长文本下的生成(prefill_length=32000), batch_size=128.
投机采样方法就是链式投机采样, 不支持tree结构。
"""

import sys
sys.path.append("./")
sys.path.append("../")

from modeling.parse_args import parse_args
from modeling.modeling_llama3 import llama3_cycles_comp
from modeling.modeling_magicdec import magicdec_draft_cycles_comp
from modeling.AAT import magicdec_aat
from roofline.draw import draw_roofline, draw_roofline_discount, draw_acc, draw_combined_model

total_tokens = list(magicdec_aat.keys())        # 也就是gamma
AATs = list(magicdec_aat.values())

def run_magicdec():
    # 先配置一些超参数
    parser = parse_args()
    parser.add_argument("--avg_accepted_tokens",    type=int,   default=4   ,     help="the average number of accepted tokens per iteration."         )
    parser.add_argument("--gamma",                  type=int,   default=1,        help="it's similar to total_tokens, (depth+1) in eagle algorithm."  )
    parser.add_argument("--draft_kv_budget",        type=int,   default=512,      help="the kv budget for draft stage."                               )
    args = parser.parse_args()

    # 1. 长文本生成, batch_size=1.
    verify_times1 = []
    draft_times1 = []
    args.prompt_len = 32000
    args.batch_size = 1
    for total_token, AAT in zip(total_tokens, AATs):
        aat = int(AAT)
        args.gamma = total_token
        # 1.1 小模型的draft阶段
        _, _, _, fused_draft_cycles = magicdec_draft_cycles_comp(args, input_len=aat)
        # 1.2 大模型的verify阶段
        _, _, _, fused_verify_cycles = llama3_cycles_comp(args, input_len=total_token, kv_len=args.prompt_len)
        draft_time = fused_draft_cycles / (args.clock_frequency * 1e6)
        verify_time = fused_verify_cycles / (args.clock_frequency * 1e6)
        verify_times1.append(verify_time)
        draft_times1.append(draft_time)

    # 2. 长文本生成，batch_size=128.
    verify_times2 = []
    draft_times2 = []
    args.prompt_len = 32000
    args.batch_size = 128
    for total_token, AAT in zip(total_tokens, AATs):
        aat = int(AAT)
        args.gamma = total_token
        # 2.1 小模型的draft阶段
        _, _, _, fused_draft_cycles = magicdec_draft_cycles_comp(args, input_len=aat)
        # 2.2 大模型的verify阶段
        _, _, _, fused_verify_cycles = llama3_cycles_comp(args, input_len=total_token, kv_len=args.prompt_len)
        draft_time = fused_draft_cycles / (args.clock_frequency * 1e6)
        verify_time = fused_verify_cycles / (args.clock_frequency * 1e6)
        verify_times2.append(verify_time)
        draft_times2.append(draft_time)

    return verify_times1, verify_times2, draft_times1, draft_times2


def draw_figures():
    """
    根据run()函数的结果来绘制基本的roofline模型, 打折后的roofline模型, 接受率曲线, 最终的组合模型。
    """
    verify_times1, verify_times2, draft_times1, draft_times2 = run_magicdec()
    # 1. 长文本，batch_size=1
    draw_roofline(prefill_lengths=total_tokens, times=verify_times1, save_path="Figures/magicdec/bs1_roofline_model.png")
    draw_roofline_discount(prefill_lengths=total_tokens, verify_times=verify_times1, draft_times=draft_times1, 
                           save_path="Figures/magicdec/bs1_discounted_roofline_model.png")
    draw_acc(prefill_lengths=total_tokens, accepted_lengths=AATs, save_path="Figures/magicdec/bs1_acc.png")
    draw_combined_model(prefill_lengths=total_tokens, verify_times=verify_times1, draft_times=draft_times1, 
                        accepted_lengths=AATs, save_path="Figures/magicdec/bs1_combined_model.png")

    # 2. 长文本，batch_size=128
    draw_roofline(prefill_lengths=total_tokens, times=verify_times2, save_path="Figures/magicdec/bs128_roofline_model.png" ,batch_size=128)
    draw_roofline_discount(prefill_lengths=total_tokens, verify_times=verify_times2, draft_times=draft_times2, 
                           save_path="Figures/magicdec/bs128_discounted_roofline_model.png", batch_size=128)
    draw_acc(prefill_lengths=total_tokens, accepted_lengths=AATs, save_path="Figures/magicdec/bs128_acc.png")
    draw_combined_model(prefill_lengths=total_tokens, verify_times=verify_times2, draft_times=draft_times2, 
                        accepted_lengths=AATs, save_path="Figures/magicdec/bs128_combined_model.png", batch_size=128)
    return


if __name__ == "__main__":
    # verify_times1, draft_times1, verify_times2, draft_times2 = run_magicdec()
    draw_figures()