"""
本文件对LongSpec算法构建长短文本下的所有模型。
Batch size恒定为1.
使用的AAT数据来自: ./Data/表3-2.xlsx, ./Data/表3-4.xlsx.
论文默认配置: gamma=4, tree_shape为:[4,16,16,16,16].
希望我搜索出来的配置能超过这些默认配置。
"""
import sys
sys.path.append("./")
sys.path.append("../")
import pandas as pd
from run_longspec import run_longspec
from roofline.draw import draw_roofline, draw_roofline_discount, draw_acc, draw_combined_model
from modeling.parse_args import parse_args
from modeling.modeling_longspec import longspec_draft_cycles_comp
from modeling.modeling_llama3 import llama3_cycles_comp

def main():
    # 先配置一些超参数
    parser = parse_args()
    parser.add_argument("--avg_accepted_tokens",    type=int,   default=4   ,     help="the average number of accepted tokens per iteration."         )
    parser.add_argument("--gamma",                  type=int,   default=1,        help="it's similar to total_tokens, (depth+1) in eagle algorithm."  )
    args = parser.parse_args()
    args.batch_size = 1

    prefill_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    data_dir1 = "./Data/表3-2.xlsx"
    data_dir2 = "./Data/表3-4.xlsx"

    # step 1: 针对链式投机采样，数据来自表3-2.xlsx
    df = pd.read_excel(data_dir1, skiprows=2, header=None)
    gammas = df.iloc[:, 0].tolist()
    verify_lens = [gamma + 1 for gamma in gammas]   # 大模型校验时的输入数据长度
    AATs_all = df.iloc[:, 1:].values.T.tolist()           # AAT数据，存为2D列表
    i = 0
    for prefill_len in prefill_lens:
        AATs = AATs_all[i]
        # 计算draft, verify的时间
        verify_times = []
        draft_times = []
        args.prompt_len = prefill_len
        for total_token, AAT in zip(verify_lens, AATs):
            aat = int(AAT)
            # 小模型的draft阶段
            _, _, _, fused_draft_cycles = longspec_draft_cycles_comp(args, input_len=aat, kv_len=args.prompt_len)
            # 大模型的verify阶段
            _, _, _, fused_verify_cycles = llama3_cycles_comp(args, input_len=total_token, kv_len=args.prompt_len)
            draft_time = fused_draft_cycles / (args.clock_frequency * 1e6)
            verify_time = fused_verify_cycles / (args.clock_frequency * 1e6)
            verify_times.append(verify_time)
            draft_times.append(draft_time)
        draw_roofline(prefill_lengths=verify_lens, times=verify_times, save_path=f"Figures/longspec_all/chain/{prefill_len}_roofline_model.png")
        draw_roofline_discount(prefill_lengths=verify_lens, verify_times=verify_times, draft_times=draft_times, 
                            save_path=f"Figures/longspec_all/chain/{prefill_len}_discounted_roofline_model.png")
        draw_acc(prefill_lengths=verify_lens, accepted_lengths=AATs, save_path=f"Figures/longspec_all/chain/{prefill_len}_acc.png")
        draw_combined_model(prefill_lengths=verify_lens, verify_times=verify_times, draft_times=draft_times, 
                            accepted_lengths=AATs, save_path=f"Figures/longspec_all/chain/{prefill_len}_combined_model.png", batch_size=1, ori_x=5)
        i += 1


    # step 2: 针对树形投机采样，数据来自表3-4.xlsx

if __name__ == "__main__":
    main()