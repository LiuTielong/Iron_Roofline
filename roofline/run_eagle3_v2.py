"""
本文件对EAGLE-3算法构建长短文本下的所有模型。
Batch size恒定为1.
使用的AAT数据来自: ./Data/表3-5.xlsx, ./Data/表3-6.xlsx.
论文默认配置: total_token=60.
希望我搜索出来的配置能超过默认配置。
"""

import sys
sys.path.append("./")
sys.path.append("../")
import pandas as pd
from roofline.draw import draw_roofline, draw_roofline_discount, draw_acc, draw_combined_model
from modeling.parse_args import parse_args
from modeling.modeling_llama3 import llama3_cycles_comp
from modeling.modeling_eagle3 import eagle3_cycles_comp, eagle3_draft_cycles_comp

def main():
    parser = parse_args()
    parser.add_argument("--total_token",            type=int,   default=60,       help="the maxinum number of new generated tokens that would be verified by LLM.")
    parser.add_argument("--top_k",                  type=int,   default=10,       help="the number of generated tokens per SSM forward process."                  )
    parser.add_argument("--depth",                  type=int,   default=6,        help="the depth of draft token tree - 2."                                       )
    args = parser.parse_args()

    data_dir1 = "./Data/表3-5.xlsx"
    data_dir2 = "./Data/表3-6.xlsx"

    # step 1: 针对链式投机采样，数据来自表3-5.xlsx
    prefill_lens = [128, 256, 512, 1024, 2048, 4096, 8192]
    df = pd.read_excel(data_dir1, skiprows=2, header=None)
    gammas = df.iloc[:, 0].tolist()
    verify_lens = [gamma + 1 for gamma in gammas]   # 大模型校验时的输入数据长度
    AATs_all = df.iloc[:, 1:].values.T.tolist()           # AAT数据，存为2D列表
    for i in range(len(prefill_lens)):
        prefill_len = prefill_lens[i]
        AATs = AATs_all[i]
        # 计算draft, verify的时间
        verify_times = []
        draft_times = []
        args.prompt_len = prefill_len
        args.batch_size = 1
        for total_token, AAT in zip(verify_lens, AATs):
            aat = int(AAT)
            args.depth = total_token - 2        # depth和top_k需要为draft阶段设置
            args.top_k = 1
            # 小模型的draft阶段
            _, _, _, fused_draft_cycles = eagle3_draft_cycles_comp(args, input_len=aat, kv_len=args.prompt_len)
            # 大模型的verify阶段
            _, _, _, fused_verify_cycles = llama3_cycles_comp(args, input_len=total_token, kv_len=args.prompt_len)
            draft_time = fused_draft_cycles / (args.clock_frequency * 1e6)
            verify_time = fused_verify_cycles / (args.clock_frequency * 1e6)
            verify_times.append(verify_time)
            draft_times.append(draft_time)
        draw_roofline(prefill_lengths=verify_lens, times=verify_times, save_path=f"Figures/eagle3_all/chain/{prefill_len}_roofline_model.png")
        draw_roofline_discount(prefill_lengths=verify_lens, verify_times=verify_times, draft_times=draft_times, 
                            save_path=f"Figures/eagle3_all/chain/{prefill_len}_discounted_roofline_model.png")
        draw_acc(prefill_lengths=verify_lens, accepted_lengths=AATs, save_path=f"Figures/eagle3_all/chain/{prefill_len}_acc.png")
        draw_combined_model(prefill_lengths=verify_lens, verify_times=verify_times, draft_times=draft_times, 
                            accepted_lengths=AATs, save_path=f"Figures/eagle3_all/chain/{prefill_len}_combined_model.png", batch_size=1, ori_x=5)
    
    # step 2: 针对树形投机采样，数据来自表3-6.xlsx
    prefill_lens = [128, 256, 512, 1024, 2048]
    df = pd.read_excel(data_dir2, skiprows=2, header=None)
    gammas = df.iloc[:, 0].tolist()
    verify_lens = [gamma + 1 for gamma in gammas]           # 大模型校验时的输入数据长度
    draft_depths = df.iloc[:, 1].tolist()                   # draft阶段的深度
    draft_topks = df.iloc[:, 2].tolist()                    # draft阶段的top_k
    AATs_all = df.iloc[:, 3:].values.T.tolist()             # AAT数据，存为2D列表
    for i in range(len(prefill_lens)):
        prefill_len = prefill_lens[i]
        AATs = AATs_all[i]
        # 计算draft, verify的时间
        verify_times = []
        draft_times = []
        args.prompt_len = prefill_len
        args.batch_size = 1
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
            verify_times.append(verify_time)
            draft_times.append(draft_time)
        draw_roofline(prefill_lengths=verify_lens, times=verify_times, save_path=f"Figures/eagle3_all/tree/{prefill_len}_roofline_model.png")
        draw_roofline_discount(prefill_lengths=verify_lens, verify_times=verify_times, draft_times=draft_times, 
                            save_path=f"Figures/eagle3_all/tree/{prefill_len}_discounted_roofline_model.png")
        draw_acc(prefill_lengths=verify_lens, accepted_lengths=AATs, save_path=f"Figures/eagle3_all/tree/{prefill_len}_acc.png")
        draw_combined_model(prefill_lengths=verify_lens, verify_times=verify_times, draft_times=draft_times, 
                            accepted_lengths=AATs, save_path=f"Figures/eagle3_all/tree/{prefill_len}_combined_model.png", batch_size=1, ori_x=60)


if __name__ == "__main__":
    main()