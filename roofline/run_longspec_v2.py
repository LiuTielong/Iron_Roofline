"""
本文件对LongSpec算法构建长短文本下的所有模型。
Batch size恒定为1.
使用的AAT数据来自: ./Data/表3-2.xlsx, ./Data/表3-4.xlsx.
论文默认配置: gamma=4, tree_shape为:[4,16,16,16,16].
希望我搜索出来的配置能超过这些默认配置。结果：确实超过了默认配置。
"""
import sys
sys.path.append("./")
sys.path.append("../")
import pandas as pd
from roofline.draw import draw_roofline, draw_roofline_discount, draw_acc, draw_combined_model
from modeling.parse_args import parse_args
from modeling.modeling_longspec import longspec_draft_cycles_comp
from modeling.modeling_llama3 import llama3_cycles_comp

def main():
    # 先配置一些超参数
    parser = parse_args()
    parser.add_argument("--avg_accepted_tokens",    type=int,   default=4   ,     help="the average number of accepted tokens per iteration."         )
    parser.add_argument("--gamma",                  type=int,   default=1,        help="it's similar to total_tokens, (depth+1) in eagle algorithm."  )
    parser.add_argument("--tree_shape", nargs="+",  type=int, default=[4, 16, 16, 16, 16], help="the tree shape of the draft token tree." )
    args = parser.parse_args()
    args.batch_size = 1

    prefill_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    data_dir1 = "./Data/表3-2.xlsx"
    data_dir2 = "./Data/表3-4.xlsx"

    # step 1: 针对链式投机采样，数据来自表3-2.xlsx
    """
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
            args.gamma = total_token - 1        # 这是draft阶段需要考虑的，draft了多少次
            # 小模型的draft阶段
            _, _, _, fused_draft_cycles = longspec_draft_cycles_comp(args, input_len=aat, kv_len=args.prompt_len, method="seq")
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
    """

    # step 2: 针对树形投机采样，数据来自表3-4.xlsx
    df = pd.read_excel(data_dir2, skiprows=2, header=None)
    gammas = df.iloc[:, 0].tolist()
    verify_lens = [gamma + 1 for gamma in gammas]   # 大模型校验时的输入数据长度
    tree_shapes = df.iloc[:, 1].tolist()            # 树形投机采样的树形结构
    AATs_all = df.iloc[:, 2:].values.T.tolist()     # AAT数据，存为2D列表
    i = 0
    for prefill_len in prefill_lens:
        AATs = AATs_all[i]
        # 计算draft, verify的时间
        verify_times = []
        draft_times = []
        args.prompt_len = prefill_len
        for total_token, AAT, tree_shape in zip(verify_lens, AATs, tree_shapes):
            aat = int(AAT)
            args.tree_shape = [int(x) for x in tree_shape.split()]
            # 小模型的draft阶段
            _, _, _, fused_draft_cycles = longspec_draft_cycles_comp(args, input_len=aat, kv_len=args.prompt_len, method="tree")
            # 大模型的verify阶段
            _, _, _, fused_verify_cycles = llama3_cycles_comp(args, input_len=total_token, kv_len=args.prompt_len)
            draft_time = fused_draft_cycles / (args.clock_frequency * 1e6)
            verify_time = fused_verify_cycles / (args.clock_frequency * 1e6)
            verify_times.append(verify_time)
            draft_times.append(draft_time)
        draw_roofline(prefill_lengths=verify_lens, times=verify_times, save_path=f"Figures/longspec_all/tree/{prefill_len}_roofline_model.png")
        draw_roofline_discount(prefill_lengths=verify_lens, verify_times=verify_times, draft_times=draft_times, 
                            save_path=f"Figures/longspec_all/tree/{prefill_len}_discounted_roofline_model.png")
        draw_acc(prefill_lengths=verify_lens, accepted_lengths=AATs, save_path=f"Figures/longspec_all/tree/{prefill_len}_acc.png")
        draw_combined_model(prefill_lengths=verify_lens, verify_times=verify_times, draft_times=draft_times, 
                            accepted_lengths=AATs, save_path=f"Figures/longspec_all/tree/{prefill_len}_combined_model.png", batch_size=1, ori_x=69)
        # 因为原始配置是[4,16,16,16,16], 节点数为1+68=69.
        i += 1


def test_vicuna_v15_7B():
    """
    测试vicuna-v1.5-7B模型的加速比.
    实验数据来自Data/表4-1.xlsx。
    实验结果：相当不错。
    """
    parser = parse_args()
    parser.add_argument("--avg_accepted_tokens",    type=int,   default=4   ,     help="the average number of accepted tokens per iteration."         )
    parser.add_argument("--gamma",                  type=int,   default=1,        help="it's similar to total_tokens, (depth+1) in eagle algorithm."  )
    parser.add_argument("--tree_shape", nargs="+",  type=int, default=[4, 16, 16, 16, 16], help="the tree shape of the draft token tree." )
    args = parser.parse_args()
    args.batch_size = 1
    
    # 首先修改模型的超参数, 修改成vicuna-v1.5-7B的配置
    args.hidden_size = 4096
    args.intermediate_size = 11008
    args.num_layers = 32
    args.num_heads = 32
    args.head_dim = 128
    args.vocab_size = 32000
    args.kv_scale = 1.0  # 不进行GQA

    # 读取数据
    data_dir = "./Data/表4-1.xlsx"
    df = pd.read_excel(data_dir, skiprows=1, header=None)
    gammas = df.iloc[:, 0].tolist()
    verify_lens = [gamma + 1 for gamma in gammas]   # 大模型校验时的输入数据长度
    tree_shapes = df.iloc[:, 1].tolist()            # 树形投机采样的树形结构
    AATs = df.iloc[:, 2].tolist()     

    # 目前只针对prefill length=1024进行了实验
    args.prompt_len = 1024
    verify_times = []
    draft_times = []
    for total_token, AAT, tree_shape in zip(verify_lens, AATs, tree_shapes):
        aat = int(AAT)
        args.tree_shape = [int(x) for x in tree_shape.split()]
        # 小模型的draft阶段
        _, _, _, fused_draft_cycles = longspec_draft_cycles_comp(args, input_len=aat, kv_len=args.prompt_len, method="tree")
        # 大模型的verify阶段
        _, _, _, fused_verify_cycles = llama3_cycles_comp(args, input_len=total_token, kv_len=args.prompt_len)
        draft_time = fused_draft_cycles / (args.clock_frequency * 1e6)
        verify_time = fused_verify_cycles / (args.clock_frequency * 1e6)
        verify_times.append(verify_time)
        draft_times.append(draft_time)
    draw_roofline(prefill_lengths=verify_lens, times=verify_times, save_path=f"Figures/longspec_other_models/vicuna_v15_7B/roofline_model.png")
    draw_roofline_discount(prefill_lengths=verify_lens, verify_times=verify_times, draft_times=draft_times, 
                            save_path=f"Figures/longspec_other_models/vicuna_v15_7B/discounted_roofline_model.png")
    draw_acc(prefill_lengths=verify_lens, accepted_lengths=AATs, save_path=f"Figures/longspec_other_models/vicuna_v15_7B/acc.png")
    draw_combined_model(prefill_lengths=verify_lens, verify_times=verify_times, draft_times=draft_times, 
                        accepted_lengths=AATs, save_path=f"Figures/longspec_other_models/vicuna_v15_7B/combined_model.png", batch_size=1, ori_x=71, naive_x=None)
    # 原始配置其实是68，不过我没有对它进行测试，所以就用71来代替了


def test_vicuna_v15_13B():
    """
    测试vicuna-v1.5-13B模型的加速比.
    实验数据来自Data/表4-1.xlsx。
    实验结果：相当不错。
    """
    parser = parse_args()
    parser.add_argument("--avg_accepted_tokens",    type=int,   default=4   ,     help="the average number of accepted tokens per iteration."         )
    parser.add_argument("--gamma",                  type=int,   default=1,        help="it's similar to total_tokens, (depth+1) in eagle algorithm."  )
    parser.add_argument("--tree_shape", nargs="+",  type=int, default=[4, 16, 16, 16, 16], help="the tree shape of the draft token tree." )
    args = parser.parse_args()
    args.batch_size = 1
    
    # 首先修改模型的超参数
    args.hidden_size = 5120
    args.intermediate_size = 13824
    args.num_layers = 40
    args.num_heads = 40
    args.head_dim = 128
    args.vocab_size = 32000
    args.kv_scale = 1.0  # 不进行GQA

    # 读取数据
    data_dir = "./Data/表4-1.xlsx"
    df = pd.read_excel(data_dir, skiprows=1, header=None)
    gammas = df.iloc[:, 0].tolist()
    verify_lens = [gamma + 1 for gamma in gammas]   # 大模型校验时的输入数据长度
    tree_shapes = df.iloc[:, 1].tolist()            # 树形投机采样的树形结构
    AATs = df.iloc[:, 3].tolist()                   # 提取本模型对应的列的AAT数据

    # 目前只针对prefill length=1024进行了实验
    args.prompt_len = 1024
    verify_times = []
    draft_times = []
    for total_token, AAT, tree_shape in zip(verify_lens, AATs, tree_shapes):
        aat = int(AAT)
        args.tree_shape = [int(x) for x in tree_shape.split()]
        # 小模型的draft阶段
        _, _, _, fused_draft_cycles = longspec_draft_cycles_comp(args, input_len=aat, kv_len=args.prompt_len, method="tree")
        # 大模型的verify阶段
        _, _, _, fused_verify_cycles = llama3_cycles_comp(args, input_len=total_token, kv_len=args.prompt_len)
        draft_time = fused_draft_cycles / (args.clock_frequency * 1e6)
        verify_time = fused_verify_cycles / (args.clock_frequency * 1e6)
        verify_times.append(verify_time)
        draft_times.append(draft_time)
    draw_roofline(prefill_lengths=verify_lens, times=verify_times, save_path=f"Figures/longspec_other_models/vicuna_v15_13B/roofline_model.png")
    draw_roofline_discount(prefill_lengths=verify_lens, verify_times=verify_times, draft_times=draft_times, 
                            save_path=f"Figures/longspec_other_models/vicuna_v15_13B/discounted_roofline_model.png")
    draw_acc(prefill_lengths=verify_lens, accepted_lengths=AATs, save_path=f"Figures/longspec_other_models/vicuna_v15_13B/acc.png")
    draw_combined_model(prefill_lengths=verify_lens, verify_times=verify_times, draft_times=draft_times, 
                        accepted_lengths=AATs, save_path=f"Figures/longspec_other_models/vicuna_v15_13B/combined_model.png", batch_size=1, ori_x=71, naive_x=None)
    # 原始配置其实是68，不过我没有对它进行测试，所以就用71来代替了


def test_longchat_7b():
    """
    测试longchat-7B模型的加速比.
    实验数据来自Data/表4-1.xlsx。
    Longchat-7B实际上架构也就是llama-2-7b的。
    """
    parser = parse_args()
    parser.add_argument("--avg_accepted_tokens",    type=int,   default=4   ,     help="the average number of accepted tokens per iteration."         )
    parser.add_argument("--gamma",                  type=int,   default=1,        help="it's similar to total_tokens, (depth+1) in eagle algorithm."  )
    parser.add_argument("--tree_shape", nargs="+",  type=int, default=[4, 16, 16, 16, 16], help="the tree shape of the draft token tree." )
    args = parser.parse_args()
    args.batch_size = 1

    # 首先修改模型的超参数, 修改成vicuna-v1.5-7B的配置
    args.hidden_size = 4096
    args.intermediate_size = 11008
    args.num_layers = 32
    args.num_heads = 32
    args.head_dim = 128
    args.vocab_size = 32000
    args.kv_scale = 1.0  # 不进行GQA

    # 读取数据
    data_dir = "./Data/表4-1.xlsx"
    df = pd.read_excel(data_dir, skiprows=1, header=None)
    gammas = df.iloc[:, 0].tolist()
    verify_lens = [gamma + 1 for gamma in gammas]   # 大模型校验时的输入数据长度
    tree_shapes = df.iloc[:, 1].tolist()            # 树形投机采样的树形结构
    AATs = df.iloc[:, 4].tolist()     

    # 目前只针对prefill length=1024进行了实验
    args.prompt_len = 1024
    verify_times = []
    draft_times = []
    for total_token, AAT, tree_shape in zip(verify_lens, AATs, tree_shapes):
        aat = int(AAT)
        args.tree_shape = [int(x) for x in tree_shape.split()]
        # 小模型的draft阶段
        _, _, _, fused_draft_cycles = longspec_draft_cycles_comp(args, input_len=aat, kv_len=args.prompt_len, method="tree")
        # 大模型的verify阶段
        _, _, _, fused_verify_cycles = llama3_cycles_comp(args, input_len=total_token, kv_len=args.prompt_len)
        draft_time = fused_draft_cycles / (args.clock_frequency * 1e6)
        verify_time = fused_verify_cycles / (args.clock_frequency * 1e6)
        verify_times.append(verify_time)
        draft_times.append(draft_time)
    draw_roofline(prefill_lengths=verify_lens, times=verify_times, save_path=f"Figures/longspec_other_models/longchat_7B/roofline_model.png")
    draw_roofline_discount(prefill_lengths=verify_lens, verify_times=verify_times, draft_times=draft_times, 
                            save_path=f"Figures/longspec_other_models/longchat_7B/discounted_roofline_model.png")
    draw_acc(prefill_lengths=verify_lens, accepted_lengths=AATs, save_path=f"Figures/longspec_other_models/longchat_7B/acc.png")
    draw_combined_model(prefill_lengths=verify_lens, verify_times=verify_times, draft_times=draft_times, 
                        accepted_lengths=AATs, save_path=f"Figures/longspec_other_models/longchat_7B/combined_model.png", batch_size=1, ori_x=71, naive_x=None)
    # 原始配置其实是68，不过我没有对它进行测试，所以就用71来代替了


def test_longchat_13B():
    """
    测试longchat-13B模型的加速比.
    实验数据来自Data/表4-1.xlsx。
    实验结果：相当不错。
    """
    parser = parse_args()
    parser.add_argument("--avg_accepted_tokens",    type=int,   default=4   ,     help="the average number of accepted tokens per iteration."         )
    parser.add_argument("--gamma",                  type=int,   default=1,        help="it's similar to total_tokens, (depth+1) in eagle algorithm."  )
    parser.add_argument("--tree_shape", nargs="+",  type=int, default=[4, 16, 16, 16, 16], help="the tree shape of the draft token tree." )
    args = parser.parse_args()
    args.batch_size = 1
    
    # 首先修改模型的超参数
    args.hidden_size = 5120
    args.intermediate_size = 13824
    args.num_layers = 40
    args.num_heads = 40
    args.head_dim = 128
    args.vocab_size = 32000
    args.kv_scale = 1.0  # 不进行GQA

    # 读取数据
    data_dir = "./Data/表4-1.xlsx"
    df = pd.read_excel(data_dir, skiprows=1, header=None)
    gammas = df.iloc[:, 0].tolist()
    verify_lens = [gamma + 1 for gamma in gammas]   # 大模型校验时的输入数据长度
    tree_shapes = df.iloc[:, 1].tolist()            # 树形投机采样的树形结构
    AATs = df.iloc[:, 5].tolist()                   # 提取本模型对应的列的AAT数据

    # 目前只针对prefill length=1024进行了实验
    args.prompt_len = 1024
    verify_times = []
    draft_times = []
    for total_token, AAT, tree_shape in zip(verify_lens, AATs, tree_shapes):
        aat = int(AAT)
        args.tree_shape = [int(x) for x in tree_shape.split()]
        # 小模型的draft阶段
        _, _, _, fused_draft_cycles = longspec_draft_cycles_comp(args, input_len=aat, kv_len=args.prompt_len, method="tree")
        # 大模型的verify阶段
        _, _, _, fused_verify_cycles = llama3_cycles_comp(args, input_len=total_token, kv_len=args.prompt_len)
        draft_time = fused_draft_cycles / (args.clock_frequency * 1e6)
        verify_time = fused_verify_cycles / (args.clock_frequency * 1e6)
        verify_times.append(verify_time)
        draft_times.append(draft_time)
    draw_roofline(prefill_lengths=verify_lens, times=verify_times, save_path=f"Figures/longspec_other_models/longchat_13B/roofline_model.png")
    draw_roofline_discount(prefill_lengths=verify_lens, verify_times=verify_times, draft_times=draft_times, 
                            save_path=f"Figures/longspec_other_models/longchat_13B/discounted_roofline_model.png")
    draw_acc(prefill_lengths=verify_lens, accepted_lengths=AATs, save_path=f"Figures/longspec_other_models/longchat_13B/acc.png")
    draw_combined_model(prefill_lengths=verify_lens, verify_times=verify_times, draft_times=draft_times, 
                        accepted_lengths=AATs, save_path=f"Figures/longspec_other_models/longchat_13B/combined_model.png", batch_size=1, ori_x=71, naive_x=None)
    # 原始配置其实是68，不过我没有对它进行测试，所以就用71来代替了


if __name__ == "__main__":
    main()
    # test_vicuna_v15_7B()
    # test_vicuna_v15_13B()
    # test_longchat_7b()
    # test_longchat_13B()