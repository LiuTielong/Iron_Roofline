"""
完全仿照run_baseline.py写。这里计算的是vicuna-v1.5-7B, 13B模型的performance. (tokens/s).
仅仅考虑生成一个token的时间, 然后用1除以它就得到performance了。
"""

import sys
sys.path.append("./")
sys.path.append("../")

from modeling.modeling_vicuna_v15 import vicuna_v15_cycles_comp
from modeling.modeling_llama3 import llama3_cycles_comp
from modeling.parse_args import parse_args

def run_baseline_7B():
    parser = parse_args()
    args = parser.parse_args()
    # 首先修改模型的超参数
    args.hidden_size = 4096
    args.intermediate_size = 11008
    args.num_layers = 32
    args.num_heads = 32
    args.head_dim = 128
    args.vocab_size = 32000
    args.kv_scale = 1.0  # 不进行GQA

    args.batch_size = 1
    prefill_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    for prefill_len in prefill_lens:
        args.prompt_len = prefill_len
        # 用vicuna_v15_cycles_comp和llama3_cycles_comp算出来的cycle数是一样的！
        # 我算是白写了vicuna_v15_cycles_comp函数了。
        # _, _, _, fused_cycles = llama3_cycles_comp(args, input_len=1, kv_len=args.prompt_len)
        _, _, _, fused_cycles = vicuna_v15_cycles_comp(args, input_len=1, kv_len=args.prompt_len)
        fused_time = fused_cycles / (args.clock_frequency * 1e6)  # 单位是秒
        perf = 1 / fused_time
        print("prefill长度为:%d, perf: %.2f tokens/s" %(prefill_len, perf))


def run_baseline_13B():
    parser = parse_args()
    args = parser.parse_args()
    # 首先修改模型的超参数
    args.hidden_size = 5120
    args.intermediate_size = 13824
    args.num_layers = 40
    args.num_heads = 40
    args.head_dim = 128
    args.vocab_size = 32000
    args.kv_scale = 1.0  # 不进行GQA

    args.batch_size = 1
    prefill_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    for prefill_len in prefill_lens:
        args.prompt_len = prefill_len
        # 用vicuna_v15_cycles_comp和llama3_cycles_comp算出来的cycle数是一样的！
        # 我算是白写了vicuna_v15_cycles_comp函数了。
        # _, _, _, fused_cycles = llama3_cycles_comp(args, input_len=1, kv_len=args.prompt_len)
        _, _, _, fused_cycles = vicuna_v15_cycles_comp(args, input_len=1, kv_len=args.prompt_len)
        fused_time = fused_cycles / (args.clock_frequency * 1e6)  # 单位是秒
        perf = 1 / fused_time
        print("prefill长度为:%d, perf: %.2f tokens/s" %(prefill_len, perf))
    return


if __name__ == "__main__":
    # run_baseline_7B()
    run_baseline_13B()  