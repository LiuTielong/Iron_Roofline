"""
计算直接跑llama-3.1-8B模型, 能达到的performance (tokens/s)。
仅仅考虑生成一个token的时间, 然后用1除以它就得到performance了。
1. 对于run_baseline()函数, 实验场景是4种: 
(1) 短文本下的生成, batch_size=1.
(2) 短文本下的生成, batch_size=128.
(3) 长文本下的生成, batch_size=1.
(4) 长文本下的生成, batch_size=128.
2. 对于run_baseline_v2()函数, 实验场景是8种.
全都是batch_size=1, 但是prefill长度为: [128,256,512,1024,2048,4096,8192,16384]。
"""

import sys
sys.path.append("./")
sys.path.append("../")

from modeling.modeling_llama3 import llama3_cycles_comp
from modeling.parse_args import parse_args


def run_baseline():
    parser = parse_args()
    args = parser.parse_args()
    
    # 短文本下的生成, batch_size=1
    args.prompt_len = 128  # 在generate的时候，这就是kv cache长度了
    args.batch_size = 1
    _, _, _, fused_cycles = llama3_cycles_comp(args, input_len=1, kv_len=args.prompt_len)
    fused_time = fused_cycles / (args.clock_frequency * 1e6)  # 单位是秒
    perf = 1 / fused_time
    print("perf: %.2f tokens/s"% perf)

    # 短文本下的生成，batch_size=128
    args.prompt_len = 128
    args.batch_size = 128
    _, _, _, fused_cycles = llama3_cycles_comp(args, input_len=1, kv_len=args.prompt_len)
    fused_time = fused_cycles / (args.clock_frequency * 1e6)  # 单位是秒
    perf = args.batch_size / fused_time
    print("perf: %.2f tokens/s"% perf)

    # 长文本下的生成，batch_size=1
    args.prompt_len = 32000
    args.batch_size = 1
    _, _, _, fused_cycles = llama3_cycles_comp(args, input_len=1, kv_len=args.prompt_len)
    fused_time = fused_cycles / (args.clock_frequency * 1e6)  # 单位是秒
    perf = 1 / fused_time
    print("perf: %.2f tokens/s"% perf)

    # 长文本下的生成，batch_size=128
    args.prompt_len = 32000
    args.batch_size = 128
    _, _, _, fused_cycles = llama3_cycles_comp(args, input_len=1, kv_len=args.prompt_len)
    fused_time = fused_cycles / (args.clock_frequency * 1e6)  # 单位是秒
    perf = args.batch_size / fused_time
    print("perf: %.2f tokens/s"% perf)
    return

def run_baseline_v2():
    parser = parse_args()
    args = parser.parse_args()
    
    # 短文本下的生成, batch_size=1
    args.batch_size = 1
    prefill_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    for prefill_len in prefill_lens:
        args.prompt_len = prefill_len
        _, _, _, fused_cycles = llama3_cycles_comp(args, input_len=1, kv_len=args.prompt_len)
        fused_time = fused_cycles / (args.clock_frequency * 1e6)  # 单位是秒
        perf = 1 / fused_time
        print("prefill长度为:%d, perf: %.2f tokens/s" %(prefill_len, perf))


if __name__ == "__main__":
    # run_baseline()
    run_baseline_v2()