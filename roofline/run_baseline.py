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
    
    # 长短文本下的生成, batch_size=1
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


"""
结果：
prefill长度为:128, perf: 40.09 tokens/s
prefill长度为:256, perf: 39.39 tokens/s  
prefill长度为:512, perf: 38.06 tokens/s  
prefill长度为:1024, perf: 35.66 tokens/s 
prefill长度为:2048, perf: 31.66 tokens/s 
prefill长度为:4096, perf: 25.86 tokens/s 
prefill长度为:8192, perf: 18.92 tokens/s 
prefill长度为:16384, perf: 12.32 tokens/s
值得注意的是: 我这里在变化prefill长度时上下文performance变化也如此大, 是因为
在load activation的时候, hbm的通道数和通道利用率都要比load weight小很多。
所以本质上我这里仍然是基于FlightLLM的一个仿真器。
"""