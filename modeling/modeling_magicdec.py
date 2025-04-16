"""
本文件用来计算MagicDec的draft模型的运行cycle数。
本来我是不打算建立这个模型的, 就直接使用modeling_llama3.llama3_cycles_comp函数。
但是权衡之下还是决定一个文件, 专门计算整个draft过程的时间。

"""

from modeling.modeling_llama3 import llama3_weight_load_size, llama3_act_load_size, llama3_act_st_size, llama3_mm_comp, llama3_cycles_comp

def magicdec_draft_cycles_comp(args, input_len):
    kv_len = args.draft_kv_budget
    gamma = args.gamma

    LD_CYCLES = 0
    ST_CYCLES = 0
    COMP_CYCLES = 0
    FUSED_CYCLES = 0

    # 1. 生成链的第一个节点
    ld_cycle, st_cycle, comp_cycle, fused_cycle = llama3_cycles_comp(args, input_len, kv_len)
    LD_CYCLES += ld_cycle
    ST_CYCLES += st_cycle
    COMP_CYCLES += comp_cycle
    FUSED_CYCLES += fused_cycle

    # 2. 生成链的后续（gamma-1个节点）
    for i in range(gamma - 1):
        ld_cycle, st_cycle, comp_cycle, fused_cycle = llama3_cycles_comp(args, 1, kv_len)
        LD_CYCLES += ld_cycle
        ST_CYCLES += st_cycle
        COMP_CYCLES += comp_cycle
        FUSED_CYCLES += fused_cycle

    return LD_CYCLES, ST_CYCLES, COMP_CYCLES, FUSED_CYCLES