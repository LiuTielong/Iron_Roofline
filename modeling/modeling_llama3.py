import math
import argparse
import numpy as np

def llama3_weight_load_size(args, input_len=1):
    """
    针对的target model: llama-3.1-8B.
    一个问题: 权重到底要不要加载很多次？
    回答: 不重要。因为我关注的都是decode阶段, input_len是一个比较小的数(gamma). 就算是batch size=128这么大, 权重也只需要加载一次。
    """
    batch_size = args.batch_size
    loop = math.ceil(input_len * batch_size / args.MM_START_M_NUMBER)
    # assert loop == 1, "loop should be 1, because we only care about the decode stage."
    bit = args.weight_bit
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    num_layers = args.num_layers
    vocab_size = args.vocab_size
    kv_scale = args.kv_scale  # kv_scale=1/4, 即: k_proj, v_proj的大小只有q_proj的1/4

    param_size = hidden_size * hidden_size * 2                                  # q, o
    param_size += hidden_size * hidden_size * 2 * kv_scale                      # k, v
    param_size += hidden_size * intermediate_size * 3                           # up, gate, down
    param_size *= num_layers                                                    # 32 layers
    param_size += hidden_size * vocab_size                                      # lm_head
    param_size *= loop                                                          # how many times the weight need to be loaded
    param_size *= (bit / 8)                                                     # byte
    return param_size

def llama3_act_load_size(args, input_len, kv_len=0):
    """
    针对的target model: llama-3.1-8B.
    对于我考虑的情况, 虽然都是decode阶段, 但是input_len不为0, 而是一个比较小的数(因为target model在做verify).
    """
    bit = args.act_bit
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    num_layers = args.num_layers
    num_heads = args.num_heads
    vocab_size = args.vocab_size
    kv_scale = args.kv_scale
    batch_size = args.batch_size

    # loading activations
    act_size  = input_len * hidden_size                                         # input norm
    act_size += input_len * hidden_size * 3                                     # q, k, v 的输入
    act_size += hidden_size * kv_len * 2 * kv_scale                             # kv cache加载
    act_size += input_len * hidden_size * (1 + kv_scale)                        # qkt_matmul
    act_size += (input_len * (input_len+kv_len) * num_heads + input_len * hidden_size * kv_scale)    # pv_matmul
    act_size += input_len *  hidden_size * 2                                    # o_proj and residual
    act_size += input_len * hidden_size                                         # post_layernorm
    act_size += input_len * hidden_size                                         # gate_proj
    act_size += input_len * hidden_size + input_len * intermediate_size         # up_proj
    act_size += (input_len * intermediate_size + input_len * hidden_size)       # down_proj and residual

    act_size *= num_layers                                                      # 32 layers
    act_size += input_len * hidden_size                                         # norm
    act_size += input_len * hidden_size                                         # lm_head

    act_size *= batch_size                                                      # batch size
    act_size *= (bit / 8)                                                       # byte
    return act_size


def llama3_act_st_size(args, input_len, kv_len):
    """
    Description:
        Compute the size of storing activations.
        Don't need to store kv cache.
    """
    bit = args.act_bit
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    num_layers = args.num_layers
    num_heads = args.num_heads
    vocab_size = args.vocab_size
    kv_scale = args.kv_scale
    batch_size = args.batch_size
    kv_len = input_len + kv_len       # 虽然不用重复存储很长的kv cache, 但是kv_len还是会影响qkt_matmul的输出的存储量。

    act_size  = input_len * hidden_size                                         # layer norm
    act_size += input_len * hidden_size * (1 + 2 * kv_scale)                    # qkv
    act_size += input_len * kv_len * num_heads                                  # qkt_matmul
    act_size += input_len * hidden_size                                         # pv_matmul
    act_size += input_len * hidden_size                                         # o_proj
    act_size += input_len * hidden_size                                         # post_layernorm
    act_size += input_len * intermediate_size * 2                               # up_proj, gate_proj
    act_size += input_len * hidden_size                                         # down_proj

    act_size *= num_layers
    act_size += input_len * hidden_size                                         # norm
    act_size += input_len * vocab_size                                          # lm_head
    act_size *= batch_size
    act_size *= (bit / 8)
    return act_size

def llama3_mm_comp(args, input_len, kv_len):
    """
    计算矩阵矩阵乘、矩阵向量乘的时间。
    对于magicdec, 还是要兼顾一下大模型和小模型。
    大模型的verify算是prefill阶段, 小模型的draft算是decode阶段。
    大小模型主要就是两个区别: input_len, kv_len.
    注意: GQA的设计并不会影响到计算量, 只会影响到访存量。??????????
    """
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    num_layers = args.num_layers
    num_heads = args.num_heads
    vocab_size = args.vocab_size
    batch_size = args.batch_size
    kv_scale = args.kv_scale

    computation = 0
    kv_len = input_len + kv_len
    computation += input_len * hidden_size * hidden_size                    # q_proj
    computation += input_len * hidden_size * (hidden_size * kv_scale) * 2   # k_proj, v_proj
    computation += input_len * kv_len * hidden_size * 2                     # qkt_matmul, pv_matmul
    computation += input_len * hidden_size * hidden_size                    # o_proj
    computation += input_len * hidden_size * intermediate_size * 3          # up_proj, gate_proj, down_proj
    
    computation *= num_layers
    computation += input_len * hidden_size * vocab_size                     # lm_head
    computation *= batch_size
    return computation

def llama3_cycles_comp(args, input_len, kv_len):
    # LD
    hbm_trans_compatibility = args.hbm_bandwidth * (1.024**3) * 1000 / args.clock_frequency # how many bytes can be transferred in one cycle.
    weight_size = llama3_weight_load_size(args, input_len)
    ld_weight_cycle = weight_size / (hbm_trans_compatibility * args.num_wide_channels / args.num_hbm_channels * args.hbm_same_uti)

    act_size = llama3_act_load_size(args, input_len, kv_len)
    ld_act_cycle = act_size / (hbm_trans_compatibility * args.num_narrow_channels / args.num_hbm_channels * args.hbm_cross_uti)

    ld_cycle = (ld_weight_cycle + ld_act_cycle)

    # ST
    act_size = llama3_act_st_size(args, input_len, kv_len)
    st_cycle = act_size / (hbm_trans_compatibility * args.num_narrow_channels / args.num_hbm_channels * args.hbm_same_uti)

    # MM/MV
    computation = llama3_mm_comp(args, input_len, kv_len)
    comp_cycle = computation / (args.mm_parallel_m * args.mm_parallel_n * args.mm_parallel_k * args.num_slr)  # 算力直接拉满
    # if input_len != 1:
    #     comp_cycle = computation / (args.mm_parallel_m * args.mm_parallel_n * args.mm_parallel_k * args.num_slr)
    # else:
    #     comp_cycle = computation / (args.mv_parallel_m * args.mv_parallel_n * args.mv_parallel_k * args.num_slr)
    
    # FUSE
    fused_cycle = max(ld_cycle, st_cycle, comp_cycle)

    return ld_cycle, st_cycle, comp_cycle, fused_cycle