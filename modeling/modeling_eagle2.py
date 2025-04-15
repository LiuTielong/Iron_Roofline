import math

def eagle2_weight_load_size(args, input_len):
    """
    针对的target model: llama-3.1-8B.
    input_len: 差不多就是top-k, 即: 树的每一层的宽度。
    """
    
    batch_size = args.batch_size
    loop = math.ceil(input_len * batch_size / args.MM_START_M_NUMBER)
    bit = args.weight_bit
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    vocab_size = args.vocab_size
    kv_scale = args.kv_scale  # kv_scale=1/4, 即: k_proj, v_proj的大小只有q_proj的1/4

    weight_size = 0
    weight_size += hidden_size * hidden_size * 2                                    # fc
    weight_size += hidden_size * hidden_size * (2 + 2 * kv_scale)                   # q, k, v, o_proj
    weight_size += hidden_size * intermediate_size * 3                              # up_proj, gate_proj, down_proj
    weight_size += hidden_size * vocab_size                                         # lm_head 
    weight_size *= loop
    weight_size *= (bit / 8)                   
    return weight_size


def eagle2_act_load_size(args, input_len, kv_len):
    bit = args.act_bit
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    num_heads = args.num_heads
    vocab_size = args.vocab_size
    batch_size = args.batch_size
    kv_scale = args.kv_scale

    act_size = 0
    act_size += input_len * hidden_size * 2                                         # fc

    act_size += input_len * hidden_size                                             # input layernorm
    act_size += input_len * hidden_size * 3                                         # q, k, v的输入
    act_size += hidden_size * kv_len * 2 * kv_scale                                 # kv cache加载
    act_size += input_len * hidden_size * (1 + kv_scale)                            # qkt_matmul
    act_size += (input_len * input_len * num_heads + input_len * hidden_size * kv_scale)  # pv_matmul
    act_size += input_len * hidden_size * 2                                        # o_proj and residual
    act_size += input_len * hidden_size                                             # post_layernorm
    act_size += input_len * hidden_size                                             # gate_proj
    act_size += input_len * hidden_size + input_len * intermediate_size             # up_proj
    act_size += (input_len * intermediate_size + input_len * hidden_size)           # down_proj and residual

    act_size += input_len * hidden_size                                             # lm_head
    act_size *= (bit / 8)                                                       
    act_size *= batch_size
    return act_size

def eagle2_act_st_size(args, input_len):
    bit = args.act_bit
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    num_heads = args.num_heads
    vocab_size = args.vocab_size
    batch_size = args.batch_size
    kv_scale = args.kv_scale

    act_size = 0
    act_size += input_len * hidden_size                                             # fc
    act_size += input_len * hidden_size                                             # input layernorm
    act_size += input_len * hidden_size * (1 + 2 * kv_scale)                        # q, k, v的输入
    act_size += input_len * input_len * num_heads                                   # qkt_matmul
    act_size += input_len * hidden_size                                             # pv_matmul
    act_size += input_len * hidden_size                                             # o_proj
    act_size += input_len * hidden_size                                             # post_layernorm
    act_size += input_len * intermediate_size * 2                                   # up_proj, gate_proj
    act_size += input_len * hidden_size                                             # down_proj

    act_size += input_len * vocab_size                                              # lm_head
    act_size *= batch_size
    act_size *= (bit / 8)
    return act_size

def eagle2_mm_comp(args, input_len, kv_len):
    """
    Description:
        Compute the computation of MM / MV operations.
    """
    bit = args.act_bit
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    num_heads = args.num_heads
    vocab_size = args.vocab_size
    kv_scale = args.kv_scale
    batch_size = args.batch_size

    kv_len = input_len + kv_len
    computation = 0

    computation += input_len * hidden_size * hidden_size * 2                                     # fc
    computation += input_len * hidden_size * hidden_size * (1 + 2 * kv_scale)                    # q, k, v generation.
    computation += input_len * kv_len * hidden_size * 2                                          # qkt_matmul, pv_matmul
    computation += input_len * hidden_size * hidden_size                                         # o_proj
    computation += input_len * hidden_size * intermediate_size * 3                               # up_proj, gate_proj, down_proj
    
    computation += input_len * hidden_size * vocab_size                                          # lm_head

    computation *= batch_size
    return computation

def eagle2_cycles_comp(args, input_len, kv_len):
    # LD
    hbm_trans_compatibility = args.hbm_bandwidth * (1.024**3) * 1000 / args.clock_frequency # how many bytes can be transferred in one cycle.
    weight_size = eagle2_weight_load_size(args, input_len)
    ld_weight_cycle = weight_size / (hbm_trans_compatibility * args.num_wide_channels / args.num_hbm_channels * args.hbm_same_uti)

    act_size = eagle2_act_load_size(args, input_len, kv_len)
    ld_act_cycle = act_size / (hbm_trans_compatibility * args.num_narrow_channels / args.num_hbm_channels * args.hbm_cross_uti)

    ld_cycle = (ld_weight_cycle + ld_act_cycle)

    # ST
    act_size = eagle2_act_st_size(args, input_len)
    st_cycle = act_size / (hbm_trans_compatibility * args.num_narrow_channels / args.num_hbm_channels * args.hbm_same_uti)

    # MM/MV
    computation = eagle2_mm_comp(args, input_len, kv_len)
    comp_cycle = computation / (args.mm_parallel_m * args.mm_parallel_n * args.mm_parallel_k * args.num_slr)

    # FUSE
    fused_cycle = max(ld_cycle, st_cycle, comp_cycle)

    return ld_cycle, st_cycle, comp_cycle, fused_cycle