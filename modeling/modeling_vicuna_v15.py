"""
针对Vicuna-7b-v1.5, Vicuna-13b-v1.5模型的仿真器建模, 仿照modeling_llama3.py写的。
注意, Vicuna-v1.5是基于llama-2进行微调的。
"""

def vicuna_v15_weight_load_size(args, input_len=1):
    bit = args.weight_bit
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    num_layers = args.num_layers
    vocab_size = args.vocab_size

    param_size = hidden_size * hidden_size * 4      # q, k, v, o proj
    param_size += hidden_size * intermediate_size * 3  # up, gate, down
    param_size *= num_layers                        # 32 layers
    param_size += hidden_size * vocab_size          # lm_head
    param_size *= (bit / 8)                 # byte
    return param_size

def vicuna_v15_act_load_size(args, input_len, kv_len=0):
    bit = args.act_bit
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    num_layers = args.num_layers
    num_heads = args.num_heads
    vocab_size = args.vocab_size
    batch_size = args.batch_size

    # loading activations
    act_size = input_len * hidden_size                                          # input norm
    act_size += input_len * hidden_size * 3                                     # q, k, v 的输入
    act_size += hidden_size * kv_len * 2                                        # kv cache加载
    act_size += input_len * hidden_size * 2                                     # qkt_matmul
    act_size += (input_len * (input_len + kv_len) * num_heads + input_len * hidden_size)  # pv_matmul
    act_size += input_len * hidden_size * 2                                     # o_proj and residual
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

def vicuna_v15_act_st_size(args, input_len, kv_len):
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
    batch_size = args.batch_size
    vocab_size = args.vocab_size
    kv_len = input_len + kv_len

    act_size = input_len * hidden_size                                          # input norm
    act_size += input_len * hidden_size * 3                                     # q, k, v 的输出
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
    act_size *= (bit / 8)                                                       # byte
    return act_size

def vicuna_v15_mm_comp(args, input_len, kv_len):
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    num_layers = args.num_layers
    vocab_size = args.vocab_size
    batch_size = args.batch_size

    computation = 0
    kv_len = input_len + kv_len 
    computation += input_len * hidden_size * hidden_size * 3                # q, k, v proj
    computation += input_len * kv_len * hidden_size * 2                     # qkt_matmul, pv_matmul
    computation += input_len * hidden_size * hidden_size                    # o_proj
    computation += input_len * hidden_size * intermediate_size * 3          # up, gate, down

    computation *= num_layers                                                
    computation += input_len * hidden_size * vocab_size                      # lm_head
    computation *= batch_size
    return computation

def vicuna_v15_cycles_comp(args, input_len, kv_len):
    # LD
    hbm_trans_compatibility = args.hbm_bandwidth * (1.024**3) * 1000 / args.clock_frequency # how many bytes can be transferred in one cycle.
    weight_size = vicuna_v15_weight_load_size(args, input_len)
    ld_weight_cycle = weight_size / (hbm_trans_compatibility * args.num_wide_channels / args.num_hbm_channels * args.hbm_same_uti)

    act_size = vicuna_v15_act_load_size(args, input_len, kv_len)
    ld_act_cycle = act_size / (hbm_trans_compatibility * args.num_narrow_channels / args.num_hbm_channels * args.hbm_cross_uti)
    
    ld_cycle = (ld_weight_cycle + ld_act_cycle)

    # ST
    act_size = vicuna_v15_act_st_size(args, input_len, kv_len)
    st_cycle = act_size / (hbm_trans_compatibility * args.num_narrow_channels / args.num_hbm_channels * args.hbm_same_uti)

    # MM/MV
    computation = vicuna_v15_mm_comp(args, input_len, kv_len)
    comp_cycle = computation / (args.mm_parallel_m * args.mm_parallel_n * args.mm_parallel_k * args.num_slr)  # 算力直接拉满

    # FUSE
    fused_cycle = max(ld_cycle, st_cycle, comp_cycle)

    return ld_cycle, st_cycle, comp_cycle, fused_cycle