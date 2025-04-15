"""
EAGLE3 for llama-3.1-8B 模型的仿真器建模。
注意: 一定要支持长文本的kv cache, 要支持多batch.
小模型的结构：
1. embedding层和lm_head层都和大模型共用。
2. 一个fc层，负责将大模型的三层隐状态[batch_size, input_len, hidden_dim*3] 映射成[batch_size, input_len, hidden_dim]. 这个fc层只有在创建
draft token tree的第一层节点时用到。
3. 一个llama_decoder_layer. 但是和正常的decoder_layer有区别的是：它的输入是: [batch_size, input_len, hidden_dim*2], 然后经过
两个宽度都是4096的layernorm。接下来经过q_proj, k_proj, v_proj（这三个权重矩阵的输入维度都是8192，输出维度分别是4096,1024,1024）。后面就正常了。
"""

import math

def eagle3_weight_load_size(args, input_len, first_layer:bool):
    """
    first_layer: 是在生成树的第一层吗?如果是, 需要load fc 层。
    """
    
    batch_size = args.batch_size
    loop = math.ceil(input_len * batch_size / args.MM_START_M_NUMBER)
    bit = args.weight_bit
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    vocab_size = args.vocab_size
    kv_scale = args.kv_scale  # kv_scale=1/4, 即: k_proj, v_proj的大小只有q_proj的1/4

    weight_size = 0
    if first_layer:
        weight_size += hidden_size * hidden_size * 3                                # fc, 注意*3，这是固定的size
    weight_size += (hidden_size * 2) * hidden_size                                  # q，注意*2
    wieght_size += (hidden_size * 2) * hidden_size * kv_scale * 2                   # k, v，注意*2
    weight_size += hidden_size * hidden_size                                        # o_proj
    weight_size += hidden_size * intermediate_size * 3                              # up_proj, gate_proj, down_proj
    weight_size += hidden_size * vocab_size                                         # lm_head 
    weight_size *= loop
    weight_size *= (bit / 8)                   
    return weight_size


def eagle3_act_load_size(args, input_len, kv_len, first_layer:bool):
    """
    first_layer: 如果是在生成draft token tree的第一层, 输入hidden_states的形状为:[1, input_len, hidden_size*3],
    而且要经过fc层。否则，输入hidden_states的形状为:[batch_size, input_len, hidden_size]，不需要经过fc层。
    """
    bit = args.act_bit
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    num_heads = args.num_heads
    vocab_size = args.vocab_size
    batch_size = args.batch_size
    kv_scale = args.kv_scale

    act_size = 0
    if first_layer:
        act_size += input_len * hidden_size * 3                                     # fc, 注意*3，这是固定的size

    act_size += input_len * (hidden_size * 2)                                       # input layernorm
    act_size += input_len * (hidden_size * 2) * 3                                   # q, k, v的输入
    act_size += hidden_size * kv_len * 2 * kv_scale                                 # kv cache加载
    act_size += input_len * hidden_size * (1 + kv_scale)                            # qkt_matmul
    act_size += (input_len * (input_len+kv_len) * num_heads + input_len * hidden_size * kv_scale)  # pv_matmul
    act_size += input_len * hidden_size * 2                                         # o_proj and residual
    act_size += input_len * hidden_size                                             # post_layernorm
    act_size += input_len * hidden_size                                             # gate_proj
    act_size += input_len * hidden_size + input_len * intermediate_size             # up_proj
    act_size += (input_len * intermediate_size + input_len * hidden_size)           # down_proj and residual

    act_size += input_len * hidden_size                                             # lm_head
    act_size *= (bit / 8)                                                       
    act_size *= batch_size
    return act_size


def eagle3_act_st_size(args, input_len, kv_len, first_layer:bool):
    """
    first_layer: 如果是在生成draft token tree的第一层，需要经过fc层然后存储结果。
    """
    bit = args.act_bit
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    num_heads = args.num_heads
    vocab_size = args.vocab_size
    batch_size = args.batch_size
    kv_scale = args.kv_scale
    kv_len = kv_len + input_len

    act_size = 0 
    if first_layer:
        act_size += input_len * hidden_size                                         # fc的输出
    
    act_size += input_len * hidden_size * 2                                         # input layernorm的输出
    act_size += input_len * hidden_size * (1 + 2 * kv_scale)                        # q, k, v的输出
    act_size += input_len * kv_len * num_heads                                      # qkt_matmul
    act_size += input_len * hidden_size                                             # pv_matmul
    act_size += input_len * hidden_size                                             # o_proj
    act_size += input_len * hidden_size                                             # post_layernorm
    act_size += input_len * intermediate_size * 2                                   # up_proj, gate_proj
    act_size += input_len * hidden_size                                             # down_proj

    act_size += input_len * vocab_size                                              # lm_head
    act_size *= (bit / 8)
    act_size *= batch_size
    return act_size


def eagle3_mm_comp(args, input_len, kv_len, first_layer:bool):
    """
    first_layer: 如果是在生成draft token tree的第一层，需要经过fc层的映射。
    """
    bit = args.weight_bit
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    num_heads = args.num_heads
    vocab_size = args.vocab_size
    batch_size = args.batch_size
    kv_scale = args.kv_scale

    computation = 0
    kv_len = kv_len + input_len

    if first_layer:
        computation += input_len * hidden_size * hidden_size * 3                          # fc
    
    computation += input_len * (hidden_size * 2) * hidden_size * 2                        # q_proj
    computation += input_len * (hidden_size * 2) * hidden_size * kv_scale * 2             # k_proj, v_proj
    computation += input_len * kv_len * hidden_size * 2                                   # qkt_matmul, pv_matmul
    computation += input_len * hidden_size * hidden_size                                  # o_proj
    computation += input_len * hidden_size * intermediate_size * 3                        # up_proj, gate_proj, down_proj
    computation += input_len * hidden_size * vocab_size                                   # lm_head

    computation *= batch_size
    return computation


def eagle3_cycles_comp(args, input_len, kv_len, first_layer:bool):
    """
    这是单独跑一次eagle3小模型的时间。即: 生成draft token树的某一层。
    """
    # LD
    hbm_trans_compatibility = args.hbm_bandwidth * (1.024**3) * 1000 / args.clock_frequency # how many bytes can be transferred in one cycle.
    weight_size = eagle3_weight_load_size(args, input_len, first_layer=first_layer)
    ld_weight_cycle = weight_size / (hbm_trans_compatibility * args.num_wide_channels / args.num_hbm_channels * args.hbm_same_uti)

    act_size = eagle3_act_load_size(args, input_len, kv_len, first_layer=first_layer)
    ld_act_cycle = act_size / (hbm_trans_compatibility * args.num_narrow_channels / args.num_hbm_channels * args.hbm_cross_uti)

    ld_cycle = (ld_weight_cycle + ld_act_cycle)

    # ST
    act_size = eagle3_act_st_size(args, input_len, kv_len, first_layer=first_layer)
    st_cycle = act_size / (hbm_trans_compatibility * args.num_narrow_channels / args.num_hbm_channels * args.hbm_same_uti)

    # MM/MV
    computation = eagle3_mm_comp(args, input_len, kv_len, first_layer=first_layer)
    comp_cycle = computation / (args.mm_parallel_m * args.mm_parallel_n * args.mm_parallel_k * args.num_slr)

    # FUSE
    fused_cycle = max(ld_cycle, st_cycle, comp_cycle)

    return ld_cycle, st_cycle, comp_cycle, fused_cycle


def eagle3_draft_cycles_comp(args, input_len, kv_len):
    """
    这是eagle3小模型生成一整棵token tree花的cycles。假如depth=6，那么它就要跑7次。
    input_len是生成第一层节点时的输入token数（也可以理解为上一个iteration 最终接受的token数)。
    kv_len的话，我们就假定在生成这棵树的过程中不变。因为在生成一棵树的过程中，kv_len变化有限，几乎不影响各个执行时间。
    """
    LD_CYCLES = 0
    ST_CYCLES = 0
    COMP_CYCLES = 0
    FUSED_CYCLES = 0

    # 1. 生成树的第一层节点
    ld_cycle, st_cycle, comp_cycle, fused_cycle = eagle3_cycles_comp(args, input_len, kv_len, first_layer=True)
    LD_CYCLES += ld_cycle
    ST_CYCLES += st_cycle
    COMP_CYCLES += comp_cycle
    FUSED_CYCLES += fused_cycle

    # 2. 生成树的后"depth"层节点
    for i in range(args.depth):
        # 这里的kv_len是指当前层的kv cache长度。
        ld_cycle, st_cycle, comp_cycle, fused_cycle = eagle3_cycles_comp(args, args.top_k, kv_len, first_layer=False)
        LD_CYCLES += ld_cycle
        ST_CYCLES += st_cycle
        COMP_CYCLES += comp_cycle
        FUSED_CYCLES += fused_cycle
    
    return LD_CYCLES, ST_CYCLES, COMP_CYCLES, FUSED_CYCLES