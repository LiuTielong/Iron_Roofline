"""
LongSpec for llama-3.1-8b模型的仿真器。
小模型的结构：
1. self_attn层：它的kv cache长度最长是512。进去之后先经过layernorm。各个模块（q,k,v,o）的维度都是和llama-3.1-8b的一个attention模块的维度一样。
2. cross-attn层：使用大模型的最后一个decoder layer的kv cache. 长度不限制。进去之后也先进行layernorm计算。在代码中冗余计算了key, values，我在仿真器中去掉这个冗余计算的部分。
3. ffn层。就是一个llamaMLP。记得要先经过一个layernorm层。
4. 小模型的输入：由大模型的embedding层将input_ids映射，同时还会使用位置编码函数给小模型提供sin, cos位置编码（不过这个数据量比较小，我先忽略）。
5. 小模型的输出：经过ffn层之后没有norm层，而是将[batch_size, input_len, hidden_size]这个张量直接给大模型的lm_head层得到预测的结果。
"""

import math

def longspec_weight_load_size(args, input_len):
    batch_size = args.batch_size
    loop = math.ceil(input_len * batch_size / args.MM_START_M_NUMBER)
    bit = args.weight_bit
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    vocab_size = args.vocab_size
    kv_scale = args.kv_scale  # kv_scale=1/4, 即: k_proj, v_proj的大小只有q_proj的1/4

    weight_size = 0
    # self-attn
    weight_size += hidden_size * hidden_size * (2 + 2 * kv_scale)                   # q, k, v, o_proj
    # cross-attn
    # 注意：我们这里省略对k, v的冗余计算
    weight_size += hidden_size * hidden_size * 2                                    # q, o_proj
    # ffn
    weight_size += hidden_size * intermediate_size * 3                              # up_proj, gate_proj, down_proj
    weight_size += hidden_size * vocab_size                                         # lm_head
    
    weight_size *= loop
    weight_size *= (bit / 8)
    return weight_size


def longspec_act_load_size(args, input_len, kv_len):
    """
    注意：self-attn的kv cache长度最长是512。cross-attn的kv cache长度不限制。
    """
    bit = args.act_bit
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    num_heads = args.num_heads
    vocab_size = args.vocab_size
    batch_size = args.batch_size
    kv_scale = args.kv_scale
    self_kv_len = min(512, input_len + kv_len)
    cross_kv_len = input_len + kv_len

    act_size = 0
    # self-attn
    act_size += input_len * hidden_size                                             # input layernorm的输入
    act_size += input_len * hidden_size * 3                                         # q, k, v的输入
    act_size += hidden_size * self_kv_len * 2 * kv_scale                            # kv cache加载
    act_size += input_len * hidden_size * (1 + kv_scale)                            # qkt_matmul (这里本来是加载整个k cache的，但是我上一行加载了)
    act_size += (input_len * self_kv_len * num_heads + input_len * hidden_size * kv_scale) # pv_matmul
    act_size += input_len * hidden_size * 2                                         # o_proj and residual
    # cross-attn
    act_size += input_len * hidden_size                                             # input layernorm的输入
    act_size += input_len * hidden_size                                             # q的输入
    act_size += hidden_size * kv_len * 2 * kv_scale                                 # kv cache加载
    act_size += input_len * hidden_size * (1 + kv_scale)                            # qkt_matmul
    act_size += (input_len * cross_kv_len * num_heads + input_len * hidden_size * kv_scale) # pv_matmul
    act_size += input_len * hidden_size * 2                                         # o_proj and residual
    # ffn
    act_size += input_len * hidden_size                                             # post_layernorm
    act_size += input_len * hidden_size                                             # gate_proj
    act_size += input_len * hidden_size + input_len * intermediate_size             # up_proj
    act_size += (input_len * intermediate_size + input_len * hidden_size)           # down_proj and residual

    act_size += input_len * hidden_size                                             # lm_head
    act_size *= (bit / 8)                                                       
    act_size *= batch_size
    return act_size


def longspec_act_st_size(args, input_len, kv_len):
    bit = args.act_bit
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    num_heads = args.num_heads
    vocab_size = args.vocab_size
    batch_size = args.batch_size
    kv_scale = args.kv_scale
    self_kv_len = min(512, input_len + kv_len)
    cross_kv_len = input_len + kv_len

    act_size = 0
    # self-attn
    act_size += input_len * hidden_size                                             # input layernorm的输出
    act_size += input_len * hidden_size * (1 + 2 * kv_scale)                        # q, k, v的输出
    act_size += input_len * self_kv_len * num_heads                                      # qkt_matmul的输出
    act_size += input_len * hidden_size                                             # pv_matmul的输出
    act_size += input_len * hidden_size                                             # o_proj的输出
    # corss-attn
    act_size += input_len * hidden_size                                             # input layernorm的输出
    act_size += input_len * hidden_size                                             # q的输出
    act_size += input_len * cross_kv_len * num_heads                                      # qkt_matmul的输出
    act_size += input_len * hidden_size                                             # pv_matmul的输出
    act_size += input_len * hidden_size                                             # o_proj的输出
    # ffn
    act_size += input_len * hidden_size                                             # post_layernorm的输出
    act_size += input_len * intermediate_size * 2                                   # up_proj, gate_proj
    act_size += input_len * hidden_size                                             # down_proj

    act_size += input_len * vocab_size                                              # lm_head
    act_size *= batch_size
    act_size *= (bit / 8)
    return act_size


def longspec_mm_comp(args, input_len, kv_len):
    bit = args.act_bit
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    num_heads = args.num_heads
    vocab_size = args.vocab_size
    kv_scale = args.kv_scale
    batch_size = args.batch_size

    self_kv_len = min(512, input_len + kv_len)
    cross_kv_len = input_len + kv_len
    computation = 0

    # self-attn
    computation += input_len * hidden_size * hidden_size * (1 + 2 * kv_scale)                   # q, k, v generation
    computation += input_len * self_kv_len * hidden_size * 2                                    # qkt_matmul, pv_matmul
    computation += input_len * hidden_size * hidden_size                                        # o_proj
    # cross-attn
    computation += input_len * hidden_size * hidden_size                                        # q generation
    computation += input_len * cross_kv_len * hidden_size * 2                                   # qkt_matmul, pv_matmul
    computation += input_len * hidden_size * hidden_size                                        # o_proj
    # ffn
    computation += input_len * hidden_size * intermediate_size * 3                              # up_proj, gate_proj, down_proj

    computation += input_len * hidden_size * vocab_size                                         # lm_head
    computation *= batch_size
    return computation


def longspec_cycles_comp(args, input_len, kv_len):
    """
    生成draft token树的某一层
    """
    # LD
    hbm_trans_compatibility = args.hbm_bandwidth * (1.024**3) * 1000 / args.clock_frequency # how many bytes can be transferred in one cycle.
    weight_size = longspec_weight_load_size(args, input_len)
    ld_weight_cycle = weight_size / (hbm_trans_compatibility * args.num_wide_channels / args.num_hbm_channels * args.hbm_same_uti)

    act_size = longspec_act_load_size(args, input_len, kv_len)
    ld_act_cycle = act_size / (hbm_trans_compatibility * args.num_narrow_channels / args.num_hbm_channels * args.hbm_cross_uti)

    ld_cycle = (ld_weight_cycle + ld_act_cycle)

    # ST
    act_size = longspec_act_st_size(args, input_len, kv_len)
    st_cycle = act_size / (hbm_trans_compatibility * args.num_narrow_channels / args.num_hbm_channels * args.hbm_same_uti)

    # MM/MV
    computation = longspec_mm_comp(args, input_len, kv_len)
    comp_cycle = computation / (args.mm_parallel_m * args.mm_parallel_n * args.mm_parallel_k * args.num_slr)

    # FUSE
    fused_cycle = max(ld_cycle, st_cycle, comp_cycle)

    return ld_cycle, st_cycle, comp_cycle, fused_cycle


def longspec_draft_cycles_comp(args, input_len, kv_len, method="seq"):
    """
    这是longspec一整个draft过程花费的cycles。
    method: [seq, tree].
    我们先使用"seq"，比较简单，可以创建和MagicDec相似的模型。
    后面再考虑使用"tree"。
    """
    LD_CYCLES = 0
    ST_CYCLES = 0
    COMP_CYCLES = 0
    FUSED_CYCLES = 0

    if method == "seq":
        # 1. 生成链的第一个节点
        ld_cycle, st_cycle, comp_cycle, fused_cycle = longspec_cycles_comp(args, input_len, kv_len)
        LD_CYCLES += ld_cycle
        ST_CYCLES += st_cycle
        COMP_CYCLES += comp_cycle
        FUSED_CYCLES += fused_cycle

        # 2. 生成树的后"depth"层节点
        for i in range(args.depth):
            # 这里的kv_len是指当前层的kv cache长度。
            ld_cycle, st_cycle, comp_cycle, fused_cycle = longspec_cycles_comp(args, 1, kv_len)
            LD_CYCLES += ld_cycle
            ST_CYCLES += st_cycle
            COMP_CYCLES += comp_cycle
            FUSED_CYCLES += fused_cycle

    elif method == "tree":
        raise NotImplementedError("Tree method is not implemented yet.")
    else:
        raise ValueError("method should be either 'seq' or 'tree'.")
    
    return LD_CYCLES, ST_CYCLES, COMP_CYCLES, FUSED_CYCLES
