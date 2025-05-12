import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # 1. llama-3.1-8B-instruct model parameters
    parser.add_argument("--hidden_size",            type=int,   default=4096                                                                                      )
    parser.add_argument("--intermediate_size",      type=int,   default=14336                                                                                     )
    parser.add_argument("--num_layers",             type=int,   default=32                                                                                        )
    parser.add_argument("--num_heads",              type=int,   default=32                                                                                        )
    parser.add_argument("--head_dim",               type=int,   default=128,      help="head_dim = hidden_size / num_heads."                                      )
    parser.add_argument("--vocab_size",             type=int,   default=128256                                                                                    )
    parser.add_argument("--kv_scale",               type=float, default=0.25,     help="kv_scale = kv_heads / num_heads"                                          )
    # algorithm parameters   
    parser.add_argument("--prompt_len",             type=int,   default=32000                                                                                     )
    parser.add_argument("--generation_len",         type=int,   default=128                                                                                       )   
    parser.add_argument("--batch_size",             type=int,   default=128                                                                                       )
    # hardware parameters
    parser.add_argument("--weight_bit",             type=int,   default=16,        help="the bit of weight of the base LLM."                                       )
    parser.add_argument("--act_bit",                type=int,   default=16,        help="the bit of weight of the base LLM."                                       )
    parser.add_argument("--MM_START_M_NUMBER",      type=int,   default=128,      help="the number of activations."                                               )
    parser.add_argument("--num_slr",                type=int,   default=4,        help="the number of Super Logic regions on FPGA."                               )
    parser.add_argument("--hbm_bandwidth",          type=int,   default=819.2 ,   help="The bandwidth of HBM on FPGA, whose unit is GB/s."                        )
    parser.add_argument("--num_hbm_channels",       type=int,   default=32,       help="the number of HBM channels."                                              )
    parser.add_argument("--num_wide_channels",      type=int,   default=32,       help="the number of HBM channels used for activation in prefill stage."         )
    parser.add_argument("--num_narrow_channels",    type=int,   default=4,        help="the number of HBM channels used for activation in decode stage."          )
    parser.add_argument("--hbm_same_uti",           type=float, default=0.7,      help="The same channel utilization of HBM on FPGA."                             )
    parser.add_argument("--hbm_cross_uti",          type=float, default=0.35,     help="the cross channel utilization of HBM on FPGA."                            )
    
    parser.add_argument("--clock_frequency",        type=int,   default=225,      help="the clock frequency of FPGA, whose unit is MHz."                          )
    parser.add_argument("--mm_parallel_m",          type=int,   default=128,      help="the parallel m of matrix-matrix multiplication."                          )
    parser.add_argument("--mm_parallel_k",          type=int,   default=16,       help="the parallel k of matrix-matrix multiplication."                          )
    parser.add_argument("--mm_parallel_n",          type=int,   default=2,        help="the parallel n of matrix-matrix multiplication."                          )
    parser.add_argument("--mv_parallel_m",          type=int,   default=1,        help="the parallel m of matrix-vector multiplication."                          )
    parser.add_argument("--mv_parallel_k",          type=int,   default=16,       help="the parallel k of matrix-vector multiplication."                          )
    parser.add_argument("--mv_parallel_n",          type=int,   default=64,       help="the parallel n of matrix-vector multiplication."                          )
    
    # 不同算法各自的超参数让它们自己实现吧
    return parser