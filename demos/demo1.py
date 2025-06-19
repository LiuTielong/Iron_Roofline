"""
这个demo是已知一组verify length, AAT, tree shape的情况下, 计算对应的FPGA上的性能。 
如果已知baseline的性能, 那么就可以计算出加速比。
"""

import sys
sys.path.append("./")
sys.path.append("../")
import pandas as pd
from roofline.draw import draw_roofline, draw_roofline_discount, draw_acc, draw_combined_model
from modeling.parse_args import parse_args
from modeling.modeling_longspec import longspec_draft_cycles_comp
from modeling.modeling_llama3 import llama3_cycles_comp
import numpy as np

verify_lengths = [
    1+1,
2+1,
3+1,
4+1,
5+1,
6+1,
8+1,
10+1,
12+1,
14+1,
16+1,
18+1,
20+1,
25+1,
30+1,
35+1,
40+1,
45+1,
50+1,
60+1,
70+1,
80+1,
]
AATs = [
    1.740,
2.194,
2.291,
2.396,
2.682,
2.721,
2.883,
2.930,
3.160,
3.259,
3.326,
3.326,
3.452,
3.531,
3.608,
3.703,
3.693,
3.924,
3.981,
4.064,
4.136,
4.222,
]

tree_shapes = [
"1 0",
"1 1",
"2 1",
"2 2 ",
"2 2 1",
"3 2 1",
"3 3 2",
"4 4 2",
"4 4 3 1",
"4 5 3 2",
"4 6 4 2",
"4 7 5 2",
"4 8 5 2 1",
"4 9 8 3 1",
"4 10 10 5 1",
"4 12 11 6 2",
"4 14 12 8 2",
"4 15 12 8 4 2",
"4 15 13 11 5 2",
"4 16 16 14 8 2",
"4 16 16 16 13 5",
"4 16 16 16 15 3",
]

parser = parse_args()
parser.add_argument("--avg_accepted_tokens",    type=int,   default=4   ,     help="the average number of accepted tokens per iteration."         )
parser.add_argument("--gamma",                  type=int,   default=1,        help="it's similar to total_tokens, (depth+1) in eagle algorithm."  )
parser.add_argument("--tree_shape", nargs="+",  type=int, default=[4, 16, 16, 16, 16], help="the tree shape of the draft token tree." )
args = parser.parse_args()
args.batch_size = 1

verify_times = []
draft_times = []
for total_token, AAT, tree_shape in zip(verify_lengths, AATs, tree_shapes):
    aat = int(AAT)
    args.tree_shape = [int(x) for x in tree_shape.split()]
    args.prompt_len = 8192
    # 小模型的draft阶段
    _, _, _, fused_draft_cycles = longspec_draft_cycles_comp(args, input_len=aat, kv_len=args.prompt_len, method="tree")
    # 大模型的verify阶段
    _, _, _, fused_verify_cycles = llama3_cycles_comp(args, input_len=total_token, kv_len=args.prompt_len)
    draft_time = fused_draft_cycles / (args.clock_frequency * 1e6)
    verify_time = fused_verify_cycles / (args.clock_frequency * 1e6)
    verify_times.append(verify_time)
    draft_times.append(draft_time)

for verify_time, draft_time, aat in zip(verify_times, draft_times, AATs):
    performance = aat / (verify_time + draft_time)

    speedup = performance / 18.92
    print(f"{speedup:.3f}")