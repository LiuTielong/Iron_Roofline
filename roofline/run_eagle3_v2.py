"""
本文件对EAGLE-3算法构建长短文本下的所有模型。
Batch size恒定为1.
使用的AAT数据来自: ./Data/表3-5.xlsx, ./Data/表3-6.xlsx.
论文默认配置: total_token=60.
希望我搜索出来的配置能超过默认配置。
"""

import sys
sys.path.append("./")
sys.path.append("../")
import pandas as pd
