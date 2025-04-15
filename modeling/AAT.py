"""
这里创建一些AAT-verified_tokens数据。
都用字典表示比较规范。
"""
# 算法: eagle2.
# 情境：短文本。
# 数据集：就是eagle2常用的数据集：MT-Bench.
eagle2_aat = {1: 1, 2: 1.75, 3: 2.225, 4: 2.522, 5: 2.718, 
              6: 2.851, 7: 2.969, 8: 3.052, 9: 3.113, 10: 3.191, 
              12: 3.288, 14: 3.392, 16: 3.445, 18: 3.508, 20: 3.569, 
              23: 3.612, 26: 3.707, 29: 3.746, 32: 3.799, 35: 3.827, 
              38: 3.87, 41: 3.896, 45: 3.949, 50: 3.981, 55: 4.037, 
              60: 4.087, 65: 4.106, 70: 4.14, 75: 4.143, 80: 4.176, 
              85: 4.186, 90: 4.211, 95: 4.234, 100: 4.254}

# 算法：eagle3.
# 情境：短文本。
# 数据集：就是eagle2常用的数据集：MT-Bench.
# 数据是80条。
eagle3_aat = {1: 1, 2: 1.824, 3: 2.488, 4: 3.014, 5: 3.473, 
              6: 3.814, 7: 4.093, 8: 4.364, 9: 4.5, 10: 4.607, 
              12: 4.807, 14: 4.959, 16: 5.057, 18: 5.155, 20: 5.255, 
              23: 5.332, 26: 5.455, 29: 5.508, 32: 5.572, 35: 5.617, 
              38: 5.689, 41: 5.708, 45: 5.745, 50: 5.819, 55: 5.897, 
              60: 5.888, 65: 5.951, 70: 5.994, 75: 5.99, 80: 5.991, 
              85: 6.041, 90: 6.062, 95: 6.051, 100: 6.086}

# 算法：MagicDec.
# 情境：长文本（prefill_length=32000).
# 数据集：pg-19 test.
# 采样了80条数据。生成长度都是128.
magicdec_aat = {1: 1.811, 2: 2.476, 3: 3.783, 4: 3.408, 5: 2.982, 
                6: 4.172, 7: 4.324, 8: 4.698, 9: 4.825, 10: 4.793, 
                11: 4.889, 12: 5.038, 13: 5.094, 14: 5.068, 15: 5.353, 
                16: 5.468, 17: 5.42, 18: 5.472, 19: 5.155, 20: 5.454}

# 算法：LongSpec。
# 情境：长文本（prefill_length=32000).
# 数据集：gov_report.
# 只有3条测试数据。但是生成长度都是1024.
prefill_lengths = list(range(1, 21))
