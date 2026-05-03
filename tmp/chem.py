from scipy import stats
import pandas as pd
import numpy as np
# x = [0,
# 0.6936,
# 1.61,
# 2.30,
# 3.91]
# 
# x = pd.DataFrame(x)
# 
# y = [
#     1.35,
# 11.6,
# 17.7,
# 30.8,
# 52.9
# ]
# 
# y = pd.DataFrame(y)
# 
# y = y*1e-3
# 
# print(y)
# 
# res = stats.linregress(x, y)
# print(res)
# print(res.slope/res.stderr)

data = [np.float64(-0.1363659146543046), np.float64(0.9827380919459611), np.float64(0.9827816428313512), np.float64(0.9827820783326882), np.float64(0.9827820826876263), np.float64(0.982782082731175), np.float64(0.9827820827316105), np.float64(0.9827820827316148), np.float64(0.9827820827316149), np.float64(0.9827820827316149), np.float64(0.9827820827316149)]
for i in range(1, len(data)):
    print(i, 10**(-i), data[i] - data[i-1])