from scipy import stats
import pandas as pd
x = [0,
0.6936,
1.61,
2.30,
3.91]

x = pd.DataFrame(x)

y = [
    1.35,
11.6,
17.7,
30.8,
52.9
]

y = pd.DataFrame(y)

y = y*1e-3

print(y)

res = stats.linregress(x, y)
print(res)
print(res.slope/res.stderr)