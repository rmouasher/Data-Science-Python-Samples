import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
dates = pd.date_range(start='2010-01-01', periods=100, freq='M')
series1 = pd.Series(np.random.normal(loc=0, scale=1, size=100).cumsum(), index=dates)
series2 = pd.Series(np.random.normal(loc=0, scale=1, size=100).cumsum(), index=dates)

cross_corr = [series1.corr(series2.shift(lag)) for lag in range(-10, 11)]

plt.figure(figsize=(8, 4))
plt.stem(range(-10, 11), cross_corr, use_line_collection=True)
plt.title('Cross-Correlation between Series 1 and Series 2')
plt.xlabel('Lag')
plt.ylabel('Correlation Coefficient')
plt.show()