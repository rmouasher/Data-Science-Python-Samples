import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

def generate_interest_rate_data():
    np.random.seed(0)
    dates = pd.date_range(start='2000-01-01', periods=240, freq='M')
    trend = np.linspace(start=1.5, stop=4.5, num=240)
    seasonal = 1.5 * np.sin(np.linspace(0, 20 * np.pi, 240))
    noise = np.random.normal(loc=0, scale=0.25, size=240)
    interest_rates = trend + seasonal + noise
    return pd.Series(interest_rates, index=dates)

data = generate_interest_rate_data()

result = seasonal_decompose(data, model='additive', period=12)

result.plot()
plt.show()