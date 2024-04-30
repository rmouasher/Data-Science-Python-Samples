import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

def fetch_interest_rate_data():
    dates = pd.date_range(start='2010-01-01', periods=120, freq='M')
    interest_rates = np.random.normal(loc=2, scale=0.5, size=120)
    data = pd.DataFrame(data={'Date': dates, 'Interest_Rate': interest_rates})
    data.set_index('Date', inplace=True)
    return data

data = fetch_interest_rate_data()

print("Basic Statistics of Interest Rates:")
print(data.describe())
data.plot(title="Historical Interest Rates")
plt.xlabel('Date')
plt.ylabel('Interest Rate (%)')
plt.show()

model = ARIMA(data, order=(1, 1, 1))
model_fit = model.fit()

forecast = model_fit.forecast(steps=12)
forecast_dates = pd.date_range(start=data.index[-1], periods=13, freq='M')[1:]

plt.figure(figsize=(10, 5))
plt.plot(data.index, data['Interest_Rate'], label='Historical Data')
plt.plot(forecast_dates, forecast, label='Forecast', color='red')
plt.title('Interest Rate Forecast')
plt.xlabel('Date')
plt.ylabel('Interest Rate (%)')
plt.legend()
plt.show()