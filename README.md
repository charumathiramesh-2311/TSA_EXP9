# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```c

from IPython import get_ipython
from IPython.display import display

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


data = pd.read_csv("/content/IOT-temp.csv.zip")

data['noted_date'] = pd.to_datetime(data['noted_date'], format='%d-%m-%Y %H:%M')
data.set_index('noted_date', inplace=True)

daily_temp = data['temp'].resample('D').mean().dropna()

def arima_model(data, order):
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    model = ARIMA(train_data, order=order)
    fitted_model = model.fit()

    forecast = fitted_model.forecast(steps=len(test_data))

    rmse = np.sqrt(mean_squared_error(test_data, forecast))
    print("Root Mean Squared Error (RMSE):", rmse)

    # Plotting code moved inside the function and indentation corrected
    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data, label='Training Data')
    plt.plot(test_data.index, test_data, label='Testing Data')
    plt.plot(test_data.index, forecast, label='Forecasted Data', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.title('ARIMA Forecasting on IoT Temperature Data')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


arima_model(daily_temp, order=(5,1,0))


```
### OUTPUT:
![image](https://github.com/user-attachments/assets/0de5fb0c-19a7-4652-bfa5-c27c0105b2d6)


### RESULT:
Thus the program run successfully based on the ARIMA model using python.
