import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def create_dataset(df, time_step=1, feature_column='value'):
    """
    Prepare the dataset for training by creating sequences for LSTM.

    Parameters
    ----------
    df : pandas.DataFrame
        The data containing the time series.
    time_step : int, optional
        The number of previous time steps to use for predicting the next time step.
    feature_column : str, optional
        The column name to predict.

    Returns
    -------
    X : numpy.ndarray
        The input features for training/testing.
    y : numpy.ndarray
        The target labels for training/testing.
    """
    X, y = [], []
    for i in range(len(df) - time_step):
        X.append(df[feature_column].iloc[i:i + time_step].values)
        y.append(df[feature_column].iloc[i + time_step])

    return np.array(X), np.array(y)


def train_test_predict_lstm(df, feature_column='value', time_step=1, test_size=0.1, epochs=50, batch_size=32, forecast_years=13):
    """
    Function to train, test, and predict using an LSTM model.

    Parameters
    ----------
    df : pandas.DataFrame
        The data containing the time series.
    feature_column : str, optional
        The column name to predict. The default is 'value'.
    time_step : int, optional
        The number of previous time steps to use for predicting the next time step.
    test_size : float, optional
        The proportion of the dataset to use as test data. The default is 0.2.
    epochs : int, optional
        Number of epochs for training. The default is 50.
    batch_size : int, optional
        The batch size for training. The default is 32.
    forecast_years : int, optional
        The number of future years to predict. Default is 13.

    Returns
    -------
    model : keras.Model
        The trained LSTM model.
    y_test : numpy.ndarray
        The actual target values for testing.
    y_pred : numpy.ndarray
        The predicted values.
    forecast : numpy.ndarray
        The predicted future values for the specified number of years.
    """


    # Scale the feature data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = pd.DataFrame(scaler.fit_transform(df[feature_column].values.reshape(-1, 1)), columns=[feature_column])

    # Prepare the data for LSTM
    X, y = create_dataset(df_scaled, time_step, feature_column)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    # Reshape X_train and X_test for LSTM input
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build LSTM Model
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=False, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # Make predictions on the test set
    y_pred_scaled = model.predict(X_test)

    # Inverse transform to get actual values
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_actual = scaler.inverse_transform(y_pred_scaled)

    # Calculate the performance (e.g., RMSE)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Forecast the future (next 13 years)
    forecast_input = df_scaled[-time_step:].values.reshape(1, time_step, 1)
    forecast = []

    for _ in range(forecast_years):
        next_pred_scaled = model.predict(forecast_input)
        forecast.append(next_pred_scaled[0, 0])
        forecast_input = np.append(forecast_input[:, 1:, :], next_pred_scaled.reshape(1, 1, 1), axis=1)

    # Inverse transform forecasted values
    forecast_actual = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

    return model, y_test_actual, y_pred_actual, forecast_actual


# Load the data
file_path = 'uk_electricity_1990_2022.csv'
df = pd.read_csv(file_path, encoding='gbk')

# Specify the features to predict
features = ['hydro_electricity_%', 'nuclear_electricity_%', 'solar_electricity_%', 'wind_electricity_%']

# Train, test, and predict for each feature
forecast_results = {}
for feature in features:
    model, y_test, y_pred, forecast = train_test_predict_lstm(df, feature_column=feature, time_step=5, epochs=100, batch_size=8, forecast_years=13)
    forecast_results[feature] = forecast

# 计算每年各个能源的总和
df['sum_renewable_%'] = df['hydro_electricity_%'] + df['nuclear_electricity_%'] + df['solar_electricity_%'] + df['wind_electricity_%']

# 同样，计算预测数据的总和
forecast_results['sum_renewable_%'] = forecast_results['hydro_electricity_%'] + forecast_results['nuclear_electricity_%'] + forecast_results['solar_electricity_%'] + forecast_results['wind_electricity_%']

# Plot actual and forecasted values
plt.figure(figsize=(14, 8))
ax = plt.gca()

# Plot the actual data for each feature
df[(df['year'] >= 1990) & (df['year'] <= 2022)].plot(
    kind='line',
    x='year',
    y=features + ['sum_renewable_%'],  # Add sum_renewable_% to the features
    ax=ax,
    title='Electricity - United Kingdom',
    grid=True
)

# Plot the forecasted data for each feature and sum
years = np.arange(df['year'].iloc[-1] + 1, df['year'].iloc[-1] + 1 + 13)
for feature in features:
    ax.plot(years, forecast_results[feature], label=f'{feature} Forecasted', linestyle='--')

# Plot the forecasted sum of renewables
ax.plot(years, forecast_results['sum_renewable_%'], label='Sum of Renewables Forecasted', linestyle='--', color='black')

# Add legends and grid
plt.legend()
plt.grid(True)
plt.show()