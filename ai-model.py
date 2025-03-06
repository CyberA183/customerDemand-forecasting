import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['month_year'], index_col='month_year')
    return df

def preprocess_data(df, feature='demand', time_steps=10):
    scaler = MinMaxScaler()
    df[feature] = scaler.fit_transform(df[[feature]])
    
    X, y = [], []
    for i in range(len(df) - time_steps):
        X.append(df[feature].values[i:i+time_steps])
        y.append(df[feature].values[i+time_steps])
    
    return np.array(X), np.array(y), scaler

def build_model(time_steps):
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(time_steps, 1)),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(model, X_train, y_train, epochs=1, batch_size=16):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

def forecast(model, last_data, scaler, steps=7):
    predictions = []
    current_input = last_data.reshape(1, -1, 1)
    
    for _ in range(steps):
        pred = model.predict(current_input, verbose=0)
        predictions.append(pred[0, 0])
        current_input = np.append(current_input[:, 1:, :], [[pred]], axis=1)
    
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

df = load_data('/Users/arielthompson/Downloads/Data_Set1.csv')
X, y, scaler = preprocess_data(df)
X_train, y_train = X.reshape(-1, 10, 1), y
model = build_model(10)
train_model(model, X_train, y_train)
forecast_values = forecast(model, X[-1], scaler)
