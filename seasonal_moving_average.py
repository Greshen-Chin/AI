# seasonal_moving_average.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_dataset(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    return df


def seasonal_moving_average(df, season=7):
    """
    Calculate moving average with defined season (e.g., 7 days).
    """
    df['SMA'] = df['value'].rolling(window=season).mean()
    return df


def plot_results(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df['date'], df['value'], label="Actual")
    plt.plot(df['date'], df['SMA'], label=f"SMA", linewidth=2)
    plt.legend()
    plt.title("Seasonal Moving Average")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.show()


if name == "main":
    df = load_dataset("indonesia_supermarket_5yr_synthetic.csv")

    df = seasonal_moving_average(df, season=7)
    plot_results(df)