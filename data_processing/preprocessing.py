import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

def feature_engineering(data):
    """Инженерия признаков"""
    data['День_недели'] = data['Дата'].dt.dayofweek
    data['День_месяца'] = data['Дата'].dt.day
    data['Месяц'] = data['Дата'].dt.month
    data['Неделя_года'] = data['Дата'].dt.isocalendar().week

    data = data.sort_values(['Товар', 'Дата'])
    for lag in [1, 2, 3, 7, 14]:
        data[f'Продажи_лаг_{lag}'] = data.groupby('Товар')['Продажи'].shift(lag)
        data[f'Остаток_лаг_{lag}'] = data.groupby('Товар')['Количество_товара'].shift(lag)

    for window in [3, 7, 14]:
        data[f'Ср_продажи_{window}д'] = data.groupby('Товар')['Продажи'].transform(
            lambda x: x.rolling(window).mean()
        )

    data = data.dropna()
    data['Цена_продажи_ratio'] = data['Цена'] / (data['Продажи'] + 1)
    return data

def check_stationarity(data, product_name=None, apply_diff=True, verbose=False):
    """Проверка стационарности временного ряда (F2)"""
    if product_name:
        product_data = data[data['Товар'] == product_name].set_index('Дата')
        series_sales = product_data['Продажи']
        series_stock = product_data['Количество_товара']
        title_prefix = f"Товар: {product_name}"
    else:
        aggregated = data.groupby('Дата').agg({'Продажи': 'sum', 'Количество_товара': 'sum'})
        series_sales = aggregated['Продажи']
        series_stock = aggregated['Количество_товара']
        title_prefix = "Все товары (агрегировано)"

    if apply_diff:
        series_sales, sales_diff_count, is_stationary_sales = make_stationary(series_sales, verbose=verbose)
        series_stock, stock_diff_count, is_stationary_stock = make_stationary(series_stock, verbose=verbose)
    else:
        is_stationary_sales = adfuller(series_sales.dropna())[1] < 0.05
        is_stationary_stock = adfuller(series_stock.dropna())[1] < 0.05
        sales_diff_count = 0
        stock_diff_count = 0

    plt.figure(figsize=(14, 6))
    plt.plot(series_sales, label=f'Продажи (diff={sales_diff_count})')
    plt.plot(series_stock, label=f'Остатки (diff={stock_diff_count})')
    plt.title(f'{title_prefix} - Стационарность после дифференцирования\n'
              f'Продажи стационарны: {is_stationary_sales} | Остатки стационарны: {is_stationary_stock}')
    plt.legend()
    plt.grid()
    plt.show()

    return is_stationary_sales, is_stationary_stock

def make_stationary(series, max_diff=3, verbose=False):
    """Дифференцирование временного ряда"""
    for d in range(max_diff + 1):
        adf_result = adfuller(series.dropna())
        p_value = adf_result[1]
        if verbose:
            print(f"Дифференциация {d} раз: p-value = {p_value:.4f}")
        if p_value < 0.05:
            return series, d, True
        series = series.diff()
    return series, max_diff, False