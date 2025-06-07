import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from ..config import DATA_FILE

def load_and_prepare_data(filename=DATA_FILE):
    """Загрузка и подготовка данных (F1)"""
    try:
        data = pd.read_csv(filename, parse_dates=['Дата'])
        return data
    except FileNotFoundError:
        return generate_demo_data()

def generate_demo_data():
    initial_products = [
        "Молоко Простоквашино 2.5%",
        "Хлеб Бородинский нарезка",
        "Яйца куриные С1 10шт",
        "Сахар песок 1кг",
        "Макароны Barilla перья"
    ]
    
    all_data = []
    for product in initial_products:
        current_stock = np.random.randint(150, 300)
        dates = [datetime.now() - timedelta(days=i) for i in range(180, 0, -1)]

        product_data = {
            "Дата": dates,
            "Товар": [product] * 180,
            "Цена": np.round(np.random.uniform(50, 500, size=180), 2),
            "Продажи": np.random.randint(1, 20, size=180),
            "Количество_товара": np.random.randint(50, 200, size=180)
        }
        all_data.append(pd.DataFrame(product_data))
    
    df = pd.concat(all_data, ignore_index=True)
    df.to_csv(DATA_FILE, index=False)
    return df