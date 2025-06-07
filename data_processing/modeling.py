import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from ..config import DATA_FILE

def add_new_product(product_name, initial_stock, price, sales_per_day, days_to_add):
    """Добавление нового товара (F3)"""
    try:
        df = pd.read_csv(DATA_FILE, parse_dates=['Дата'])
        last_date = df['Дата'].max()
        new_dates = [last_date + timedelta(days=i) for i in range(1, days_to_add+1)]
        
        current_stock = initial_stock
        new_data = []
        
        for date in new_dates:
            max_sale = min(current_stock, int(current_stock * np.random.uniform(0.62, 0.81)))
            daily_sale = np.random.randint(int(0.5*max_sale), max_sale)
            
            new_data.append({
                "Дата": date,
                "Товар": product_name,
                "Цена": price,
                "Продажи": daily_sale,
                "Количество_товара": current_stock
            })
            
            current_stock -= daily_sale
            
            if len(new_data) % np.random.randint(3, 6) == 0 or current_stock < 50:
                current_stock += np.random.randint(100, 200)
                current_stock = min(current_stock, 400)
        
        new_df = pd.DataFrame(new_data)
        updated_df = pd.concat([df, new_df], ignore_index=True)
        updated_df.to_csv(DATA_FILE, index=False, encoding='utf-8-sig')
        
        return f"Товар '{product_name}' успешно добавлен за {days_to_add} дней!"
    except Exception as e:
        return f"Ошибка: {str(e)}"

def train_and_predict(data):
    """Обучение модели и прогноз (F4)"""
    X = data.drop(['Дата', 'Количество_товара', 'Продажи'], axis=1)
    y = data['Количество_товара']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = ['Товар']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    models = {
        'LinearRegression': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ]),
        'RandomForest': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=41,
                n_jobs=-1
            ))
        ]),
        'XGBoost': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', XGBRegressor(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                random_state=41,
                n_jobs=-1
            ))
        ])
    }
    
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = model.score(X_test, y_test)
        
        results.append({
            'Модель': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        })
    
    return models, pd.DataFrame(results), X_test, y_test