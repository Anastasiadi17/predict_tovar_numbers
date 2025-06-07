import pandas as pd
import matplotlib.pyplot as plt
from ..config import PREDICTIONS_FILE, MODEL_METRICS_FILE

def visualize_results(models, results, X_test, y_test, data):
    """Визуализация результатов (F5)"""
    results.to_csv(MODEL_METRICS_FILE, index=False, encoding='utf-8-sig')
    
    best_model_name = results.loc[results['MAE'].idxmin(), 'Модель']
    best_model = models[best_model_name]
    predictions = best_model.predict(X_test)
    
    result_df = pd.DataFrame({
        'Дата': data.loc[X_test.index, 'Дата'],
        'Товар': X_test['Товар'],
        'Фактический_остаток': y_test,
        'Прогноз': predictions,
        'Ошибка': np.abs(y_test - predictions)
    })
    
    result_df.to_csv(PREDICTIONS_FILE, index=False, encoding='utf-8-sig')
    
    top_products = result_df.groupby('Товар')['Фактический_остаток'].count().nlargest(3).index
    
    for product in top_products:
        product_data = result_df[result_df['Товар'] == product].set_index('Дата')
        original_data = data[data['Товар'] == product].set_index('Дата')
        
        plt.figure(figsize=(16, 8))
        plt.plot(original_data['Количество_товара'], label='Фактические остатки', alpha=0.51)
        plt.scatter(product_data.index, product_data['Фактический_остаток'],
                   color='red', label='Тестовые данные')
        plt.plot(product_data.index, product_data['Прогноз'],
                '--', label='Прогноз')
        plt.fill_between(product_data.index,
                        product_data['Прогноз'] - product_data['Ошибка'],
                        product_data['Прогноз'] + product_data['Ошибка'],
                        alpha=0.2, color='blue', label='Погрешность')
        
        plt.title(f'Прогнозирование остатков: {product}\n({best_model_name})')
        plt.xlabel('Дата')
        plt.ylabel('Количество товара')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return result_df