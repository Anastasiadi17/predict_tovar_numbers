from data_processing.data_loading import load_and_prepare_data
from data_processing.preprocessing import feature_engineering, check_stationarity
from data_processing.modeling import train_and_predict
from data_processing.visualization import visualize_results
from data_processing.reporting import generate_reports
from app.gradio_app import create_gradio_interface
import numpy as np

def main():
    # F1: Загрузка данных
    data = load_and_prepare_data()
    
    # F2: Проверка стационарности
    random_products = np.random.choice(data['Товар'].unique(), 3, replace=False)
    for product in random_products:
        check_stationarity(data, product)
    check_stationarity(data)
    
    # Предобработка
    data = feature_engineering(data)
    
    # F4: Обучение моделей
    print("\nОбучение моделей:")
    models, metrics, X_test, y_test = train_and_predict(data)
    
    # F5: Визуализация
    predictions = visualize_results(models, metrics, X_test, y_test, data)
    print("\nПрогнозы сохранены в 'Прогнозы.csv'")
    
    # F6: Отчеты
    report = generate_reports()
    print("\nОтчет:", report)
    
    # Запуск Gradio интерфейса (F3)
    demo = create_gradio_interface()
    demo.launch()

if __name__ == "__main__":
    main()