import pandas as pd
from ..config import PREDICTIONS_FILE, MODEL_METRICS_FILE

def generate_reports():
    """Генерация отчетов (F6)"""
    try:
        metrics = pd.read_csv(MODEL_METRICS_FILE)
        predictions = pd.read_csv(PREDICTIONS_FILE)
        
        report = {
            "best_model": metrics.loc[metrics['MAE'].idxmin(), 'Модель'],
            "average_error": predictions['Ошибка'].mean(),
            "top_products": predictions['Товар'].value_counts().nlargest(3).to_dict()
        }
        
        return report
    except Exception as e:
        return {"error": str(e)}