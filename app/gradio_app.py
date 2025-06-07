import gradio as gr
import pandas as pd
from data_processing.modeling import add_new_product
from data_processing.data_loading import load_and_prepare_data

def view_existing_products():
    try:
        df = load_and_prepare_data()
        unique_products = df['Товар'].unique().tolist()
        return "Существующие товары:\n- " + "\n- ".join(unique_products)
    except:
        return "Файл данных не найден или пуст."

def create_gradio_interface():
    with gr.Blocks(title="Управление товарами") as demo:
        gr.Markdown("## Добавление нового товара в систему")

        with gr.Row():
            with gr.Column():
                product_name = gr.Textbox(label="Название товара", placeholder="Например: Молоко Простоквашино 2.5%")
                initial_stock = gr.Number(label="Начальный остаток", value=200)
                price = gr.Number(label="Цена", value=100)
                sales_per_day = gr.Slider(label="Ожидаемый уровень продаж (% от остатка)", minimum=10, maximum=90, value=70)
                days_to_add = gr.Number(label="Количество дней для добавления", value=30)
                add_btn = gr.Button("Добавить товар")

            with gr.Column():
                output = gr.Textbox(label="Результат")
                view_btn = gr.Button("Показать существующие товары")
                existing_products = gr.Textbox(label="Список товаров")

        add_btn.click(
            fn=add_new_product,
            inputs=[product_name, initial_stock, price, sales_per_day, days_to_add],
            outputs=output
        )

        view_btn.click(
            fn=view_existing_products,
            outputs=existing_products
        )

        gr.Markdown("### Инструкция:")
        gr.Markdown("""
        1. Введите данные о новом товаре
        2. Укажите начальный остаток и цену
        3. Задайте ожидаемый уровень продаж (в % от остатка)
        4. Укажите количество дней для добавления
        5. Нажмите кнопку 'Добавить товар'
        """)
    
    return demo