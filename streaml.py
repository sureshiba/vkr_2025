import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor, Pool


@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model('catboost_model_3.cbm')
    return model


@st.cache_data
def load_data():
    data = pd.read_csv('train_mmsc.csv', sep=';')
    return data


# Добавляем константы для денормализации
SALES_VOLUME_MIN = 14.0
SALES_VOLUME_MAX = 133.0
STOCK_QUANTITY_MIN = 7.0
STOCK_QUANTITY_MAX = 141.0


def get_typical_values(data, product_name, category):
    # Фильтруем данные по выбранному продукту и категории
    product_data = data[(data['Product_Name'] == product_name) &
                        (data['Category'] == category)]

    # Если нет точных совпадений, фильтруем только по категории
    if len(product_data) == 0:
        product_data = data[data['Category'] == category]

    # Денормализуем значения для отображения
    def denormalize(value, min_val, max_val):
        return int(value * (max_val - min_val) + min_val)

    # Вычисляем типичные значения
    typical_values = {
        'Stock_Quantity': denormalize(product_data['Stock_Quantity'].median(), STOCK_QUANTITY_MIN, STOCK_QUANTITY_MAX),
        'Sales_Volume': denormalize(product_data['Sales_Volume'].median(), SALES_VOLUME_MIN, SALES_VOLUME_MAX),
        'Reorder_Level': product_data['Reorder_Level'].median(),
        'Reorder_Quantity': product_data['Reorder_Quantity'].median(),
        'Inventory_Turnover_Rate': product_data['Inventory_Turnover_Rate'].mean(),
        'Average_Price_Per_Category': product_data['Average_Price_Per_Category'].mean(),
        'Average_Price_Per_Product_Name': product_data['Average_Price_Per_Product_Name'].mean(),
        'Price_to_Sales_Ratio': product_data['Price_to_Sales_Ratio'].mean(),
        'Day_of_Week': product_data['Day_of_Week'].mode()[0],
        'Month': product_data['Month'].mode()[0]
    }
    return typical_values


def predict_price(model, features):
    cat_features = ['Product_Name', 'Status', 'Category']
    pool = Pool(features, cat_features=cat_features)
    return model.predict(pool)


def main():
    st.title('Прогноз цен для потребительского ритейла')
    model = load_model()
    data = load_data()

    # Создаем словари для преобразования
    weekdays = {
        0.000000: "Понедельник",
        0.166667: "Вторник",
        0.333333: "Среда",
        0.500000: "Четверг",
        0.666667: "Пятница",
        0.833333: "Суббота",
        1.000000: "Воскресенье"
    }

    months = {
        0.000000: "Январь",
        0.090909: "Февраль",
        0.181818: "Март",
        0.272727: "Апрель",
        0.363636: "Май",
        0.454545: "Июнь",
        0.545455: "Июль",
        0.636364: "Август",
        0.727273: "Сентябрь",
        0.818182: "Октябрь",
        0.909091: "Ноябрь",
        1.000000: "Декабрь"
    }

    st.header('Введите основные параметры товара')

    # Основные параметры для ввода
    category = st.selectbox('Категория', data['Category'].unique())
    product_name = st.selectbox('Название продукта',
                                data[data['Category'] == category]['Product_Name'].unique())
    status = st.selectbox('Статус', data['Status'].unique())

    # Получаем типичные значения для выбранного продукта
    typical_values = get_typical_values(data, product_name, category)

    # Дополнительные параметры (можно изменить)
    with st.expander("Дополнительные параметры (необязательно)"):
        stock_quantity = st.number_input('Количество на складе',
                                         value=typical_values['Stock_Quantity'],
                                         min_value=int(STOCK_QUANTITY_MIN),
                                         max_value=int(STOCK_QUANTITY_MAX),
                                         step=1)

        sales_volume = st.number_input('Объем продаж',
                                       value=typical_values['Sales_Volume'],
                                       min_value=int(SALES_VOLUME_MIN),
                                       max_value=int(SALES_VOLUME_MAX),
                                       step=1)

        # Выбор дня недели и месяца
        day_keys = sorted(weekdays.keys())
        default_day_idx = min(range(len(day_keys)), key=lambda i: abs(day_keys[i] - typical_values['Day_of_Week']))
        day_of_week = st.selectbox('День недели',
                                   options=day_keys,
                                   format_func=lambda x: weekdays[x],
                                   index=default_day_idx)

        month_keys = sorted(months.keys())
        default_month_idx = min(range(len(month_keys)), key=lambda i: abs(month_keys[i] - typical_values['Month']))
        month = st.selectbox('Месяц',
                             options=month_keys,
                             format_func=lambda x: months[x],
                             index=default_month_idx)

    # Нормализуем значения перед подачей в модель
    def normalize(value, min_val, max_val):
        return (value - min_val) / (max_val - min_val)

    # Формируем финальный DataFrame
    features = pd.DataFrame({
        'Product_Name': [product_name],
        'Stock_Quantity': [normalize(stock_quantity, STOCK_QUANTITY_MIN, STOCK_QUANTITY_MAX)],
        'Reorder_Level': [typical_values['Reorder_Level']],
        'Reorder_Quantity': [typical_values['Reorder_Quantity']],
        'Sales_Volume': [normalize(sales_volume, SALES_VOLUME_MIN, SALES_VOLUME_MAX)],
        'Inventory_Turnover_Rate': [typical_values['Inventory_Turnover_Rate']],
        'Status': [status],
        'Category': [category],
        'Average_Price_Per_Category': [typical_values['Average_Price_Per_Category']],
        'Average_Price_Per_Product_Name': [typical_values['Average_Price_Per_Product_Name']],
        'Price_to_Sales_Ratio': [typical_values['Price_to_Sales_Ratio']],
        'Day_of_Week': [day_of_week],
        'Month': [month]
    })

    if st.button('Прогнозировать цену'):
        prediction = predict_price(model, features)
        st.success(f'Прогнозируемая цена: {prediction[0]:.2f}')

        # Показываем использованные параметры
        st.info("Использованные параметры:")
        st.json({
            "Основные параметры": {
                "Продукт": product_name,
                "Статус": status,
                "Категория": category,
                "Количество на складе": stock_quantity,
                "Объем продаж": sales_volume,
                "День недели": weekdays[day_of_week],
                "Месяц": months[month]
            }
        })


if __name__ == '__main__':
    main()