import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor

# Загружаем модель и пайплайн
@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model("catboost_model.cbm")
    return model

st.title("🏡 Предсказание стоимости недвижимости")

# Форма для загрузки CSV
uploaded_file = st.file_uploader("Загрузите CSV-файл с характеристиками недвижимости", type="csv")

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)
    st.subheader("📄 Загруженные данные")
    st.write(input_df.head())

    model = load_model()

    try:
        # Обрабатываем данные
        moda_col = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'Electrical', 'KitchenQual', 'Functional', 'GarageCars', 'GarageArea', 'SaleType']
        none_col = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                    'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType',
                    'GarageFinish','GarageQual', 'GarageCond', 'PoolQC',
                    'Fence', 'MiscFeature', 'MasVnrType']
        zero_col = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt', 'MasVnrArea']
        
        input_df[none_col] = input_df[none_col].fillna('No')
        input_df[moda_col] = input_df[moda_col].apply(lambda x: x.fillna(x.mode()[0]))
        input_df['LotFrontage'] = input_df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median()))
        input_df[zero_col] = input_df[zero_col].fillna(0)

        input_df['Total_Area'] = input_df['GrLivArea'] + input_df['TotalBsmtSF'] + input_df['1stFlrSF'] + input_df['2ndFlrSF'] + input_df['WoodDeckSF'] + input_df['OpenPorchSF'] + input_df['EnclosedPorch'] + input_df['3SsnPorch'] + input_df['ScreenPorch'] + input_df['PoolArea']

        current_year = 2025  
        input_df['Age_of_House'] = current_year - input_df['YearBuilt']

        input_df['Living_Area_Ratio'] = input_df['GrLivArea'] / input_df['Total_Area']
        input_df['Total_Bathrooms'] = input_df['FullBath'] + (input_df['HalfBath'] * 0.5) + input_df['BsmtFullBath'] + (input_df['BsmtHalfBath'] * 0.5)
        input_df['Renovation_Age'] = input_df['Age_of_House'] - (current_year - input_df['YearRemodAdd'])
        input_df['Garage_Area_to_House_Area_Ratio'] = input_df['GarageArea'] / input_df['Total_Area']
        input_df['Overall_Quality_Condition'] = input_df['OverallQual'] + input_df['OverallCond']

        X_test = input_df.drop('Id', axis=1)

        # Предсказание логарифма стоимости
        preds_log = model.predict(X_test)
        
        # Преобразуем обратно из логарифма
        preds = np.exp(preds_log)
        
        # Вывод результата
        result = input_df.copy()
        result["Predicted Price"] = preds.astype(int)

        st.subheader("💰 Предсказанные цены:")
        st.write(result[["Predicted Price"]].head())

        # Кнопка для скачивания
        csv = result.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Скачать результат",
            data=csv,
            file_name='predicted_prices.csv',
            mime='text/csv',
        )
    except Exception as e:
        st.error(f"Произошла ошибка: {e}")
