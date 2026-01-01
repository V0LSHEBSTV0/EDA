import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from explainerdashboard import ClassifierExplainer, ExplainerDashboard

def preprocess_for_dashboard(df_raw):
    
    month_map = {"Feb": 2, "Mar": 3, "May": 5, "June": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
    df_raw['Month_Num'] = df_raw['Month'].map(month_map).fillna(0).astype(int)
    df_raw['Month_sin'] = np.sin(2 * np.pi * df_raw['Month_Num'] / 12)
    df_raw['Month_cos'] = np.cos(2 * np.pi * df_raw['Month_Num'] / 12)
    
    df_raw['Product_Avg_Duration'] = df_raw.apply(
        lambda row: row['ProductRelated_Duration'] / row['ProductRelated'] if row['ProductRelated'] > 0 else 0.0, axis=1
    )
    
    # VisitorType Mapping 
    visitor_map = {"New_Visitor": 0, "Returning_Visitor": 1, "Other": 2}

    if df_raw['VisitorType'].dtype == 'object':
        df_raw['VisitorType'] = df_raw['VisitorType'].map(visitor_map).fillna(2).astype(int)
    
    # Удаление лишнего
    cols_to_drop = ['Month', 'Month_Num', 'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'Revenue']
    df = df_raw.drop(columns=[c for c in cols_to_drop if c in df_raw.columns])
    
    # Принудительные типы 
    df['Weekend'] = df['Weekend'].astype(float)
    df['VisitorType'] = df['VisitorType'].astype(int)
    
    return df

print("Loading and processing data...")
# Загружаем сырые данные
df_raw = pd.read_csv("online_shoppers_intention.csv")

X_test = preprocess_for_dashboard(df_raw)
y_test = df_raw['Revenue'].astype(int)

X_test_sample = X_test.iloc[:1000]
y_test_sample = y_test.iloc[:1000]

print("Data loaded. Using sample of 1000 rows for dashboard.")

# Загружаем модель
model = CatBoostClassifier()
model.load_model('shoppers_model.cbm')
print("Model loaded.")

# Генерируем Explainer
explainer = ClassifierExplainer(
    model, X_test_sample, y_test_sample, 
    cats=[], 
    labels=['Not Buy', 'Buy']
)

print("Generating dashboard configuration (this might take a minute)...")
db = ExplainerDashboard(
    explainer, 
    title="Shoppers Intention Dashboard", 
    whatif=False, 
    shap_interaction=False
)

# Сохраняем конфигурацию в файл 
db.to_yaml("dashboard.yaml", explainerfile="explainer.joblib", dump_explainer=True)
print("Dashboard generated and saved to dashboard.yaml")
