from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

app = Flask(__name__)

# Загружаем модель
model = CatBoostClassifier()
try:
    model.load_model('shoppers_model.cbm')
    print("Model loaded successfully.")
except Exception as e:
    print(f"FAILED to load model: {e}")

# Словари для маппинга 
MONTH_MAP = {"Feb": 2, "Mar": 3, "May": 5, "June": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
VISITOR_MAP = {"New_Visitor": 0, "Returning_Visitor": 1, "Other": 2}

def preprocess_input(data_dict):
    df = pd.DataFrame([data_dict])
    
    # Month Processing
    if 'Month' in df.columns:
        df['Month_Num'] = df['Month'].map(MONTH_MAP).fillna(0).astype(int)
        df['Month_sin'] = np.sin(2 * np.pi * df['Month_Num'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month_Num'] / 12)
        df = df.drop(columns=['Month', 'Month_Num'])
   
    # Duration Processing
    if 'ProductRelated' in df.columns and 'ProductRelated_Duration' in df.columns:
        df['Product_Avg_Duration'] = df.apply(
            lambda row: row['ProductRelated_Duration'] / row['ProductRelated'] if row['ProductRelated'] > 0 else 0.0, 
            axis=1
        )
        df = df.drop(columns=['ProductRelated', 'ProductRelated_Duration'])
        
    if 'BounceRates' in df.columns:
        df = df.drop(columns=['BounceRates'])
        
    if 'VisitorType' in df.columns:
        if df['VisitorType'].dtype == 'object':
             df['VisitorType'] = df['VisitorType'].map(VISITOR_MAP).fillna(2).astype(int)
    
    # Fill missing columns
    expected_cols = [
        "Administrative", "Administrative_Duration", 
        "Informational", "Informational_Duration", 
        "PageValues", "SpecialDay", 
        "OperatingSystems", "Browser", "Region", "TrafficType", 
        "VisitorType", "Weekend", 
        "Month_sin", "Month_cos", 
        "Product_Avg_Duration", "ExitRates" 
    ]
    
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0.0
            
    df = df[expected_cols]
    
    for col in df.columns:
        if col == "VisitorType":
            df[col] = df[col].astype(int)
        else:
            df[col] = df[col].astype(float)

    return df
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        content = request.json
        processed_df = preprocess_input(content)
        
        prediction_prob = model.predict_proba(processed_df)[0, 1]
        prediction_class = int(model.predict(processed_df)[0])
        
        return jsonify({
            'purchase_probability': float(prediction_prob),
            'will_buy': bool(prediction_class)
        })
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
