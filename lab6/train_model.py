import pandas as pd
import polars as pl
import numpy as np
from catboost import CatBoostClassifier

def train():
    # Загрузка
    schema_overrides = {
        "Administrative": pl.Int64, "Administrative_Duration": pl.Float64,
        "Informational": pl.Int64, "Informational_Duration": pl.Float64,
        "ProductRelated": pl.Int64, "ProductRelated_Duration": pl.Float64,
        "BounceRates": pl.Float64, "ExitRates": pl.Float64,
        "PageValues": pl.Int64, "SpecialDay": pl.Int64,
        "Month": pl.Categorical, "OperatingSystems": pl.Int64,
        "Browser": pl.Int64, "Region": pl.Int64, 
        "TrafficType": pl.Int64, "VisitorType": pl.Categorical,
        "Weekend": pl.Boolean, "Revenue": pl.Boolean
    }

    df = pl.read_csv('online_shoppers_intention.csv', schema_overrides=schema_overrides, ignore_errors=True)

    # Feature Engineering
    month_map = {"Feb": 2, "Mar": 3, "May": 5, "June": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
    visitor_map = {"New_Visitor": 0, "Returning_Visitor": 1, "Other": 2}

    # Маппинг Month
    df = df.with_columns(
        pl.col("Month").cast(pl.String).replace(month_map, default=None).cast(pl.Int32).fill_null(0).alias("Month_Num")
    )
    df = df.with_columns([
        (np.sin(2 * np.pi * pl.col("Month_Num") / 12)).alias("Month_sin"),
        (np.cos(2 * np.pi * pl.col("Month_Num") / 12)).alias("Month_cos")
    ]).drop(["Month", "Month_Num"])

    # Маппинг VisitorType 
    df = df.with_columns(
        pl.col("VisitorType").cast(pl.String).replace(visitor_map, default=2).cast(pl.Int64)
    )
    df = df.with_columns(
        pl.when(pl.col("ProductRelated") == 0).then(0.0)
        .otherwise(pl.col("ProductRelated_Duration") / pl.col("ProductRelated"))
        .fill_nan(0).fill_null(0).alias("Product_Avg_Duration")
    ).drop(["BounceRates", "ProductRelated", "ProductRelated_Duration"])
    
    df = df.with_columns([
        pl.col("Weekend").cast(pl.Float64),  # Weekend -> FLOAT
        pl.col("Revenue").cast(pl.Int8)
    ])

    df_pd = df.to_pandas()
    X = df_pd.drop(columns=["Revenue"])
    y = df_pd["Revenue"]

    cat_features = [] 
    
    print(f"Training with columns: {X.columns.tolist()}")
    print(f"Weekend type in X: {X['Weekend'].dtype}")
    
    model = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, verbose=100)
    model.fit(X, y, cat_features=cat_features) 
    model.save_model("shoppers_model.cbm")
    print("Model saved.")

if __name__ == "__main__":
    train()
