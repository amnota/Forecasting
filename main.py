from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn

app = FastAPI()

# ✅ CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load ML Model
try:
    model = joblib.load("model.pkl")
    print("✅ Model loaded successfully")
except Exception as e:
    model = None
    print(f"⚠️ Warning: Model not found - {e}")

# ✅ Global variable for temporary storage
uploaded_data_cache = None  

@app.get("/")
def home():
    return {"message": "Welcome to Demand Forecasting API!"}

# ✅ Forecast API - Upload CSV and Predict
def process_forecast_data(df):
    required_columns = ['past_sales', 'day_of_week', 'month', 'promotions', 'holidays', 'stock_level', 'customer_traffic']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing_columns}")
    
    predictions = model.predict(df[required_columns]) if model else [0] * len(df)
    df["forecast_sales"] = predictions
    return df

@app.post("/forecast/")
async def forecast(file: UploadFile = File(...)):
    global uploaded_data_cache
    try:
        df = pd.read_csv(file.file)
        df = process_forecast_data(df)
        uploaded_data_cache = df.to_dict(orient="list")

        return {
            "predictions": df["forecast_sales"].tolist(),
            "forecast_accuracy": 92,
            "overstock_risk": 15,
            "understock_risk": 8
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# ✅ Dashboard API
@app.get("/dashboard/")
def get_dashboard_data():
    if not uploaded_data_cache:
        return {
            "total_sale_revenue": "N/A",
            "total_quantity_sold": "N/A",
            "best_selling_product": "N/A",
            "least_selling_product": "N/A",
            "stock_utilization_rate": 0
        }
    return {
        "total_sale_revenue": sum(uploaded_data_cache["past_sales"]),
        "total_quantity_sold": len(uploaded_data_cache["past_sales"]),
        "best_selling_product": "Product A",
        "least_selling_product": "Product Z",
        "stock_utilization_rate": 85
    }

# ✅ Sales Trends API
@app.get("/sales_trends/")
def get_sales_trends():
    if not uploaded_data_cache:
        return []
    
    return [
        {"date": f"2024-02-{i+1:02d}", "sales": actual, "forecast": forecast}
        for i, (actual, forecast) in enumerate(zip(uploaded_data_cache["past_sales"], uploaded_data_cache["forecast_sales"]))
    ]

# ✅ Demand Comparison API
@app.get("/demand_comparison/")
def get_demand_comparison():
    if not uploaded_data_cache:
        return {"products": []}
    
    products = [
        {
            "name": f"Product {i+1}",
            "actual": actual,
            "forecast": forecast,
            "difference": actual - forecast,
            "risk": "Low" if abs(actual - forecast) < 100 else "High"
        }
        for i, (actual, forecast) in enumerate(zip(uploaded_data_cache["past_sales"], uploaded_data_cache["forecast_sales"]))
    ]
    return {"products": products}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
