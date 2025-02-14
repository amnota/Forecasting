from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import joblib
import json
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn

app = FastAPI()

# ✅ Global variable for temporary storage (Memory)
uploaded_data_cache = None  

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

# ✅ Homepage Route
@app.get("/")
def home():
    return {"message": "Welcome to Demand Forecasting API!"}

# ✅ Forecast API
@app.post("/forecast/")
async def forecast(file: UploadFile = File(...)):
    global uploaded_data_cache
    try:
        df = pd.read_csv(file.file)

        # ✅ Validate required columns
        required_columns = ['past_sales', 'day_of_week', 'month', 'promotions', 'holidays', 'stock_level', 'customer_traffic']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing_columns}")

        # ✅ Predict sales using ML Model
        predictions = model.predict(df[required_columns]) if model else [0] * len(df)
        df["forecast_sales"] = predictions

        # ✅ Save to memory (cache)
        uploaded_data_cache = df.to_dict(orient="list")

        return {
            "predictions": df["forecast_sales"].tolist(),
            "forecast_accuracy": 92,  # Replace with real calculation
            "overstock_risk": 15,
            "understock_risk": 8
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# ✅ Dashboard API (Fetch from memory)
@app.get("/dashboard/")
def get_dashboard_data():
    if not uploaded_data_cache:
        raise HTTPException(status_code=404, detail="No uploaded data available")
    
    return {
        "total_sale_revenue": sum(uploaded_data_cache["past_sales"]),
        "total_quantity_sold": len(uploaded_data_cache["past_sales"]),
        "best_selling_product": "Product A",
        "least_selling_product": "Product Z",
        "stock_utilization_rate": 85  # Replace with actual calculation
    }

# ✅ Demand Comparison API (Fetch from memory)
@app.get("/demand_comparison/")
def get_demand_comparison():
    if not uploaded_data_cache:
        raise HTTPException(status_code=404, detail="No uploaded data available")
    
    products = [
        {"name": f"Product {i+1}", "actual": actual, "forecast": forecast, "difference": actual - forecast, "risk": "Low" if abs(actual - forecast) < 100 else "High"}
        for i, (actual, forecast) in enumerate(zip(uploaded_data_cache["past_sales"], uploaded_data_cache["forecast_sales"]))
    ]
    return {"products": products}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
