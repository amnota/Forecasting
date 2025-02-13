from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
from utils import read_file

app = FastAPI()

# โหลดโมเดล
try:
    model = joblib.load("model.pkl")
    print("✅ Model loaded successfully")
except Exception as e:
    model = None
    print(f"⚠️ Warning: Model not found - {e}")

@app.get("/")
def home():
    return {"message": "Welcome to Demand Forecasting API!"}

@app.post("/forecast/")
async def forecast(file: UploadFile = File(...)):
    df, error = await read_file(file)
    if error:
        return {"error": error}

    df.dropna(inplace=True)

    required_columns = ['past_sales', 'day_of_week', 'month', 'promotions', 'holidays', 'stock_level', 'customer_traffic']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        return {"error": f"CSV file ต้องมีคอลัมน์ {missing_columns}"}

    try:
        predictions = model.predict(df[required_columns])
        df['forecast_sales'] = predictions

        # คำนวณ Accuracy และ Risk Metrics
        if 'actual_sales' in df.columns:
            df['error'] = abs(df['forecast_sales'] - df['actual_sales'])
            forecast_accuracy = 100 - (df['error'].mean() / df['actual_sales'].mean() * 100)
            overstock_risk = (df[df['forecast_sales'] > df['actual_sales']].shape[0] / len(df)) * 100
            understock_risk = (df[df['forecast_sales'] < df['actual_sales']].shape[0] / len(df)) * 100
        else:
            forecast_accuracy = None
            overstock_risk = None
            understock_risk = None

        return {
            "predictions": df['forecast_sales'].tolist(),
            "forecast_accuracy": forecast_accuracy,
            "overstock_risk": overstock_risk,
            "understock_risk": understock_risk
        }

    except Exception as e:
        return {"error": f"Prediction failed: {e}"}

@app.get("/dashboard/")
def get_dashboard_data():
    return {
        "total_sale_revenue": 125000,
        "total_quantity_sold": 1500,
        "best_selling_product": "Product A",
        "least_selling_product": "Product Z",
        "stock_utilization_rate": 85
    }

@app.get("/demand_comparison/")
def get_demand_comparison():
    return {
        "products": [
            {"name": "Product A", "actual": 1200, "forecast": 1000, "difference": "+200", "risk": "Medium"},
            {"name": "Product B", "actual": 800, "forecast": 1200, "difference": "-400", "risk": "High"},
            {"name": "Product C", "actual": 1500, "forecast": 1450, "difference": "+50", "risk": "Low"},
            {"name": "Product D", "actual": 2000, "forecast": 1500, "difference": "+500", "risk": "High"},
            {"name": "Product E", "actual": 900, "forecast": 950, "difference": "-50", "risk": "Low"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

