from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import joblib
import json
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn

app = FastAPI()

# ✅ ใช้ Cache สำหรับเก็บข้อมูลอัปโหลดล่าสุด
uploaded_data_cache = None  

# ✅ ตั้งค่า CORS ให้รองรับทุกโดเมน (แก้ไขถ้าจำเป็น)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ โหลดโมเดล Machine Learning (ถ้ามี)
try:
    model = joblib.load("model.pkl")
    print("✅ Model loaded successfully")
except Exception as e:
    model = None
    print(f"⚠️ Warning: Model not found - {e}")

# ✅ หน้าแรกของ API
@app.get("/")
def home():
    return {"message": "Welcome to Demand Forecasting API!"}

# ✅ อัปโหลดไฟล์ CSV และทำ Forecast
@app.post("/forecast/")
async def forecast(file: UploadFile = File(...)):
    global uploaded_data_cache
    try:
        df = pd.read_csv(file.file)

        # ✅ ตรวจสอบคอลัมน์ที่จำเป็น
        required_columns = ['past_sales', 'day_of_week', 'month', 'promotions', 'holidays', 'stock_level', 'customer_traffic']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing_columns}")

        # ✅ ใช้โมเดลคาดการณ์ยอดขาย
        predictions = model.predict(df[required_columns]) if model else [0] * len(df)
        df["forecast_sales"] = predictions

        # ✅ เก็บข้อมูลล่าสุดไว้ใน Cache
        uploaded_data_cache = df.to_dict(orient="list")

        return {
            "predictions": df["forecast_sales"].tolist(),
            "forecast_accuracy": 92,  # ปรับเป็นค่าจริงถ้าต้องการ
            "overstock_risk": 15,
            "understock_risk": 8
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# ✅ ดึงข้อมูล Dashboard
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
        "stock_utilization_rate": 85  # ปรับค่าจริงได้
    }

# ✅ เพิ่ม API สำหรับแสดงแนวโน้มยอดขายในรูปกราฟ
@app.get("/sales_trends/")
def get_sales_trends():
    if not uploaded_data_cache:
        return []

    sales_trends = [
        {"date": f"2024-02-{str(i+1).zfill(2)}", "sales": actual, "forecast": forecast}
        for i, (actual, forecast) in enumerate(zip(uploaded_data_cache["past_sales"], uploaded_data_cache["forecast_sales"]))
    ]
    return sales_trends

# ✅ ดึงข้อมูล Demand Comparison
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

# ✅ รันเซิร์ฟเวอร์
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
