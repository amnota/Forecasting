from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import joblib
from utils import read_file
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn

app = FastAPI()
UPLOAD_FOLDER = "uploaded_data"  # 📂 โฟลเดอร์เก็บไฟล์ CSV ที่อัปโหลด

# ✅ CORS Configuration - รองรับทั้ง Local และ Production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ✅ Fix CORS
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ โหลดโมเดล Machine Learning
try:
    model = joblib.load("model.pkl")
    print("✅ Model loaded successfully")
except Exception as e:
    model = None
    print(f"⚠️ Warning: Model not found - {e}")

# ✅ Route สำหรับ Homepage
@app.get("/")
def home():
    print("🔹 GET / - API is running")
    return {"message": "Welcome to Demand Forecasting API!"}

# ✅ API สำหรับทำ Forecast
@app.post("/forecast/")
async def forecast(file: UploadFile = File(...)):
    print("🔹 POST /forecast/ - File received:", file.filename)

    df, error = await read_file(file)
    if error:
        print("⚠️ Error reading file:", error)
        raise HTTPException(status_code=400, detail=error)

    df.dropna(inplace=True)

    required_columns = ['past_sales', 'day_of_week', 'month', 'promotions', 'holidays', 'stock_level', 'customer_traffic']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"⚠️ Missing columns: {missing_columns}")
        raise HTTPException(status_code=400, detail=f"CSV file ต้องมีคอลัมน์ {missing_columns}")

    if model is None:
        print("⚠️ Model is not loaded")
        raise HTTPException(status_code=500, detail="Model is not loaded. Please check deployment.")

    try:
        predictions = model.predict(df[required_columns])
        df['forecast_sales'] = predictions

        if 'actual_sales' in df.columns:
            df['error'] = abs(df['forecast_sales'] - df['actual_sales'])
            forecast_accuracy = 100 - (df['error'].mean() / df['actual_sales'].mean() * 100)
            overstock_risk = (df[df['forecast_sales'] > df['actual_sales']].shape[0] / len(df)) * 100
            understock_risk = (df[df['forecast_sales'] < df['actual_sales']].shape[0] / len(df)) * 100
        else:
            forecast_accuracy = None
            overstock_risk = None
            understock_risk = None

        # ✅ บันทึกไฟล์ CSV ไว้ที่ UPLOAD_FOLDER
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        df.to_csv(file_path, index=False)

        response = {
            "predictions": df['forecast_sales'].tolist(),
            "forecast_accuracy": forecast_accuracy,
            "overstock_risk": overstock_risk,
            "understock_risk": understock_risk
        }

        print("✅ Prediction Success:", response)
        return response

    except Exception as e:
        print(f"⚠️ Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# ✅ API สำหรับ Dashboard Summary
@app.get("/dashboard/")
def get_dashboard_data():
    print("🔹 GET /dashboard/ - Fetching data from latest uploaded file")

    try:
        files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.csv')]
        if not files:
            return {"error": "No uploaded files found"}

        latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(UPLOAD_FOLDER, x)))
        file_path = os.path.join(UPLOAD_FOLDER, latest_file)
        df = pd.read_csv(file_path)

        total_sale_revenue = df["forecast_sales"].sum()
        total_quantity_sold = len(df)
        best_selling_product = "Product A"  # 🛠️ ปรับให้เหมาะสมหากมีข้อมูลสินค้า
        least_selling_product = "Product Z"
        stock_utilization_rate = 85  # คำนวณให้เหมาะสมหากมีข้อมูลสินค้าคงคลัง

        return {
            "total_sale_revenue": total_sale_revenue,
            "total_quantity_sold": total_quantity_sold,
            "best_selling_product": best_selling_product,
            "least_selling_product": least_selling_product,
            "stock_utilization_rate": stock_utilization_rate
        }

    except Exception as e:
        return {"error": f"Failed to fetch dashboard data: {str(e)}"}

# ✅ API สำหรับเปรียบเทียบ Demand
@app.get("/demand_comparison/")
def get_demand_comparison():
    print("🔹 GET /demand_comparison/ - Fetching data from latest uploaded file")

    try:
        files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.csv')]
        if not files:
            return {"error": "No uploaded files found"}

        latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(UPLOAD_FOLDER, x)))
        file_path = os.path.join(UPLOAD_FOLDER, latest_file)
        df = pd.read_csv(file_path)

        df["forecast"] = df["past_sales"] * 1.1  # 🛠️ คำนวณ Forecast (เปลี่ยนได้)
        df["difference"] = df["forecast"] - df["past_sales"]

        def calculate_risk(row):
            if row["difference"] > 500:
                return "High"
            elif row["difference"] > 100:
                return "Medium"
            else:
                return "Low"

        df["risk"] = df.apply(calculate_risk, axis=1)

        products = df.head(5).to_dict(orient="records")

        return {"products": products}

    except Exception as e:
        return {"error": f"Failed to fetch demand comparison data: {str(e)}"}

# ✅ Run Uvicorn สำหรับ Deploy บน Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # ดึงค่าจาก Environment Variable
    uvicorn.run(app, host="0.0.0.0", port=port)
