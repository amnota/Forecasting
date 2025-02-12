import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# สร้างตัวอย่างข้อมูล
data = {
    'past_sales': [100, 200, 300, 400, 500],
    'day_of_week': [1, 2, 3, 4, 5],
    'month': [1, 1, 2, 2, 3],
    'promotions': [1, 0, 1, 0, 1],
    'holidays': [0, 0, 1, 0, 1],
    'stock_level': [50, 60, 70, 80, 90],
    'customer_traffic': [200, 250, 300, 350, 400],
    'sales': [120, 210, 340, 450, 560]  # Target (ยอดขายจริง)
}

df = pd.DataFrame(data)

# Train โมเดล
X = df.drop(columns=['sales'])
y = df['sales']

model = LinearRegression()
model.fit(X, y)

# บันทึกโมเดล
joblib.dump(model, "model.pkl")

print("✅ Model training completed and saved as 'model.pkl'")
