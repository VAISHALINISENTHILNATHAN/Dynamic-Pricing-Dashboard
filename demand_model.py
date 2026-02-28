import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# -----------------------------
# 1️⃣ Load Cleaned Dataset
# -----------------------------
df = pd.read_csv('retail_price_cleaned.csv')

# -----------------------------
# 2️⃣ Encode Categorical Variables
# -----------------------------
df = pd.get_dummies(df, columns=['product_category_name'], drop_first=True)

# -----------------------------
# 3️⃣ Define Features & Target
# -----------------------------
target = 'qty'
drop_columns = ['qty', 'month_year', 'product_id']

X = df.drop(columns=drop_columns)
y = df[target]

# -----------------------------
# 4️⃣ Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)

# -----------------------------
# 5️⃣ Train XGBoost Model
# -----------------------------
model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    objective='reg:squarederror',
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# 6️⃣ Model Evaluation
# -----------------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"\n✅ Test RMSE: {rmse:.2f}")

# -----------------------------
# 7️⃣ Feature Importance
# -----------------------------
plt.figure(figsize=(10,6))
xgb.plot_importance(model, max_num_features=10)
plt.title("Top 10 Features Influencing Demand")
plt.show()

# ==========================================================
# 💰 8️⃣ Dynamic Pricing Recommendation Engine
# ==========================================================

print("\n🔹 Running Dynamic Pricing Simulation...")

# Select one sample product row from test set
sample_product = X_test.iloc[0:1].copy()

original_price = sample_product['unit_price'].values[0]

# Define price scenarios (±10%)
price_options = [
    original_price * 0.9,
    original_price,
    original_price * 1.1
]

revenues = []

for price in price_options:
    temp_data = sample_product.copy()
    temp_data['unit_price'] = price

    predicted_units = model.predict(temp_data)[0]
    revenue = predicted_units * price
    revenues.append(revenue)

# Get optimal price
optimal_index = np.argmax(revenues)
optimal_price = price_options[optimal_index]

print("\n📊 Price Simulation Results:")
for i, price in enumerate(price_options):
    print(f"Price: {price:.2f} → Expected Revenue: {revenues[i]:.2f}")

print(f"\n💰 Optimal Price Recommendation: {optimal_price:.2f}")