# Dynamic Pricing & Demand Forecasting Dashboard

## 🔹 Project Overview
This project is an **end-to-end dynamic pricing and demand forecasting system** built with **Python**, **XGBoost**, and **Streamlit**. It predicts product demand, simulates multiple price points, and recommends the **optimal price** to maximize revenue.  

The system demonstrates **advanced ML techniques, feature engineering, and interactive dashboard development**, making it a strong portfolio project.

---

## 🔹 Key Features
- **Demand Forecasting:** Predict units sold based on price, competitor price, lag demand, seasonality, and category features.  
- **Dynamic Pricing Engine:** Simulates different price points and selects the price that maximizes revenue.  
- **Feature Importance Visualization:** Identifies which factors most impact product demand.  
- **Interactive Dashboard:**  
  - Select products from the sidebar  
  - View predicted demand  
  - See optimal price recommendations  
  - Visualize historical sales trends  

---

## 🔹 Dataset
- Synthetic dataset (~50k rows) or Kaggle retail dataset.  
- Cleaned CSV included: `retail_price_cleaned.csv`  
- Columns include:  
  - `product_id` – Unique product identifier  
  - `category` – Product category (Electronics, Clothing, etc.)  
  - `date` – Sale date  
  - `units_sold` – Quantity sold  
  - `price` – Product price  
  - `cost` – Cost of product  
  - `competitor_price` – Average competitor price  
  - `promotion` – Boolean indicating promotion  

---

## 🔹 Project Structure
```text
dynamic-pricing-dashboard/
│
├── app.py                 # Streamlit interactive dashboard
├── demand_model.py        # XGBoost demand model + dynamic pricing engine
├── retail_price_cleaned.csv  # Dataset
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

🔹 Installation

Clone the repository:

git clone https://github.com/VAISHALINISENTHILNATHAN/Dynamic-Pricing-Dashboard.git
cd Dynamic-Pricing-Dashboard

Install dependencies:

pip install -r requirements.txt

🔹 How to Run

Launch the dashboard locally:

streamlit run app.py

Steps:

Select a product from the sidebar

View predicted demand

See optimal price suggestion

Explore historical sales trends

🔹 Results & Insights

RMSE: Shows prediction accuracy for units sold
Feature Importance: Price, previous demand, and competitor price are often the most influential
Revenue Optimization: Simulated price scenarios identify the most profitable pricing strategy

