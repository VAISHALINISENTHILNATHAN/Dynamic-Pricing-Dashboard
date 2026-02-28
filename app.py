import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split


st.set_page_config(page_title="Dynamic Pricing System",layout="wide")
st.title("💰 Dynamic Pricing & Demand Forecasting Dashboard")

@st.cache_data
def load_data():
    df=pd.read_csv("retail_price_cleaned.csv")
    return df

df=load_data()

df_model=pd.get_dummies(df,columns=['product_category_name'],drop_first=True)
target='qty'
drop_columns=['qty','month_year','product_id']

X=df_model.drop(columns=drop_columns)
y=df_model[target]

@st.cache_resource
def train_model(X,y):
    model=xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        objective='reg:squarederror',
        random_state=42
    )
    model.fit(X,y)
    return model

model=train_model(X,y)


st.sidebar.header("Select Product")
product_ids=df['product_id'].unique()
selected_product=st.sidebar.selectbox("Choose Product ID",product_ids)

product_data=df[df['product_id']==selected_product].iloc[-1:]
product_model_data=df_model[df_model['product_id']==selected_product].iloc[-1:]

st.subheader("📦 Product Information")
col1,col2,col3=st.columns(3)
col1.metric("Current Price",f"{product_data['unit_price'].values[0]:.2f}")
col2.metric("Last Units Sold",f"{product_data['qty'].values[0]:.0f}")
col3.metric("Avg Competitor Price",f"{product_data['avg_comp_price'].values[0]:.2f}")

X_product=product_model_data.drop(columns=drop_columns)
predicted_units=model.predict(X_product)[0]

st.subheader("📈 Predicted Demand")
st.metric("Forecasted Units Sold", f"{predicted_units:.0f}")

st.subheader("💰 Dynamic Pricing Recommendation")
original_price=product_data['unit_price'].values[0]
price_options=[
    original_price*0.9,
    original_price,
    original_price*1.1
]
revenues=[]

for price in price_options:
    temp_data=X_product.copy()
    temp_data['unit_price']=price
    predicted=model.predict(temp_data)[0]
    revenue=predicted*price
    revenues.append(revenue)

optimal_index=np.argmax(revenues)
optimal_price=price_options[optimal_index]

st.write("### Price Simulation Results")
for i,price in enumerate(price_options):
    st.write(f"Price: {price:.2f} -> Expected Revenue: {revenues[i]:.2f}")
st.success(f"✅ Optimal Price Recommendation: {optimal_price:.2f}")


st.subheader("📊 Historical Sales Trend")
product_history=df[df['product_id']==selected_product]
product_history=product_history.sort_values("month_year")
st.line_chart(product_history.set_index("month_year")["qty"])
