import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('retail_price.csv')

print("First 5 rows:")
print(df.head())

print("\nMissing values per column:")
print(df.isnull().sum())

numeric_cols=df.select_dtypes(include=['float64','int64']).columns
for col in numeric_cols:
    df[col].fillna(df[col].median(),inplace=True)

categorical_cols=df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col].fillna('Unknown',inplace=True)


df['month_year'] = pd.to_datetime(df['month_year'], dayfirst=True, errors='coerce')
df['month']=df['month_year'].dt.month
df['year']=df['month_year'].dt.year

df['avg_comp_price']=df[['comp_1','comp_2','comp_3']].mean(axis=1)
df['price_elasticity']=df['unit_price']/df['avg_comp_price']

df['prev_month_qty']=df.groupby('product_id')['qty'].shift(1)
df['prev_month_qty'].fillna(df['qty'].median(),inplace=True)

df.to_csv('retail_price_cleaned.csv',index=False)
print("\nCleaned dataset saved")

plt.figure(figsize=(8,5))
sns.histplot(df['qty'],bins=30,kde=True)
plt.title('Units Sold Distribution')
plt.xlabel('Units Sold')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(data=df,x='product_category_name',y='qty',estimator=np.mean)
plt.title('Average Units Sold per Product Category')
plt.xlabel('Category')
plt.ylabel('Avg Units Sold')
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(data=df,x='price_elasticity',y='qty',alpha=0.6)
plt.title('Price Elasticity vs Units Sold')
plt.xlabel('Price / Avg Competitor Price')
plt.ylabel('Units Sold')
plt.show()