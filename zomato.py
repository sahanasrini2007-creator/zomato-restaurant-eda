import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r"C:\Users\Lenovo\Downloads\EDA THIRANEX\zomato.csv")
print(df.shape) #to fetch no of rows and columns
print(df.head())#to fetch the first five rows from the dataset

print(df.info()) #to fetch the datatypes
print(df.isna().sum()) #to fetch the number of missing values per column
print(df.duplicated().sum())#to fetch the number of duplicate rows

# what drives a restaurant's rating on zomato?
#Dataset:<https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants?select=zomato.csv>

df=df.drop(columns=['url','phone','address','reviews_list'],errors="ignore")
df=df.rename(columns={"approx_cost(for two people)":"cost","menu_items":"menu","listed_in(type)":"type"})
df=df.drop_duplicates()
df["rate"] = df["rate"].astype(str).str.replace("/5","",regex=False).str.strip()
df["rate"]=pd.to_numeric(df["rate"],errors='coerce')
df["cost"]=df["cost"].astype(str).str.replace(",","",regex=False)
df["cost"]=pd.to_numeric(df["cost"],errors="coerce")
df["cost"] = df["cost"].fillna(df["cost"].median())
df=df.dropna(subset=["rate","location","cuisines"])
df.to_csv(r"C:\Users\Lenovo\Downloads\EDA THIRANEX\zomato.clean.csv",index=False)

#plots
plt.figure(figsize=(10,5))
sns.histplot(df["rate"],kde=True)
plt.title("Distribution of restaurant ratings")
plt.xlabel("Ratings")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
top_loc=df["location"].value_counts().head(10)
sns.barplot(x=top_loc.values,y=top_loc.index)
plt.title("Top 10 locations by restaurant")
plt.show()

plt.figure(figsize=(10,5))
cuis=df["cuisines"].str.split(", ").explode().value_counts().head(10)
sns.barplot(x=cuis.values,y=cuis.index)
plt.title("top 10 cuisines")
plt.xlabel("Restaurants")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(data=df,x="online_order",y="rate")
plt.title("Rating by online ordering")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
sns.scatterplot(data=df,x="cost",y="rate")
plt.title("Cost vs rating")
plt.tight_layout()
plt.show()

#correlations and findings

d=df.copy()
d["online_order"]=(d["online_order"].astype(str).str.strip().str.lower().map({"yes":1,"no":0}))
d["book_table"]=(d["book_table"].astype(str).str.strip().str.lower().map({"yes":1,"no":0}))
cols=["rate","votes","online_order","book_table","cost"]
corr_matrix=d[cols].corr(method="spearman")
print(corr_matrix)
sns.heatmap(corr_matrix,annot=True,cmap="coolwarm")
plt.title("spearman correlation heatmap")
plt.tight_layout()
plt.show()

print("\nCorrelation with restaurant ratings:")
print(corr_matrix["rate"].sort_values(ascending=False))
