import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.title("Linear Regression Web Application")
st.subheader("Data Science With Poresh")

#sidebar
st.sidebar.header("Upload CSV Data Or Use Sample")
use_example = st.sidebar.checkbox("Use Example Dataset")

#Load Data
if use_example:
  df = sns.load_dataset('tips')
  df = df.dropna()
  st.success("Loaded sample dataset:'tips'")
else:
  uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type = ['CSV'])
  if uploaded_file:
    df = pd.read_csv(uploaded_file)
  else:
    st.warning("Please upload a file or use the example dataset")
    st.stop()


#Show Dataset

st.subheader("Dataset Preview")
st.write(df.head())

#Model feature selection

numeric_cols = df.select_dtypes(include = np.number).columns.tolist()
if len(numeric_cols) < 2:
  st.error("Need atleast two numeric columns for regression.")
  st.stop()

target = st.selectbox("Select target variable", numeric_cols)
features = st.multiselect("Select input feature columns", [col for col in numeric_cols if col != target], default = [col for col in numeric_cols if col != target])


if len(features) == 0:
  st.write("Please select atleast one feature")
  st.stop()


df = df[features + [target]].dropna()

x = df[features]
y = df[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split (X_scaled, y, test_size = 0, random_state = 42)

model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(X_test)





    
  
