import streamlit as st
import pandas
import numpy
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
  st.sidebar.file_uploader("Upload your CSV file")
  
