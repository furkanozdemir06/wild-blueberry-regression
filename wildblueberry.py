import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# 1. Page Configuration
st.set_page_config(page_title="Wild Blueberry Yield", page_icon="🫐", layout="wide")
st.title("🫐 Wild Blueberry Harvest Yield Prediction")

# 2. Data Loading 
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    try:
        return pd.read_csv("train.csv")
    except:
        return None

# Widget cache 
uploaded_file = st.sidebar.file_uploader("Upload dataset (train.csv)", type=["csv"])
df = load_data(uploaded_file)

if df is None:
    st.info("👆 Please upload `train.csv` from the sidebar to start.")
    st.stop()

# Feature & Target Preparation
X = df.drop(columns=["id", "yield"], errors="ignore")
y = df["yield"]

# 3. Modern Tabbed Layout
tab_overview, tab_eda, tab_train, tab_predict = st.tabs([
    "📋 Overview", "📊 EDA", "🤖 Model Training", "🔮 Predict"
])

# --- TAB 1: OVERVIEW ---
with tab_overview:
    cols = st.columns(3)
    cols[0].metric("Total Samples", df.shape[0])
    cols[1].metric("Feature Count", X.shape[1])
    cols[2].metric("Missing Values", df.isnull().sum().sum())
    st.dataframe(df.head(10), use_container_width=True)

# --- TAB 2: EDA ---
with tab_eda:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# --- TAB 3: MODEL TRAINING ---
with tab_train:
    test_size = st.slider("Test Size Ratio", 0.1, 0.4, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
    }
    
    if st.button("Train Models"):
        results = []
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            results.append({
                "Model": name,
                "MAE": round(mean_absolute_error(y_test, preds), 4),
                "R² Score": round(r2_score(y_test, preds), 4)
            })
        st.table(pd.DataFrame(results))

# --- TAB 4: PREDICTION TOOL ---
with tab_predict:
    st.subheader("Input Feature Values")
    
    inputs = {}
    cols = st.columns(3)
    for i, col_name in enumerate(X.columns):
        with cols[i % 3]:
            inputs[col_name] = st.number_input(col_name, value=float(X[col_name].mean()))

    if st.button("🚀 Predict Yield"):
        rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
        input_df = pd.DataFrame([inputs])
        pred = rf.predict(input_df)[0]
        st.balloons()
        st.success(f"🌾 **Estimated Harvest Yield:** `{pred:.2f}`")