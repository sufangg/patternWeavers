# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import confusion_matrix

# =====================
# PAGE CONFIG & SIDEBAR
# =====================
st.set_page_config(page_title="Walmart Retail Dashboard", layout="wide")

st.sidebar.title("Walmart Retail Dashboard")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Classification", "Regression", "Time Series", "Association Rules"]
)

# =====================
# DATA LOADING FUNCTIONS
# =====================

# -------- Classification --------
def load_class_data():
    return pd.read_csv("final_classification_predictions.csv")

def get_class_importance():
    df = pd.read_csv("final_classification_feature_importance.csv")
    return df.sort_values("importance", ascending=False).head(10)

def plot_confusion_matrix(df):
    cm = confusion_matrix(df["actual_class"], df["predicted_class"])

    fig, ax = plt.subplots(figsize=(4, 3))  # compact size
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
        ax=ax
    )
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig

# -------- Regression --------
def load_reg_data():
    return pd.read_csv("sample_predictions.csv")

def get_reg_importance():
    df = pd.read_csv("feature_importance.csv")
    return df.sort_values("importance", ascending=False).head(10)

def get_reg_coefficients():
    model = joblib.load("final_regression_pipeline.pkl")
    features = pd.read_json("features_used.json")["features_used"]

    coef = model.named_steps["model"].feature_importances_
    return pd.DataFrame({
        "feature": features,
        "coefficient": coef
    }).sort_values("coefficient", ascending=False)

# -------- Time Series --------
def load_ts_data():
    return pd.read_csv("final_time_series_forecast.csv")

def get_ts_importance():
    df = pd.read_csv("final_time_series_feature_importance.csv")
    return df.sort_values("importance", ascending=False).head(10)

# -------- Association Rules --------
def load_top_rules():
    return pd.read_csv("top_association_rules.csv")

# =====================
# PAGE CONTENT
# =====================

if page == "Home":
    st.title("Smart AI Dashboard")
    st.markdown("""
    **Available Pipelines**
    - **Classification**: prediction table, confusion matrix, top 10 features  
    - **Regression**: actual vs predicted, coefficients, top 10 features  
    - **Time Series**: forecast vs actual, top 10 features  
    - **Association Rules**: top rules table, lift vs confidence analysis  
    """)

# =====================
# CLASSIFICATION
# =====================
elif page == "Classification":
    st.title("Classification Dashboard")

    df = load_class_data()

    st.subheader("Prediction Table")
    st.dataframe(df, use_container_width=True)

    st.subheader("Confusion Matrix")
    cm_fig = plot_confusion_matrix(df)
    st.pyplot(cm_fig)

    st.subheader("Top 10 Feature Importance")
    st.bar_chart(get_class_importance().set_index("feature"))

# =====================
# REGRESSION
# =====================
elif page == "Regression":
    st.title("Regression Dashboard")

    df = load_reg_data()

    st.subheader("Actual vs Predicted Weekly Sales")
    st.line_chart(
        df.set_index(df.columns[0])[["Actual_Weekly_Sales", "Predicted_Weekly_Sales"]]
    )

    st.subheader("Top 10 Feature Importance")
    st.bar_chart(get_reg_importance().set_index("feature"))

    st.subheader("Model Coefficients")
    st.bar_chart(get_reg_coefficients().set_index("feature"))

# =====================
# TIME SERIES
# =====================
elif page == "Time Series":
    st.title("Time Series Dashboard")

    df_ts = load_ts_data()

    st.subheader("Actual vs Forecast")
    st.line_chart(df_ts.set_index("timestamp")[["actual", "forecast"]])

    st.subheader("Top 10 Feature Importance")
    st.bar_chart(get_ts_importance().set_index("feature"))

# =====================
# ASSOCIATION RULES
# =====================
elif page == "Association Rules":
    st.title("Association Rules Dashboard")

    rules_df = load_top_rules()

    st.subheader("Top Association Rules")
    st.dataframe(rules_df, use_container_width=True)

    st.subheader("Lift vs Confidence Analysis")
    fig = px.scatter(
        rules_df,
        x="confidence",
        y="lift",
        size="support",
        hover_data=["antecedents", "consequents"],
        labels={
            "confidence": "Rule Confidence",
            "lift": "Lift Value",
            "support": "Support"
        },
        title="Association Rules: Lift vs Confidence"
    )

    st.plotly_chart(fig, use_container_width=True)
