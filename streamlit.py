# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import confusion_matrix

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Walmart Retail AI Dashboard",
    layout="wide"
)

# ======================================================
# SIDEBAR NAVIGATION
# ======================================================
st.sidebar.title("Walmart Retail Dashboard")
page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Classification",
        "Regression",
        "Time Series",
        "Association Rules"
    ]
)

# ======================================================
# DATA LOADING FUNCTIONS (NO TRAINING)
# ======================================================

# ---------------------- CLASSIFICATION ----------------------
def load_class_data():
    return pd.read_csv("final_classification_predictions.csv")

def get_class_importance():
    df = pd.read_csv("final_classification_feature_importance.csv")
    return df.sort_values("importance", ascending=False).head(10)

def plot_confusion_matrix(df):
    labels = sorted(df["actual_class"].unique())
    cm = confusion_matrix(
        df["actual_class"],
        df["predicted_class"],
        labels=labels
    )

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax
    )
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title("Confusion Matrix")
    return fig

# ---------------------- REGRESSION ----------------------
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

# ---------------------- TIME SERIES ----------------------
def load_ts_data():
    return pd.read_csv("final_time_series_forecast.csv")

def load_ts_components():
    return pd.read_csv("final_time_series_components.csv")

# ---------------------- ASSOCIATION RULES ----------------------
def load_rules():
    return pd.read_csv("final_association_rules.csv")

def get_top_rules():
    df = load_rules()
    return df.sort_values(
        by=["lift", "confidence"],
        ascending=False
    ).head(10)

# ======================================================
# PAGE CONTENT
# ======================================================

# ---------------------- HOME ----------------------
if page == "Home":
    st.title("📊 Walmart Retail Dashboard")
    st.markdown("""
    **This dashboard provides explainability and visualization for finalized models.**

    **Pipelines Included:**
    -  **Classification**: predictions, confusion matrix, feature importance
    -  **Regression**: actual vs predicted, feature impact, coefficients
    -  **Time Series**: forecast comparison, trend components
    -  **Association Rules**: top rules, lift vs confidence analysis)

# ---------------------- CLASSIFICATION ----------------------
elif page == "Classification":
    st.title("Classification Dashboard")

    df = load_class_data()

    st.subheader("Prediction Results")
    st.dataframe(df, use_container_width=True)

    st.subheader("Confusion Matrix")
    st.pyplot(plot_confusion_matrix(df))

    st.subheader("Top 10 Feature Importance")
    fi = get_class_importance()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(fi["feature"], fi["importance"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    ax.set_title("Most Influential Features")
    st.pyplot(fig)

    if "predicted_probability" in df.columns:
        st.subheader("Prediction Probability Distribution")
        fig_prob = px.histogram(
            df,
            x="predicted_probability",
            nbins=20,
            title="Predicted Probability Distribution"
        )
        st.plotly_chart(fig_prob, use_container_width=True)

# ---------------------- REGRESSION ----------------------
elif page == "Regression":
    st.title("Regression Dashboard")

    df = load_reg_data()

    st.subheader("Actual vs Predicted Values")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df.iloc[:, 0], df["Actual_Weekly_Sales"], label="Actual")
    ax.plot(df.iloc[:, 0], df["Predicted_Weekly_Sales"], label="Predicted")
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Weekly Sales")
    st.pyplot(fig)

    st.subheader("Top 10 Feature Importance")
    imp = get_reg_importance()

    fig_imp, ax = plt.subplots(figsize=(6, 4))
    ax.barh(imp["feature"], imp["importance"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    st.pyplot(fig_imp)

    st.subheader("Model Coefficients (Impact Direction)")
    coef = get_reg_coefficients()

    fig_coef, ax = plt.subplots(figsize=(6, 4))
    ax.barh(coef["feature"], coef["coefficient"])
    ax.axvline(0)
    ax.invert_yaxis()
    ax.set_xlabel("Coefficient Value")
    st.pyplot(fig_coef)

# ---------------------- TIME SERIES ----------------------
elif page == "Time Series":
    st.title("Time Series Dashboard")

    df_ts = load_ts_data()

    st.subheader("Actual vs Forecast")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_ts["timestamp"], df_ts["actual"], label="Actual")
    ax.plot(df_ts["timestamp"], df_ts["forecast"], label="Forecast")
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    st.pyplot(fig)

    st.subheader("Trend & Seasonality Components")
    st.dataframe(load_ts_components(), use_container_width=True)

# ---------------------- ASSOCIATION RULES ----------------------
elif page == "Association Rules":
    st.title("Association Rules Dashboard")

    rules = get_top_rules()

    st.subheader("Top 10 Association Rules")
    st.dataframe(rules, use_container_width=True)

    st.subheader("Lift vs Confidence Analysis")
    fig = px.scatter(
        rules,
        x="confidence",
        y="lift",
        size="support",
        hover_data=["antecedents", "consequents"],
        labels={
            "confidence": "Confidence",
            "lift": "Lift",
            "support": "Support"
        },
        title="Association Rules: Lift vs Confidence"
    )
    st.plotly_chart(fig, use_container_width=True)
