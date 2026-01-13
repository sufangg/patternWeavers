# -*- coding: utf-8 -*- 
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# =====================
# SIDEBAR NAVIGATION
# =====================
st.set_page_config(page_title="Walmart Retail Dashboard", layout="wide")
st.sidebar.title("Walmart Retail Dashboard")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Classification", "Regression", "Time Series", "Association Rules"]
)

# =====================
# FUNCTIONS
# =====================

# -------- Classification --------
def load_class_data():
    return pd.read_csv('final_classification_predictions.csv')

def get_class_importance():
    df = pd.read_csv('final_classification_feature_importance.csv')
    return df.sort_values('importance', ascending=False).head(10)

def plot_confusion_matrix(df, true_col='actual_class', pred_col='predicted_class'):
    labels = sorted(df[true_col].unique())
    cm = confusion_matrix(df[true_col], df[pred_col], labels=labels)
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted Class', fontsize=12)
    ax.set_ylabel('Actual Class', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    return fig

# -------- Regression --------
def load_reg_data():
    return pd.read_csv('sample_predictions.csv')

def load_reg_model():
    return joblib.load('final_regression_pipeline.pkl')

def get_reg_coefficients():
    model = load_reg_model()
    try:
        coef = model.named_steps['model'].feature_importances_
        features = pd.read_json('features_used.json')['features_used']
    except:
        coef = getattr(model, 'coef_', None)
        features = [f'Feature_{i}' for i in range(len(coef))]
    return pd.DataFrame({'feature': features, 'coefficient': coef}).sort_values('coefficient', ascending=False)

def get_reg_importance():
    df = pd.read_csv('feature_importance.csv')
    return df.sort_values('importance', ascending=False).head(10)

# -------- Time Series --------
def load_ts_data():
    return pd.read_csv('final_time_series_forecast.csv')

def get_ts_importance():
    df = pd.read_csv('final_time_series_feature_importance.csv')
    return df.sort_values('importance', ascending=False).head(10)

# -------- Association Rules (placeholder) --------
def load_rules():
    return pd.read_csv('final_association_rules.csv')

def get_top_rules():
    df = load_rules()
    return df.sort_values(by=['lift', 'confidence'], ascending=False).head(10)

# =====================
# PAGE CONTENT
# =====================

if page == "Home":
    st.title("Smart AI Dashboard")
    st.markdown("""
    **Available Pipelines:**
    - Classification: predicted class distribution + confusion matrix + top 10 features
    - Regression: actual vs predicted + coefficients + top 10 features
    - Time Series: forecast + top 10 features
    - Association Rules: top 10 rules + lift vs confidence (placeholder)
    """)

elif page == "Classification":
    st.title("Classification Dashboard")
    df = load_class_data()

    # Toggle between table and charts
    view_option = st.radio("Select view:", ["Table", "Charts"])

    if view_option == "Table":
        st.subheader("Prediction Table")
        st.dataframe(df, use_container_width=True)
    else:
        # Predicted class distribution (bar chart)
        st.subheader("Predicted Class Distribution")
        fig = px.histogram(
            df,
            x="predicted_class",
            color="predicted_class",
            title="Predicted Class Counts",
            labels={"predicted_class":"Predicted Class"},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(
            xaxis_title="Predicted Class",
            yaxis_title="Number of Instances",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # Confusion matrix (heatmap)
        st.subheader("Confusion Matrix")
        cm_fig = plot_confusion_matrix(df, true_col='actual_class', pred_col='predicted_class')
        st.pyplot(cm_fig)

    # Top 10 feature importance (bar chart)
    st.subheader("Top 10 Feature Importance")
    st.bar_chart(get_class_importance().set_index('feature'))

elif page == "Regression":
    st.title("Regression Dashboard")
    df = load_reg_data()

    # Actual vs Predicted line chart
    st.subheader("Actual vs Predicted")
    st.line_chart(df.set_index(df.columns[0])[['Actual_Weekly_Sales', 'Predicted_Weekly_Sales']])

    # Top 10 Feature Importance (precomputed)
    st.subheader("Top 10 Feature Importance (CSV)")
    st.bar_chart(get_reg_importance().set_index('feature'))
    
    # Model Coefficients
    st.subheader("Model Coefficients")
    coeff_df = get_reg_coefficients()
    st.bar_chart(coeff_df.set_index('feature'))

elif page == "Time Series":
    st.title("Time Series Dashboard")
    df_ts = load_ts_data()

    st.subheader("Actual vs Forecast")
    st.line_chart(df_ts.set_index('timestamp')[['actual', 'forecast']])

    st.subheader("Top 10 Feature Importance")
    st.bar_chart(get_ts_importance().set_index('feature'))

elif page == "Association Rules":
    st.title("Association Rules Dashboard")
    st.info("placeholder - awaiting Person A data")
