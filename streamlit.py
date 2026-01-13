# -*- coding: utf-8 -*- 
import streamlit as st
import pandas as pd
import joblib
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

def display_confusion_matrix(df):
    st.subheader("Confusion Matrix")
    # Compute the matrix
    cm = confusion_matrix(df['actual_class'], df['predicted_class'])
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted 0', 'Predicted 1'], 
                yticklabels=['Actual 0', 'Actual 1'], ax=ax)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(fig)
    
    # Interpretation for the user
    st.write("**Interpretation:**")
    st.write(f"- The model correctly identified {cm[1,1]} 'High Performers' (True Positives).")
    st.write(f"- It missed {cm[1,0]} actual 'High Performers' (False Negatives).")

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
    - Classification: prediction table + confusion matrix + top 10 features
    - Regression: actual vs predicted + coefficients + top 10 features
    - Time Series: forecast + top 10 features
    - Association Rules: top 10 rules + lift vs confidence (placeholder)
    """)

elif page == "Classification":
    st.title("Classification Dashboard")
    df = load_class_data()

    # --- Prediction Table ---
    st.subheader("Prediction Table")
    st.dataframe(df, use_container_width=True)

    # --- Confusion Matrix ---
    st.subheader("Confusion Matrix")
    cm_fig = plot_confusion_matrix(df, true_col='actual_class', pred_col='predicted_class')
    st.pyplot(cm_fig)

    # --- Top 10 Feature Importance ---
    st.subheader("Top 10 Feature Importance")
    st.bar_chart(get_class_importance().set_index('feature'))

elif page == "Regression":
    st.title("Regression Dashboard")
    df = load_reg_data()
    st.subheader("Actual vs Predicted")
    st.line_chart(df.set_index(df.columns[0])[['Actual_Weekly_Sales', 'Predicted_Weekly_Sales']])
    st.subheader("Top 10 Feature Importance")
    st.bar_chart(get_reg_importance().set_index('feature'))
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
    st.info("placeholder")
