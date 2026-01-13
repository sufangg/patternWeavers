# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.metrics import confusion_matrix

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(page_title="Walmart Retail Dashboard", layout="wide")

# =====================
# MODULAR FUNCTIONS 
# =====================

# --- UI ---
def export_importance(df, filename):
    df.to_csv(filename, index=False)

# --- CLASSIFICATION ---
def load_class_data():
    return pd.read_csv("final_classification_predictions.csv")

def load_model(): 
    return joblib.load("final_classification_model.pkl")

def get_class_importance():
    df = pd.read_csv("final_classification_feature_importance.csv")
    top_10 = df.sort_values("importance", ascending=False).head(10)
    export_importance(top_10, "final_classification_feature_importance.csv")
    return top_10

# --- REGRESSION ---
def load_reg_data():
    return pd.read_csv("sample_predictions.csv")

def train_model(): 
    return joblib.load("final_regression_pipeline.pkl")

def get_reg_importance():
    df = pd.read_csv("feature_importance.csv")
    top_10 = df.sort_values("importance", ascending=False).head(10)
    export_importance(top_10, "final_feature_importance.csv")
    return top_10

# --- TIME SERIES ---
def load_ts_data():
    df = pd.read_csv("final_time_series_forecast.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def get_components():
    df = pd.read_csv("final_time_series_feature_importance.csv")
    top_10 = df.sort_values("importance", ascending=False).head(10)
    export_importance(top_10, "final_time_series_components.csv")
    return top_10

# --- ASSOCIATION ---
def load_rules():
    return pd.read_csv("top_association_rules.csv")

def get_top_rules(df):
    # Sort by Lift (Primary) and Confidence (Secondary)
    return df.sort_values(by=['lift', 'confidence'], ascending=False).head(10)

# =====================
# SIDEBAR & NAVIGATION
# =====================
st.sidebar.title("Walmart Retail Analytics")
page = st.sidebar.radio("Navigate", ["Home", "Classification", "Regression", "Time Series", "Association Rules"])

# =====================
# PAGE CONTENT
# =====================

if page == "Home":
    st.title("Walmart Retail Dashboard")
    
    st.markdown("""
    Welcome to the **Walmart Retail Analytics Dashboard**, where you can explore insights from multiple data mining task.  

    **1. Classification**  
    - View prediction results per store/month  
    - Analyze model performance via a confusion matrix  
    - Examine the top 10 influential features  

    **2. Regression**  
    - Compare actual vs predicted weekly sales  
    - Check top 10 most important features  

    **3. Time Series**  
    - Forecast sales vs actual trends  
    - Review top 10 drivers/components affecting predictions  

    **4. Association Rules**  
    - Explore top 10 association rules ranked by lift  
    - Visualize rule strength and lift vs confidence relationships  

    Navigate using the sidebar to dive into each data mining task.
    """)


elif page == "Classification":
    st.title("Classification Dashboard")
    df = load_class_data()
    fi = get_class_importance()
    
    st.subheader("Prediction Table")
    st.dataframe(df, height=350)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(df["actual_class"], df["predicted_class"])
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                           x=['Pred 0', 'Pred 1'], y=['Act 0', 'Act 1'], height=350)
    st.plotly_chart(fig_cm, use_container_width=True)
        
    st.subheader("Top 10 Feature Importance")
    fig_fi = px.bar(fi, x='importance', y='feature', orientation='h', color='importance')
    st.plotly_chart(fig_fi, use_container_width=True)

elif page == "Regression":
    st.title("Regression Dashboard")
    df = load_reg_data()
    fi = get_reg_importance()
    
    st.subheader("Actual vs Predicted Weekly Sales")
    fig_reg = px.line(df, y=['Actual_Weekly_Sales', 'Predicted_Weekly_Sales'], 
                      title="Sales Performance Tracking")
    st.plotly_chart(fig_reg, use_container_width=True)

    st.subheader("Top 10 Feature Importance")
    fig_fi = px.bar(fi, x='importance', y='feature', orientation='h', color='importance')
    st.plotly_chart(fig_fi, use_container_width=True)

elif page == "Time Series":
    st.title("Time Series Dashboard")
    df_ts = load_ts_data()
    comp = get_components()
    
    st.subheader("Forecast vs Actual")
    df_plot = df_ts.groupby('timestamp')[['actual', 'forecast']].sum().reset_index()
    fig_ts = px.line(df_plot, x='timestamp', y=['actual', 'forecast'], color_discrete_sequence=['blue', 'red'])
    st.plotly_chart(fig_ts, use_container_width=True)

    st.subheader("Time Series Components (Top Drivers)")
    fig_comp = px.bar(comp, x='importance', y='feature', orientation='h', color='importance')
    st.plotly_chart(fig_comp, use_container_width=True)

elif page == "Association Rules":
    st.title("Association Rules")
    rules_df = load_rules()
    top_rules = get_top_rules(rules_df)
    
    st.subheader("Top 10 Rules (Ranked by Lift)")
    st.dataframe(top_rules, use_container_width=True)

    st.subheader("Top 10 Purchase Rules by Lift Strength")
    top_rules['rule'] = top_rules['antecedents'].astype(str) + " -> " + top_rules['consequents'].astype(str)
    fig_rule_bar = px.bar(top_rules, x='lift', y='rule', orientation='h', 
                          color='lift', title="Rule Strength Analysis")
    st.plotly_chart(fig_rule_bar, use_container_width=True)
    
    st.subheader("Lift vs Confidence Analysis")
    fig_scatter = px.scatter(rules_df, x="confidence", y="lift", size="support", color="lift",
                             hover_data=["antecedents", "consequents"])
    st.plotly_chart(fig_scatter, use_container_width=True)
