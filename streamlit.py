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
page = st.sidebar.radio("Navigate", ["Home", "Classification", "Regression", "Time Series", "Association Rules", "Work Models"])

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
    
    st.markdown("""
    ### Business Insight:

    **What is happening:**  
    The confusion matrix shows that the classification model performs very well, with high correct predictions for both classes. Most observations fall along the diagonal, indicating strong model accuracy. There are very few false positives (2 cases) and a small number of false negatives (10 cases), suggesting the model is reliable in identifying both outcomes.

    The feature importance chart clearly shows that **Month** is the most influential feature, while **Year** and **Store** contribute very little to the prediction.

    **Why it matters:**  
    This indicates that **seasonality plays a dominant role** in determining the target class, while long-term trends (Year) and store location (Store) have minimal impact. Customer behavior changes significantly depending on the month, reflecting predictable seasonal demand patterns.

    The low misclassification rate means the model can be trusted for operational decision-making.

    **Decision supported:**  
    - Focus planning and decision-making around **monthly and seasonal trends**  
    - Apply **month-based strategies** for promotions, staffing, and inventory  
    - Reduce reliance on **store-specific or year-based assumptions**, as they add limited predictive value
    """)

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
    st.markdown("""
    ### Business Insight:
  
    **What is happening:**  
    The Actual vs Predicted Weekly Sales chart shows that the regression model closely follows actual sales patterns, including major peaks and drops. This indicates strong predictive performance, as the model successfully captures sales fluctuations over time with only minor deviations.

    The feature importance analysis shows that **Temperature** is the most influential factor affecting weekly sales, followed by **Department**. Other variables such as Store, MarkDowns, fuel price, and economic indicators contribute less significantly to the prediction.

    **Why it matters:**  
    This suggests that **external environmental conditions**, particularly temperature, have a strong impact on customer purchasing behavior. Different departments respond differently to these conditions, highlighting the importance of department-level demand sensitivity.
    
    Accurate sales prediction enables better operational planning and reduces uncertainty in decision-making.

    **Decision supported:**  
    - Incorporate **weather forecasts** into sales and inventory planning  
    - Adjust stock levels and promotions based on **department-specific demand patterns**  
    - Use the model to anticipate **sales spikes during extreme temperature periods**  
    - Reduce overstock and stockouts through improved demand forecasting
    """)

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
    st.markdown("""
    ### Business Insight:

    **What is happening:**  
    The forecasted sales closely follow the actual sales trend, indicating that the time series model effectively captures both short-term fluctuations and long-term patterns. The alignment between forecast and actual values demonstrates reliable forecasting performance.

    The feature importance analysis highlights **Lag_12** as the strongest contributor, indicating the presence of strong annual seasonality in sales behavior. Recent lag features also contribute, showing that recent sales influence near-term predictions.

    **Why it matters:**  
    Sales patterns tend to repeat on a yearly basis, making historical seasonal data highly valuable for forecasting future demand. Understanding these recurring patterns improves the accuracy of long-term sales planning.

    Reliable time series forecasting supports both strategic and operational decision-making.

    **Decision supported:**  
    - Support **long-term seasonal planning**, such as annual budgeting and promotional strategies  
    - Enable **short-term operational adjustments** based on recent sales trends  
    - Improve inventory planning by anticipating recurring seasonal demand cycles
    """)

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

    st.markdown("""
    ### Business Insight:

    **What is happening:**  
    The association rule analysis identifies strong co-purchasing patterns between specific departments and time periods. Rules with high lift values indicate that certain items or departments are purchased together more frequently than expected by chance, revealing meaningful relationships within customer baskets.

    The Lift vs Confidence analysis shows several rules with both high confidence and high lift, indicating that these purchasing relationships are reliable and not random.

    **Why it matters:**  
    High-lift association rules reveal valuable **cross-selling opportunities** and highlight how customer purchasing behavior changes across seasons. These insights help identify complementary products and department combinations that naturally belong together.

    Understanding these patterns allows Walmart to increase basket size and improve customer shopping experience.

    **Decision supported:**  
    - Design **bundle promotions** and cross-department discounts  
    - Optimize **product placement** by positioning complementary items closer together  
    - Improve **seasonal marketing strategies** based on co-purchasing behavior  
    - Enhance **inventory planning** by stocking related products together
    """)

elif page == "Work Models":
