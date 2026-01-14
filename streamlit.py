# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import Lasso

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
    st.title("Work Models Dashboard")
    st.markdown("Predict High/Low Sales, Weekly Sales, Monthly Forecast, and view Association Rules based on your input.")

    import ast
    # Load reference data for input ranges
    feature_ref = pd.read_csv("sample_predictions.csv")

    # Load models safely
    try:
        clf_model = joblib.load("final_classification_model.pkl")  # Decision Tree
    except Exception as e:
        st.error(f"Failed to load classification model: {e}")
        clf_model = None

    try:
        reg_model = joblib.load("final_regression_pipeline.pkl")   # Decision Tree Regressor
    except Exception as e:
        st.error(f"Failed to load regression model: {e}")
        reg_model = None

    try:
        ts_model = joblib.load("final_time_series_model.pkl")      # Lasso
    except Exception as e:
        st.error(f"Failed to load time series model: {e}")
        ts_model = None

    try:
        rules_df = pd.read_csv("final_association_rules.csv")        # Apriori rules
    except Exception as e:
        st.error(f"Failed to load association rules: {e}")
        rules_df = pd.DataFrame()

    # --- USER INPUT ---
    col1, col2, col3 = st.columns(3)
    with col1:
        store = st.number_input(
            "Store ID", int(feature_ref['Store'].min()), int(feature_ref['Store'].max()), int(feature_ref['Store'].min())
        )
        dept = st.number_input(
            "Department ID", int(feature_ref['Dept'].min()), int(feature_ref['Dept'].max()), int(feature_ref['Dept'].min())
        )
        size = st.number_input(
            "Store Size", int(feature_ref['Size'].min()), int(feature_ref['Size'].max()), int(feature_ref['Size'].mean())
        )
    with col2:
        month = st.number_input("Month", 1, 12, 1)
        is_holiday = st.selectbox("Is Holiday?", [0, 1])
    with col3:
        temp = st.number_input(
            "Temperature", float(feature_ref['Temperature'].min()), float(feature_ref['Temperature'].max()), float(feature_ref['Temperature'].mean())
        )
        fuel = st.number_input(
            "Fuel Price", float(feature_ref['Fuel_Price'].min()), float(feature_ref['Fuel_Price'].max()), float(feature_ref['Fuel_Price'].mean())
        )
        unemp = st.number_input(
            "Unemployment (%)", float(feature_ref['Unemployment'].min()), float(feature_ref['Unemployment'].max()), float(feature_ref['Unemployment'].mean())
        )

    # Prepare input DataFrame for models
    # Prepare input DataFrame for models
    # We add the "missing" columns using defaults from your data (feature_ref)
    input_df = pd.DataFrame({
        "Store": [store],
        "Dept": [dept],
        "Month": [month],
        "Year": [2012],  # Added: Using 2012 as a standard reference year
        "IsHoliday": [is_holiday],
        "Size": [size],
        "Temperature": [temp],
        "Fuel_Price": [fuel],
        "CPI": [feature_ref['CPI'].mean()],
        "Unemployment": [unemp],
        
        # --- Missing Regression Features ---
        "Type": [feature_ref['Type'].mode()[0]], # Added: Most common store type
        "MarkDown1": [feature_ref['MarkDown1'].mean()], # Added
        "MarkDown2": [feature_ref['MarkDown2'].mean()], # Added
        "MarkDown3": [feature_ref['MarkDown3'].mean()], # Added
        "MarkDown4": [feature_ref['MarkDown4'].mean()], # Added
        "MarkDown5": [feature_ref['MarkDown5'].mean()], # Added
        
        # --- Missing Time Series Features ---
        "Lag_1": [feature_ref['Actual_Weekly_Sales'].mean()], # Added: Proxy for previous sales
        "Lag_12": [feature_ref['Actual_Weekly_Sales'].mean()], # Added: Proxy for last year sales
        "Rolling_Mean_3": [feature_ref['Actual_Weekly_Sales'].mean()] # Added: Proxy for trend
    })

    # --- PREDICTIONS ---
    if st.button("Predict"):
        st.subheader("Prediction Results")

        # Classification
        if clf_model is not None:
            try:
                class_pred = clf_model.predict(input_df)[0]
                class_prob = clf_model.predict_proba(input_df)[0][class_pred]
                st.write("### Decision Tree (Classification)")
                st.success(f"{'High Sales' if class_pred==1 else 'Low Sales'} (Prob: {class_prob:.2f})")
            except Exception as e:
                st.error(f"Classification prediction failed: {e}")

        # Regression
        if reg_model is not None:
            try:
                reg_pred = reg_model.predict(input_df)[0]
                st.write("### Decision Tree Regressor (Regression)")
                st.success(f"RM {reg_pred:,.2f} weekly sales")
            except Exception as e:
                st.error(f"Regression prediction failed: {e}")

        # Time Series Forecast
        if ts_model is not None:
            try:
                ts_pred = ts_model.predict(input_df)[0]
                st.write("### Lasso (Time Series Forecast)")
                st.success(f"RM {ts_pred:,.2f} monthly forecast")
            except Exception as e:
                st.error(f"Time series prediction failed: {e}")

        # Association Rules
        st.write("### Apriori (Association Rules)")
        
        try:
            work_rules_df = pd.read_csv("final_association_rules.csv")
        except:
            work_rules_df = pd.DataFrame()

        if not work_rules_df.empty:
            # Helper function to clean 'frozenset({1})' into '1'
            def clean_format(text):
                for char in ["frozenset", "{", "}", "(", ")", "'"]:
                    text = text.replace(char, "")
                return text

            # Filter rules containing the selected department in the antecedent
            match = work_rules_df[
                work_rules_df['antecedents'].astype(str).str.contains(f"{{{dept}}}") | 
                work_rules_df['antecedents'].astype(str).str.contains(f" {dept},") |
                work_rules_df['antecedents'].astype(str).str.contains(f" {dept}}}")
            ].head(5)

            if not match.empty:
                for i, row in match.iterrows():
                    # Use the cleaning function instead of ast.literal_eval
                    ant = clean_format(str(row['antecedents']))
                    con = clean_format(str(row['consequents']))
                    
                    st.success(f"**IF** a customer buys from **Dept {ant}**, **THEN** they are likely to buy **Dept {con}**")
                    st.caption(f"Lift: {row['lift']:.2f} | Confidence: {row['confidence']:.2%}")
            else:
                st.info(f"No specific rules found for Department {dept}.")
        else:
            st.info("Association rules file not found.")
