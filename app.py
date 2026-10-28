import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Medical Cost Predictor & Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        border-left: 5px solid #2b82d9;
    }
    .badge-obese { background-color: #fee2e2; color: #991b1b; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
    .badge-normal { background-color: #dcfce7; color: #166534; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
    .badge-overweight { background-color: #fef3c7; color: #92400e; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Data Loading & Model Training (Cached)
# ---------------------------------------------------------
@st.cache_data
def load_dataset():
    df = pd.read_csv("insurance.csv")
    df = df.drop_duplicates().reset_index(drop=True)
    return df


@st.cache_resource
def train_models(df):
    # Encode categorical features matching notebook specifications
    df_encoded = df.copy()
    df_encoded['sex_encoded'] = (df_encoded['sex'] == 'male').astype(int)
    df_encoded['smoker_encoded'] = (df_encoded['smoker'] == 'yes').astype(int)
    
    # One-hot encode region with 'northeast' as baseline
    region_dummies = pd.get_dummies(df_encoded['region'], prefix='region', drop_first=False)
    for col in ['region_northwest', 'region_southeast', 'region_southwest']:
        df_encoded[col] = region_dummies[col].astype(int)

    feature_cols = [
        'age', 'bmi', 'children', 'sex_encoded', 'smoker_encoded',
        'region_northwest', 'region_southeast', 'region_southwest'
    ]

    X = df_encoded[feature_cols]
    y = df_encoded['charges']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 1. Multiple Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    lr_metrics = {
        'R2': r2_score(y_test, y_pred_lr),
        'MAE': mean_absolute_error(y_test, y_pred_lr),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lr))
    }

    # 2. Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_metrics = {
        'R2': r2_score(y_test, y_pred_rf),
        'MAE': mean_absolute_error(y_test, y_pred_rf),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf))
    }

    return {
        'lr_model': lr,
        'rf_model': rf,
        'feature_cols': feature_cols,
        'lr_metrics': lr_metrics,
        'rf_metrics': rf_metrics,
        'X_train': X_train,
        'X_test': X_test,
        'y_test': y_test
    }


df = load_dataset()
models = train_models(df)

# ---------------------------------------------------------
# Sidebar - User Inputs
# ---------------------------------------------------------
st.sidebar.header("👤 Patient Profile")
st.sidebar.markdown("Provide demographic & lifestyle attributes:")

age = st.sidebar.slider("Age (years)", min_value=18, max_value=65, value=30, step=1)
sex = st.sidebar.radio("Sex", options=["female", "male"], horizontal=True)
bmi = st.sidebar.slider("BMI (Body Mass Index)", min_value=15.0, max_value=55.0, value=25.0, step=0.1)

# Clinical BMI category display
if bmi < 18.5:
    bmi_badge = "Underweight (< 18.5)"
    badge_class = "badge-overweight"
elif bmi < 25.0:
    bmi_badge = "Normal weight (18.5 - 24.9)"
    badge_class = "badge-normal"
elif bmi < 30.0:
    bmi_badge = "Overweight (25.0 - 29.9)"
    badge_class = "badge-overweight"
else:
    bmi_badge = "Obese (≥ 30.0)"
    badge_class = "badge-obese"

st.sidebar.markdown(f"**BMI Category:** <span class='{badge_class}'>{bmi_badge}</span>", unsafe_allow_html=True)

children = st.sidebar.slider("Number of Children / Dependents", min_value=0, max_value=5, value=0, step=1)
smoker = st.sidebar.selectbox("Smoking Status", options=["no", "yes"], index=0, help="Smoking has the strongest impact on medical charges.")
region = st.sidebar.selectbox("US Region", options=["northeast", "northwest", "southeast", "southwest"])

model_choice = st.sidebar.selectbox(
    "Prediction Algorithm",
    options=["Random Forest Regressor (Recommended - Nonlinear)", "Multiple Linear Regression (OLS)"]
)

# ---------------------------------------------------------
# Main UI Tabs
# ---------------------------------------------------------
st.title("🏥 Medical Cost Prediction & Statistics Dashboard")
st.markdown("An end-to-end Machine Learning and Statistical Analysis system based on the **Medical Cost Personal Dataset**.")

tab1, tab2, tab3 = st.tabs(["🔮 Cost Predictor", "📊 Exploratory Data Analysis", "🧪 Statistical Models & Benchmarks"])

# ---------------------------------------------------------
# TAB 1: Cost Predictor
# ---------------------------------------------------------
with tab1:
    st.subheader("Personalized Insurance Cost Estimation")
    
    # Prepare input feature row
    user_data = {
        'age': age,
        'bmi': bmi,
        'children': children,
        'sex_encoded': 1 if sex == 'male' else 0,
        'smoker_encoded': 1 if smoker == 'yes' else 0,
        'region_northwest': 1 if region == 'northwest' else 0,
        'region_southeast': 1 if region == 'southeast' else 0,
        'region_southwest': 1 if region == 'southwest' else 0,
    }
    input_df = pd.DataFrame([user_data])[models['feature_cols']]

    # Model selection
    active_model = models['rf_model'] if "Random Forest" in model_choice else models['lr_model']
    active_metrics = models['rf_metrics'] if "Random Forest" in model_choice else models['lr_metrics']
    
    predicted_charge = active_model.predict(input_df)[0]
    predicted_charge = max(0.0, float(predicted_charge))
    monthly_charge = predicted_charge / 12

    # Display Metrics in Columns
    col1, col2, col3 = st.columns([1.5, 1.2, 1.3])
    
    with col1:
        st.metric(
            label="Estimated Annual Medical Charge",
            value=f"${predicted_charge:,.2f}",
            delta=f"+${(predicted_charge - df['charges'].mean()):,.2f} vs Avg" if predicted_charge > df['charges'].mean() else f"-${(df['charges'].mean() - predicted_charge):,.2f} vs Avg"
        )
    with col2:
        st.metric(
            label="Estimated Monthly Payment",
            value=f"${monthly_charge:,.2f}"
        )
    with col3:
        # Risk / Cost Tier determination
        if predicted_charge < 5000:
            tier_label, tier_color = "Low Cost Tier", "🟢"
        elif predicted_charge < 15000:
            tier_label, tier_color = "Moderate Cost Tier", "🟡"
        elif predicted_charge < 30000:
            tier_label, tier_color = "High Cost Tier", "🟠"
        else:
            tier_label, tier_color = "Very High Cost Tier", "🔴"
        st.metric(label="Cost Risk Classification", value=f"{tier_color} {tier_label}")

    st.markdown("---")

    # Input Summary & Key Cost Drivers
    st.markdown("### 📋 Profile Summary & Primary Drivers")
    c1, c2 = st.columns([1, 1])

    with c1:
        st.dataframe(pd.DataFrame({
            "Attribute": ["Age", "Sex", "BMI", "Children", "Smoker", "Region"],
            "Value": [f"{age} yrs", sex.capitalize(), f"{bmi:.1f} ({bmi_badge.split('(')[0].strip()})", str(children), smoker.upper(), region.capitalize()]
        }), use_container_width=True, hide_index=True)

    with c2:
        st.markdown("**Model Performance for Chosen Algorithm:**")
        st.write(f"- **Test Set R² Score:** `{active_metrics['R2']:.4f}` ({active_metrics['R2']*100:.1f}% variance explained)")
        st.write(f"- **Mean Absolute Error (MAE):** `${active_metrics['MAE']:,.2f}`")
        st.write(f"- **Root Mean Squared Error (RMSE):** `${active_metrics['RMSE']:,.2f}`")
        
        if smoker == "yes":
            st.warning("⚠️ **Smoker Flag:** Smoking is the dominant driver of high medical charges in this dataset (contributing on average ~$24,000+ per beneficiary).")
        if bmi >= 30.0 and smoker == "yes":
            st.error("🚨 **High Risk Interaction:** Beneficiaries who are both Obese (BMI ≥ 30) and Smokers exhibit the highest risk with exponential cost increases.")

# ---------------------------------------------------------
# TAB 2: Exploratory Data Analysis
# ---------------------------------------------------------
with tab2:
    st.subheader("Dataset Exploratory Data Analysis (EDA)")
    
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    col_kpi1.metric("Total Beneficiaries", f"{len(df):,}")
    col_kpi2.metric("Mean Annual Charge", f"${df['charges'].mean():,.2f}")
    col_kpi3.metric("Median Annual Charge", f"${df['charges'].median():,.2f}")
    col_kpi4.metric("Smoker Percentage", f"{(df['smoker'] == 'yes').mean()*100:.1f}%")

    st.markdown("---")

    col_plot1, col_plot2 = st.columns(2)

    with col_plot1:
        st.markdown("#### 1. Charges Distribution: Smokers vs Non-Smokers")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data=df, x='charges', hue='smoker', kde=True, bins=30, palette={'no': '#2b82d9', 'yes': '#e63946'}, ax=ax)
        ax.set_title("Medical Charges by Smoking Status")
        ax.set_xlabel("Charges ($)")
        st.pyplot(fig)

    with col_plot2:
        st.markdown("#### 2. BMI vs Charges (Stratified by Smoker)")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df, x='bmi', y='charges', hue='smoker', alpha=0.7, palette={'no': '#2b82d9', 'yes': '#e63946'}, ax=ax)
        ax.axvline(30, color='gray', linestyle='--', label='Obesity Threshold (30)')
        ax.set_title("BMI vs Charges: Notice Non-Smoker vs Smoker Separation")
        ax.set_xlabel("BMI")
        ax.set_ylabel("Charges ($)")
        ax.legend()
        st.pyplot(fig)

    col_plot3, col_plot4 = st.columns(2)

    with col_plot3:
        st.markdown("#### 3. Age vs Medical Charges")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df, x='age', y='charges', hue='smoker', alpha=0.6, palette={'no': '#2b82d9', 'yes': '#e63946'}, ax=ax)
        ax.set_title("Age vs Charges (3 Distinct Cost Strata)")
        ax.set_xlabel("Age")
        ax.set_ylabel("Charges ($)")
        st.pyplot(fig)

    with col_plot4:
        st.markdown("#### 4. Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(6, 4))
        numeric_corr = df[['age', 'bmi', 'children', 'charges']].corr()
        sns.heatmap(numeric_corr, annot=True, cmap='Blues', fmt='.3f', ax=ax, cbar=False)
        ax.set_title("Pearson Correlation Matrix (Numerical Columns)")
        st.pyplot(fig)

# ---------------------------------------------------------
# TAB 3: Statistical Models & Benchmarks
# ---------------------------------------------------------
with tab3:
    st.subheader("Statistical Testing & Model Evaluation")
    
    st.markdown("""
    ### 🔬 Statistical Findings From the Project
    - **Independent Two-Sample Welch's t-test**:
      - Hypotheses: $H_0: \mu_{\\text{smoker}} = \mu_{\\text{non-smoker}}$ vs $H_1: \mu_{\\text{smoker}} \\neq \mu_{\\text{non-smoker}}$
      - **Result:** $t = 32.75$, $p < 10^{-100}$. Reject $H_0$; smokers incur statistically significantly higher charges.
    - **Two-Way ANOVA (Smoker × Sex)**:
      - The main effect of smoking is overwhelming ($F > 1300$, $p < 0.001$).
      - Smoker × Sex interaction is also statistically significant.
    - **Multicollinearity (VIF)**:
      - All Variance Inflation Factor values are $< 2.0$, confirming no problematic collinearity among baseline regressors.
    """)

    st.markdown("---")
    st.markdown("### 📈 Model Benchmark Comparison")
    
    benchmark_df = pd.DataFrame([
        {
            'Algorithm': 'Multiple Linear Regression',
            'Test R²': f"{models['lr_metrics']['R2']:.4f}",
            'Test MAE ($)': f"${models['lr_metrics']['MAE']:,.2f}",
            'Test RMSE ($)': f"${models['lr_metrics']['RMSE']:,.2f}",
            'Strengths': 'Highly interpretable, standard statistical baseline'
        },
        {
            'Algorithm': 'Random Forest Regressor (depth=6)',
            'Test R²': f"{models['rf_metrics']['R2']:.4f}",
            'Test MAE ($)': f"${models['rf_metrics']['MAE']:,.2f}",
            'Test RMSE ($)': f"${models['rf_metrics']['RMSE']:,.2f}",
            'Strengths': 'Captures non-linear BMI × Smoker interaction, lower error'
        }
    ])
    st.table(benchmark_df)
