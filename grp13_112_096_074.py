import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm

# 1. Dashboard Page Configuration
st.set_page_config(page_title="Customer Churn Analytical Dashboard", layout="wide")

st.title("Dynamic Customer Churn Analytical Dashboard")
st.markdown("**Group ID:** 112_096_074 | **Sample Size:** 2,501 Records")

# 2. Data Loading, Sampling, and Preprocessing
@st.cache_data
def load_data():
    sheet_url = 'https://docs.google.com/spreadsheets/d/1GwrWQcp17kluJOW_7_ODe60AbWofQY1_nKY49ekF6p0/export?format=csv&gid=0'
    try:
        df = pd.read_csv('grp13_sampled_data.csv')
    except Exception:
        df_raw = pd.read_csv(sheet_url)
        random_seed = 112096074
        random.seed(random_seed)
        np.random.seed(random_seed)
        df = df_raw.sample(n=2501, random_state=random_seed)
        df.to_csv('grp13_sampled_data.csv', index=False)
    
    # Cast numerical columns to explicit numeric types to prevent tab rendering errors
    numeric_columns = ['age', 'tenure_months', 'monthly_fee', 'csat_score', 'total_revenue', 'avg_session_time', 'churn']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df

grp13_112_096_074 = load_data()

# 3. Interactive Sidebar Filters
st.sidebar.header("Global Filters")

gender_options = list(grp13_112_096_074['gender'].dropna().unique())
gender_filter = st.sidebar.multiselect("Select Gender", options=gender_options, default=gender_options)

contract_options = list(grp13_112_096_074['contract_type'].dropna().unique())
contract_filter = st.sidebar.multiselect("Select Contract Type", options=contract_options, default=contract_options)

filtered_df = grp13_112_096_074[
    (grp13_112_096_074['gender'].isin(gender_filter)) & 
    (grp13_112_096_074['contract_type'].isin(contract_filter))
]

# Guard against empty filter selections
if filtered_df.empty:
    st.warning("⚠️ No data available for the selected filter combination. Please select at least one option in the sidebar.")
    st.stop()

# 4. Navigation Tabs
tab_overview, tab_desc, tab_vis, tab_infer, tab_causal = st.tabs([
    "📋 Overview & Raw Data",
    "📊 Descriptive Statistics",
    "📈 Visualizations & Correlation",
    "🔬 Inferential Statistics",
    "⚙️ Causal Analysis & Regression"
])

# TAB 1: OVERVIEW & RAW DATA
with tab_overview:
    st.subheader("Dataset Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Sample Size", f"{len(filtered_df)} records")
    col2.metric("Total Variables", f"{filtered_df.shape[1]} attributes")
    col3.metric("Churn Rate", f"{(filtered_df['churn'].mean() * 100):.2f}%")
    
    st.subheader("Raw Sampled Data Preview")
    st.dataframe(filtered_df.head(100), use_container_width=True)

# TAB 2: DESCRIPTIVE STATISTICS
with tab_desc:
    st.subheader("Non-Categorical Statistics")
    num_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
    
    desc_df = filtered_df[num_cols].describe().T
    desc_df['Median'] = filtered_df[num_cols].median()
    desc_df['Mode'] = filtered_df[num_cols].mode().iloc[0]
    desc_df['Range'] = desc_df['max'] - desc_df['min']
    desc_df['Std Dev'] = desc_df['std']
    desc_df['Variance'] = filtered_df[num_cols].var()
    desc_df['Skewness'] = filtered_df[num_cols].skew()
    desc_df['Kurtosis'] = filtered_df[num_cols].kurtosis()
    
    st.dataframe(
        desc_df[['count', 'mean', 'Median', 'Mode', 'min', 'max', 'Range', 'Std Dev', 'Variance', '25%', '50%', '75%', 'Skewness', 'Kurtosis']], 
        use_container_width=True
    )
    
    st.subheader("Categorical Variable Analysis")
    cat_cols = filtered_df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        selected_cat = st.selectbox("Select Categorical Feature", cat_cols, index=0)
        
        cat_counts = filtered_df[selected_cat].value_counts()
        cat_props = filtered_df[selected_cat].value_counts(normalize=True) * 100
        cat_summary = pd.DataFrame({'Frequency': cat_counts, 'Relative Frequency (%)': cat_props})
        
        col_c1, col_c2 = st.columns([1, 2])
        with col_c1:
            st.dataframe(cat_summary, use_container_width=True)
            st.info(f"**Highest Frequency:** {cat_counts.idxmax()} ({cat_counts.max()})\n\n**Lowest Frequency:** {cat_counts.idxmin()} ({cat_counts.min()})")
        with col_c2:
            fig_cat, ax_cat = plt.subplots(figsize=(6, 3.5))
            sns.barplot(x=cat_summary.index, y=cat_summary['Frequency'], ax=ax_cat, palette='viridis')
            ax_cat.set_title(f"Frequency Distribution: {selected_cat}")
            plt.xticks(rotation=45)
            st.pyplot(fig_cat)

# TAB 3: VISUALIZATIONS & CORRELATION
with tab_vis:
    st.subheader("Measures of Correlation")
    selected_num_cols = [c for c in ['age', 'tenure_months', 'monthly_fee', 'csat_score', 'total_revenue', 'avg_session_time'] if c in filtered_df.columns]
    
    corr_type = st.radio("Select Correlation Method", ["Pearson", "Spearman"], horizontal=True)
    corr_matrix = filtered_df[selected_num_cols].corr(method=corr_type.lower())
    
    c_col1, c_col2 = st.columns([1, 1.5])
    with c_col1:
        st.dataframe(corr_matrix, use_container_width=True)
    with c_col2:
        fig_hm, ax_hm = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_hm)
        ax_hm.set_title(f"{corr_type} Correlation Heatmap")
        st.pyplot(fig_hm)
        
    st.subheader("Visual Exploration")
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        st.write("**Scatter Plot: Monthly Fee vs Total Revenue**")
        fig_sc, ax_sc = plt.subplots(figsize=(5, 3.5))
        sns.scatterplot(x='monthly_fee', y='total_revenue', hue='churn', data=filtered_df, ax=ax_sc)
        st.pyplot(fig_sc)
        
        st.write("**Violin Plot: CSAT Score by Churn**")
        fig_v, ax_v = plt.subplots(figsize=(5, 3.5))
        sns.violinplot(x='churn', y='csat_score', data=filtered_df, ax=ax_v, palette='muted')
        st.pyplot(fig_v)
        
    with v_col2:
        st.write("**Box Plot: Tenure by Contract Type**")
        fig_bx, ax_bx = plt.subplots(figsize=(5, 3.5))
        sns.boxplot(x='contract_type', y='tenure_months', data=filtered_df, ax=ax_bx, palette='Set2')
        st.pyplot(fig_bx)
        
        st.write("**Pie Plot: Signup Channel Distribution**")
        fig_pie, ax_pie = plt.subplots(figsize=(5, 3.5))
        filtered_df['signup_channel'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax_pie, startangle=90)
        ax_pie.set_ylabel('')
        st.pyplot(fig_pie)

# TAB 4: INFERENTIAL STATISTICS
with tab_infer:
    st.subheader("Statistical Tests")
    col_i1, col_i2 = st.columns(2)
    
    with col_i1:
        st.markdown("### 1. Test of Normality (Shapiro-Wilk)")
        try:
            stat_norm, p_norm = stats.shapiro(filtered_df['age'].dropna())
            st.write(f"**Variable:** Age")
            st.write(f"**Statistic:** {stat_norm:.4f}, **P-Value:** {p_norm:.4e}")
        except Exception as e:
            st.error(f"Error computing Normality test: {e}")
        
        st.markdown("---")
        st.markdown("### 2. Two-Sample T-Test (Test of Mean)")
        try:
            churn_csat = filtered_df[filtered_df['churn'] == 1]['csat_score'].dropna()
            retain_csat = filtered_df[filtered_df['churn'] == 0]['csat_score'].dropna()
            t_stat, t_pval = stats.ttest_ind(churn_csat, retain_csat)
            st.write(f"**Testing CSAT Score between Churned vs Retained**")
            st.write(f"**T-Statistic:** {t_stat:.4f}, **P-Value:** {t_pval:.4e}")
        except Exception as e:
            st.error(f"Error computing T-test: {e}")
        
    with col_i2:
        st.markdown("### 3. Chi-Square Test of Independence")
        try:
            contingency = pd.crosstab(filtered_df['contract_type'], filtered_df['churn'])
            chi2, p_chi2, dof, _ = stats.chi2_contingency(contingency)
            st.write(f"**Variables:** Contract Type vs Churn")
            st.write(f"**Chi2 Stat:** {chi2:.4f}, **DOF:** {dof}, **P-Value:** {p_chi2:.4e}")
        except Exception as e:
            st.error(f"Error computing Chi-Square test: {e}")
        
        st.markdown("---")
        st.markdown("### 4. Test of Variance (Levene's Test)")
        try:
            churn_csat = filtered_df[filtered_df['churn'] == 1]['csat_score'].dropna()
            retain_csat = filtered_df[filtered_df['churn'] == 0]['csat_score'].dropna()
            lev_stat, lev_pval = stats.levene(churn_csat, retain_csat)
            st.write(f"**Testing Variance of CSAT Score across Churn groups**")
            st.write(f"**Levene Stat:** {lev_stat:.4f}, **P-Value:** {lev_pval:.4e}")
        except Exception as e:
            st.error(f"Error computing Levene's test: {e}")

# TAB 5: CAUSAL REGRESSION MODELING
with tab_causal:
    st.subheader("Regression Models")
    col_r1, col_r2 = st.columns(2)
    
    with col_r1:
        st.markdown("### Linear OLS Regression")
        st.caption("Target: CSAT Score | Predictors: Monthly Fee, Tenure, Age")
        try:
            reg_df = filtered_df[['csat_score', 'monthly_fee', 'tenure_months', 'age']].dropna()
            X_lin = reg_df[['monthly_fee', 'tenure_months', 'age']]
            X_lin = sm.add_constant(X_lin)
            y_lin = reg_df['csat_score']
            ols_model = sm.OLS(y_lin, X_lin).fit()
            st.text(str(ols_model.summary()))
        except Exception as e:
            st.error(f"Error running Linear Regression: {e}")
        
    with col_r2:
        st.markdown("### Logistic Regression Model")
        st.caption("Target: Churn (Binary) | Predictors: Tenure, Monthly Fee, CSAT Score")
        try:
            logit_df = filtered_df[['churn', 'tenure_months', 'monthly_fee', 'csat_score']].dropna()
            X_log = logit_df[['tenure_months', 'monthly_fee', 'csat_score']]
            X_log = sm.add_constant(X_log)
            y_log = logit_df['churn']
            logit_model = sm.Logit(y_log, X_log).fit(disp=0)
            st.text(str(logit_model.summary()))
        except Exception as e:
            st.error(f"Error running Logistic Regression: {e}")
