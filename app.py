# --------------------------------------------------------
# Streamlit App for Plastic Waste Prediction Dashboard
# --------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------
# Configuration
# ----------------------------
st.set_page_config(
    page_title="🌍 Plastic Waste Dashboard",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2a5298;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 8px 16px;
    }
    .prediction-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

DATA_PATH = "Plastic_Waste_Level_Dataset..csv"     # Update if renamed
MODEL_PATH = "best_plastic_waste_model.joblib"     # Saved best model from notebook

# ----------------------------
# Load dataset
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# ----------------------------
# Load Model
# ----------------------------
model_loaded = None
encoder = None
num_cols = []
cat_cols = []

if Path(MODEL_PATH).exists():
    try:
        model_data = joblib.load(MODEL_PATH)
        model_loaded = model_data["model"]
        encoder = model_data["encoder"]
        num_cols = model_data["num_cols"]
        cat_cols = model_data["cat_cols"]
        st.sidebar.success("Model loaded successfully")
    except Exception as e:
        st.sidebar.error(f"Model loading failed: {e}")
else:
    st.sidebar.warning("No model file found. Train & save the best model in the notebook first.")

# ----------------------------
# Sidebar Filters
# ----------------------------
with st.sidebar:
    st.markdown("### 🎛️ Dashboard Controls")
    
    st.markdown("#### 🌍 Geographic Filters")
    country_list = ["All"] + sorted(df["Country"].unique())
    country_filter = st.selectbox("🏳️ Select Country", country_list)
    
    source_list = ["All"] + sorted(df["Main_Sources"].unique())
    source_filter = st.selectbox("🏭 Main Source Filter", source_list)
    
    st.markdown("#### 📊 Range Filters")
    recycle_min, recycle_max = float(df['Recycling_Rate'].min()), float(df['Recycling_Rate'].max())
    recycle_range = st.slider("♻️ Recycling Rate (%)", recycle_min, recycle_max, (recycle_min, recycle_max))
    
    waste_min, waste_max = float(df['Per_Capita_Waste_KG'].min()), float(df['Per_Capita_Waste_KG'].max())
    waste_range = st.slider("🗑️ Per Capita Waste (kg)", waste_min, waste_max, (waste_min, waste_max))
    
    st.markdown("---")
    if st.button("🔄 Reset Filters", use_container_width=True):
        st.rerun()

# ----------------------------
# Apply Filters
# ----------------------------
df_filtered = df.copy()

if country_filter != "All":
    df_filtered = df_filtered[df_filtered["Country"] == country_filter]

if source_filter != "All":
    df_filtered = df_filtered[df_filtered["Main_Sources"] == source_filter]

df_filtered = df_filtered[
    (df_filtered["Recycling_Rate"] >= recycle_range[0]) &
    (df_filtered["Recycling_Rate"] <= recycle_range[1]) &
    (df_filtered["Per_Capita_Waste_KG"] >= waste_range[0]) &
    (df_filtered["Per_Capita_Waste_KG"] <= waste_range[1])
]

# ----------------------------
# Header + KPIs
# ----------------------------
st.markdown("""
<div class="main-header">
    <h1>🌍 Global Plastic Waste Analytics Dashboard</h1>
    <p>Advanced analytics and ML-powered predictions for sustainable waste management</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🏭 Total Waste (MT)", 
        f"{df['Total_Plastic_Waste_MT'].sum():,.0f}",
        delta=f"{df['Total_Plastic_Waste_MT'].std():.0f} std"
    )

with col2:
    st.metric(
        "♻️ Avg Recycling Rate", 
        f"{df['Recycling_Rate'].mean():.1f}%",
        delta=f"{df['Recycling_Rate'].max() - df['Recycling_Rate'].mean():.1f}% to max"
    )

with col3:
    st.metric(
        "👤 Per Capita Waste", 
        f"{df['Per_Capita_Waste_KG'].mean():.1f} kg",
        delta=f"{df['Per_Capita_Waste_KG'].median() - df['Per_Capita_Waste_KG'].mean():.1f} vs median"
    )

with col4:
    st.metric(
        "🌍 Countries Analyzed", 
        df["Country"].nunique(),
        delta=f"{len(df_filtered)} records filtered"
    )

# ----------------------------
# Dataset Preview
# ----------------------------
with st.expander("📋 View Filtered Dataset Sample", expanded=False):
    st.dataframe(
        df_filtered.head(10).style.highlight_max(axis=0),
        use_container_width=True
    )
    st.caption(f"Showing 10 of {len(df_filtered)} filtered records")

# ----------------------------
# Visualizations
# ----------------------------
st.markdown("## 📊 Interactive Analytics")

tab1, tab2, tab3, tab4 = st.tabs(["🏆 Rankings", "📈 Correlations", "🥧 Distribution", "🔥 Heatmap"])

with tab1:
    col1, col2 = st.columns([3, 1])
    with col2:
        top_n = st.slider("Countries to show", 5, 20, 10)
    
    top_countries = df.groupby("Country")["Total_Plastic_Waste_MT"].sum().sort_values(ascending=False).head(top_n)
    
    fig1 = px.bar(
        top_countries,
        x=top_countries.values,
        y=top_countries.index,
        orientation="h",
        title=f"🏭 Top {top_n} Plastic Waste Generating Countries",
        color=top_countries.values,
        color_continuous_scale="Reds"
    )
    fig1.update_layout(showlegend=False, height=500)
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    fig2 = px.scatter(
        df_filtered,
        x="Per_Capita_Waste_KG",
        y="Recycling_Rate",
        color="Plastic_Waste_Level",
        size="Population_Millions",
        hover_data=["Country"],
        title="🔄 Per Capita Waste vs Recycling Efficiency",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig2.update_layout(height=500)
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        src_counts = df_filtered["Main_Sources"].value_counts().reset_index()
        src_counts.columns = ["Main_Sources", "count"]
        
        fig3 = px.pie(
            src_counts, 
            names="Main_Sources", 
            values="count", 
            title="🏭 Waste Sources Distribution",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        level_counts = df_filtered["Plastic_Waste_Level"].value_counts().reset_index()
        level_counts.columns = ["Level", "count"]
        
        fig3b = px.pie(
            level_counts,
            names="Level",
            values="count",
            title="⚠️ Waste Level Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.4
        )
        st.plotly_chart(fig3b, use_container_width=True)

with tab4:
    num_df = df_filtered.select_dtypes(include=["number"])
    corr_matrix = num_df.corr()
    
    fig4 = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="🔥 Feature Correlation Matrix",
        color_continuous_scale="RdBu_r"
    )
    fig4.update_layout(height=600)
    st.plotly_chart(fig4, use_container_width=True)

# ----------------------------
# Prediction Section
# ----------------------------
st.markdown("---")

st.markdown("""
<div class="prediction-container">
    <h2>🔮 AI-Powered Waste Level Prediction</h2>
    <p>Use machine learning to predict plastic waste levels based on country characteristics</p>
</div>
""", unsafe_allow_html=True)

if model_loaded is None:
    st.error("🚫 Model not available. Please train and save the model first.")
else:
    with st.form("prediction_form"):
        st.markdown("### 📝 Input Parameters")
        
        input_data = {}
        
        if num_cols and cat_cols:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Numerical Features")
                for col in num_cols:
                    input_data[col] = st.number_input(
                        f"📈 {col.replace('_', ' ').title()}",
                        value=float(df[col].mean()),
                        help=f"Range: {df[col].min():.2f} - {df[col].max():.2f}"
                    )
            
            with col2:
                st.markdown("#### 🏷️ Categorical Features")
                for col in cat_cols:
                    options = sorted(df[col].unique().tolist())
                    input_data[col] = st.selectbox(
                        f"🔖 {col.replace('_', ' ').title()}",
                        options
                    )
        
        submit = st.form_submit_button("🚀 Generate Prediction", use_container_width=True)

    if submit:
        with st.spinner("🔄 Processing prediction..."):
            X_num = pd.DataFrame([{col: input_data[col] for col in num_cols}])
            X_cat = pd.DataFrame([{col: input_data[col] for col in cat_cols}])

            if encoder is not None and len(cat_cols) > 0:
                cat_encoded = encoder.transform(X_cat)
                X_cat = pd.DataFrame(cat_encoded, columns=cat_cols)

            final_input = pd.concat([X_num, X_cat], axis=1)
            pred = model_loaded.predict(final_input)[0]
            
            # Enhanced prediction display
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if pred == "High":
                    st.error(f"🚨 **Prediction: {pred} Waste Level**")
                elif pred == "Medium":
                    st.warning(f"⚠️ **Prediction: {pred} Waste Level**")
                else:
                    st.success(f"✅ **Prediction: {pred} Waste Level**")

# ----------------------------
# Download & Export
# ----------------------------
st.markdown("---")
with st.container():
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("### 📥 Export Data")
        st.caption(f"Download {len(df_filtered)} filtered records")
    
    with col2:
        st.download_button(
            label="📊 Download CSV",
            data=df_filtered.to_csv(index=False),
            file_name="filtered_plastic_waste.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col3:
        st.download_button(
            label="📈 Download JSON",
            data=df_filtered.to_json(orient="records"),
            file_name="filtered_plastic_waste.json",
            mime="application/json",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🌍 Built with Streamlit • Data-driven insights for sustainable future</p>
</div>
""", unsafe_allow_html=True)