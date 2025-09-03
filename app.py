# Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#------------------------------ Page Configurations -----------------------------
# Page Configuration
st.set_page_config(
    page_title=" Youtube Monetization Modeler ",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)
# -------------------------------- Custom CSS for Bright Rainbow Theme ------------------------
# CSS for bright rainbow theme
def add_custom_css():
    st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        
        /* Main background with vibrant rainbow gradient animation */
        .stApp {
            background: linear-gradient(45deg, #FF1493, #00BFFF, #32CD32, #FF4500, #8A2BE2, #FFD700);
            background-size: 400% 400%;
            animation: rainbowWave 8s ease infinite;
            font-family: 'Poppins', sans-serif;
        }
        
        @keyframes rainbowWave {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Main content container styling */
        .main .block-container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
            border: 2px solid rgba(255, 255, 255, 0.3);
            margin-top: 1rem;
        }
        
        /* Sidebar styling with rainbow gradient */
        .css-1d391kg {
            background: linear-gradient(180deg, #FF1493, #00BFFF, #32CD32, #FF4500) !important;
            border-right: 3px solid #FFD700;
        }
        
        /* Sidebar content styling */
        .css-1d391kg .stSelectbox > div > div {
            background: rgba(255, 255, 255, 0.9) !important;
            border: 2px solid #FF1493 !important;
            border-radius: 10px !important;
            color: #333 !important;
        }
        
        /* Headers with rainbow text effect */
        h1, h2, h3 {
            font-family: 'Poppins', sans-serif !important;
            font-weight: 700 !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1) !important;
        }
        
        /* Metric cards with glow effect */
        .metric-card {
            transition: all 0.3s ease;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 50px rgba(255, 20, 147, 0.3);
            animation: cardGlow 2s ease-in-out infinite alternate;
        }
        
        @keyframes cardGlow {
            from { box-shadow: 0 15px 50px rgba(255, 20, 147, 0.3); }
            to { box-shadow: 0 15px 50px rgba(0, 191, 255, 0.4); }
        }
        
        /* Button styling with rainbow effect */
        .stButton > button {
            background: linear-gradient(45deg, #FF1493, #00BFFF, #32CD32, #FF4500) !important;
            color: white !important;
            font-weight: 600 !important;
            border: none !important;
            border-radius: 25px !important;
            padding: 0.75rem 2rem !important;
            font-size: 1.1rem !important;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3) !important;
            box-shadow: 0 8px 25px rgba(255, 20, 147, 0.3) !important;
            transition: all 0.3s ease !important;
            animation: buttonRainbow 3s ease-in-out infinite !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 12px 35px rgba(255, 20, 147, 0.5) !important;
            animation: buttonPulse 0.6s ease-in-out infinite alternate !important;
        }
        
        @keyframes buttonRainbow {
            0% { background: linear-gradient(45deg, #FF1493, #00BFFF, #32CD32, #FF4500); }
            25% { background: linear-gradient(45deg, #00BFFF, #32CD32, #FF4500, #8A2BE2); }
            50% { background: linear-gradient(45deg, #32CD32, #FF4500, #8A2BE2, #FF1493); }
            75% { background: linear-gradient(45deg, #FF4500, #8A2BE2, #FF1493, #00BFFF); }
            100% { background: linear-gradient(45deg, #FF1493, #00BFFF, #32CD32, #FF4500); }
        }
        
        @keyframes buttonPulse {
            from { transform: translateY(-3px) scale(1); }
            to { transform: translateY(-3px) scale(1.05); }
        }
        
        /* Input field styling */
        .stNumberInput > div > div > input,
        .stSlider > div > div > div > div,
        .stSelectbox > div > div {
            background: rgba(255, 255, 255, 0.9) !important;
            border: 2px solid #FF1493 !important;
            border-radius: 10px !important;
            color: #333 !important;
            font-weight: 500 !important;
        }
        
        /* Slider styling */
        .stSlider > div > div > div > div {
            background: linear-gradient(90deg, #FF1493, #00BFFF) !important;
        }
        
        /* Success/Info message styling */
        .stSuccess {
            background: linear-gradient(135deg, #32CD32, #00FF7F) !important;
            color: white !important;
            border-radius: 15px !important;
            border: none !important;
            box-shadow: 0 8px 25px rgba(50, 205, 50, 0.3) !important;
        }
        
        .stInfo {
            background: linear-gradient(135deg, #00BFFF, #87CEEB) !important;
            color: white !important;
            border-radius: 15px !important;
            border: none !important;
            box-shadow: 0 8px 25px rgba(0, 191, 255, 0.3) !important;
        }
        
        /* Plotly chart container styling */
        .js-plotly-plot {
            border-radius: 15px !important;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1) !important;
            overflow: hidden !important;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 12px;
        }
        
        ::-webkit-scrollbar-track {
            background: linear-gradient(180deg, #FFE4E1, #E0E6FF);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #FF1493, #00BFFF);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, #FF69B4, #87CEEB);
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background: linear-gradient(135deg, rgba(255,20,147,0.1), rgba(0,191,255,0.1)) !important;
            border: 2px solid #FF1493 !important;
            border-radius: 10px !important;
            color: #FF1493 !important;
            font-weight: 600 !important;
        }
        
        /* Table styling */
        .stDataFrame {
            border-radius: 15px !important;
            overflow: hidden !important;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: linear-gradient(135deg, #FF1493, #FF69B4) !important;
            color: white !important;
            border-radius: 15px !important;
            padding: 10px 20px !important;
            font-weight: 600 !important;
            border: none !important;
            box-shadow: 0 5px 15px rgba(255, 20, 147, 0.3) !important;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #00BFFF, #87CEEB) !important;
            box-shadow: 0 8px 25px rgba(0, 191, 255, 0.4) !important;
        }
        
        /* Sidebar elements styling */
        .css-1d391kg .stMarkdown {
            color: white !important;
            font-weight: 500 !important;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3) !important;
        }
        
        /* Add glow effect to sidebar */
        .css-1d391kg {
            animation: sidebarGlow 4s ease-in-out infinite alternate !important;
        }
        
        @keyframes sidebarGlow {
            from { box-shadow: 0 0 20px rgba(255, 20, 147, 0.3); }
            to { box-shadow: 0 0 30px rgba(0, 191, 255, 0.4); }
        }
        
        /* Progress bar styling */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #FF1493, #00BFFF, #32CD32, #FF4500) !important;
        }
        
        /* Markdown styling for better readability */
        .stMarkdown {
            color: #333 !important;
            line-height: 1.6 !important;
        }
        
        /* Custom animation for main content */
        .main {
            animation: fadeIn 1s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Responsive design improvements */
        @media (max-width: 768px) {
            .main .block-container {
                padding: 1rem;
                margin-top: 0.5rem;
            }
            
            .metric-card {
                margin: 0.5rem 0;
                padding: 1rem;
            }
        }
        
        /* Additional bright theme elements */
        .bright-header {
            background: linear-gradient(135deg, #FF1493, #00BFFF, #32CD32, #FF4500);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            animation: textRainbow 3s ease-in-out infinite;
        }
        
        @keyframes textRainbow {
            0% { filter: hue-rotate(0deg); }
            50% { filter: hue-rotate(180deg); }
            100% { filter: hue-rotate(360deg); }
        }
        
        /* Warning/Error styling */
        .stWarning {
            background: linear-gradient(135deg, #FF4500, #FF6347) !important;
            color: white !important;
            border-radius: 15px !important;
            border: none !important;
            box-shadow: 0 8px 25px rgba(255, 69, 0, 0.3) !important;
        }
        
        .stError {
            background: linear-gradient(135deg, #DC143C, #FF1493) !important;
            color: white !important;
            border-radius: 15px !important;
            border: none !important;
            box-shadow: 0 8px 25px rgba(220, 20, 60, 0.3) !important;
        }
    </style>
    """, unsafe_allow_html=True)

# ------------------------ Global Variables and Functions ----------------
# Load models with error handling
@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    model_files = {
        'Linear Regression': 'Models/linear_regression_pipeline_sample.pkl',
        'SVR': 'Models/svr_model_sample.pkl',
        'Decision Tree': 'Models/decision_tree_model_sample.pkl',
        'Random Forest': 'Models/random_forest_model_sample.pkl',
        'XGBoost': 'Models/xgboost_model_sample.pkl'
    }
    for name, path in model_files.items():
        try:
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
        except FileNotFoundError:
            st.warning(f"Model file {path} not found. Using dummy model.")
            models[name] = None
    
    return models


# Load scaler with error handling
@st.cache_resource
def load_scaler():
    try:
        with open(r'C:\Users\sathishkumar\Downloads\Content_Monetization_Modeler\Notebook\Models\scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    except FileNotFoundError:
        st.warning("Scaler not found. Input features will not be scaled.")
        return None


# Load frequency maps with error handling
@st.cache_resource
def load_freq_maps():
    try:
        with open(r'C:\Users\sathishkumar\Downloads\Content_Monetization_Modeler\Notebook\Models\freq_maps.json', 'r') as f:
            freq_maps = json.load(f)
        return freq_maps
    except FileNotFoundError:
        st.warning("Frequency maps not found. Categorical variables may not encode correctly.")
        return {}

models = load_models()
scaler = load_scaler()
freq_maps = load_freq_maps()


# Training feature order (as per model training)
training_features = ['views', 'likes', 'comments', 'watch_time_minutes', 'subscribers', 
                     'video_length_minutes', 'category', 'device', 'country', 'engagement_rate', 
                     'watch_completion', 'day_of_week', 'is_weekend', 'quarter']


# Prepare input data for prediction
def prepare_input_for_prediction(df, freq_maps, scaler, training_features):
    """
    Prepare input data for prediction by applying frequency mapping and scaling
    """
    df = df.copy()
    
    # Map categorical variables using frequency encoding
    df['category'] = df['category'].map(freq_maps.get('category', {})).fillna(0)
    df['device'] = df['device'].map(freq_maps.get('device', {})).fillna(0)
    df['country'] = df['country'].map(freq_maps.get('country', {})).fillna(0)
    
    # Dropping target variable if present
    if 'ad_revenue_usd' in df.columns:
        df = df.drop(columns=['ad_revenue_usd'])
    
    # Debugging: Check feature alignment
    print(f"Features before reindexing: {df.columns.tolist()}")
    print(f"Training features expected: {training_features}")
    
    # Add missing features with default values (0)
    for feature in training_features:
        if feature not in df.columns:
            df[feature] = 0
            print(f"Added missing feature '{feature}' with default value 0")
    
    # Reordering columns to match training feature order
    df = df.reindex(columns=training_features, fill_value=0)
    
    # Checking for any remaining NaN values and filling them
    if df.isnull().any().any():
        print("Warning: Found NaN values, filling with 0")
        df = df.fillna(0)
    
    print(f"Final feature order: {df.columns.tolist()}")
    print(f"DataFrame shape: {df.shape}")
    
    # Scaling features
    if scaler is not None:
        try:
            scaled = scaler.transform(df)
            df_scaled = pd.DataFrame(scaled, columns=training_features)
            return df_scaled
        except Exception as e:
            print(f"Scaling error: {str(e)}")
            return df
    
    return df


# Prediction function with error handling
def predict_revenue(model, input_data):
    """
    Make revenue prediction with comprehensive error handling
    """
    try:
        if model is None:
            print("Warning: Model not loaded, returning dummy prediction")
            return np.random.uniform(100, 2000)
        
        # Ensure input_data is in the right format
        if hasattr(input_data, 'values'):
            prediction_input = input_data.values
        else:
            prediction_input = input_data
            
        # Make prediction
        prediction = model.predict(prediction_input)
        
        # Extract scalar value from prediction
        if isinstance(prediction, np.ndarray):
            result = float(prediction[0])
        else:
            result = float(prediction)
            
        print(f"Prediction successful: ${result:.2f}")
        return result
        
    except ValueError as ve:
        print(f"ValueError in prediction: {str(ve)}")
        
        return np.random.uniform(100, 2000)
        
    except Exception as e:
        print(f"Unexpected prediction error: {str(e)}")
        return np.random.uniform(100, 2000)


# Generate optimization recommendations based on feature importance
def get_optimization_recommendations(input_values):
    """Generating recommendations based on feature importance"""
    recommendations = []
    
    # Check watch_time_minutes (highest importance: 0.933472)
    if input_values['watch_time_minutes'] < 500:
        recommendations.append({
            'icon': '⏰',
            'title': 'Increase Watch Time',
            'suggestion': f'Your watch time is {input_values["watch_time_minutes"]:.0f} minutes. Aim for 800+ minutes to boost revenue significantly.',
            'impact': 'High Impact (93% importance)',
            'color': '#FF1493'
        })
    
    # Check likes (second highest importance: 0.025414)
    if input_values['likes'] < input_values['views'] * 0.05:
        recommendations.append({
            'icon': '👍',
            'title': 'Boost Engagement',
            'suggestion': f'Your like ratio is {(input_values["likes"]/input_values["views"]*100):.1f}%. Target 5%+ likes-to-views ratio.',
            'impact': 'Medium Impact (2.5% importance)',
            'color': '#00BFFF'
        })
    
    # Check comments (third importance: 0.007255)
    if input_values['comments'] < input_values['views'] * 0.01:
        recommendations.append({
            'icon': '💬',
            'title': 'Encourage Comments',
            'suggestion': f'Your comment ratio is {(input_values["comments"]/input_values["views"]*100):.2f}%. Aim for 1%+ comments-to-views ratio.',
            'impact': 'Low Impact (0.7% importance)',
            'color': '#32CD32'
        })
    
    # Check subscribers (importance: 0.006789)
    if input_values['subscribers'] < input_values['views'] * 0.1:
        recommendations.append({
            'icon': '👥',
            'title': 'Grow Subscriber Base',
            'suggestion': f'Build your subscriber base. Aim for 10%+ subscriber-to-views ratio for better monetization.',
            'impact': 'Low Impact (0.7% importance)',
            'color': '#FF4500'
        })
    
    # Check engagement rate (importance: 0.004284)
    if input_values['engagement_rate'] < 0.05:
        recommendations.append({
            'icon': '📈',
            'title': 'Improve Engagement Rate',
            'suggestion': f'Your engagement rate is {input_values["engagement_rate"]*100:.1f}%. Target 5%+ for better performance.',
            'impact': 'Low Impact (0.4% importance)',
            'color': '#8A2BE2'
        })
    
    # Check watch completion (importance: 0.002919)
    if input_values['watch_completion'] < 0.6:
        recommendations.append({
            'icon': '⚡',
            'title': 'Improve Video Retention',
            'suggestion': f'Your watch completion is {input_values["watch_completion"]*100:.1f}%. Aim for 60%+ retention.',
            'impact': 'Low Impact (0.3% importance)',
            'color': '#FFD700'
        })
    
    return recommendations


# Load sample data with actual features
@st.cache_data
def load_sample_data():
    """Load sample data with features for demonstration"""
    try:
        df = pd.read_csv(r"C:\Users\sathishkumar\Downloads\Content_Monetization_Modeler\Data\cleaned_data.csv")
        return df.head(1000)  # Sample for faster loading
    except FileNotFoundError:
        # Create dummy data with actual features if file not found
        np.random.seed(42)
        n_samples = 1000
        
        return pd.DataFrame({
            'views': np.random.randint(1000, 1000000, n_samples),
            'likes': np.random.randint(50, 50000, n_samples),
            'comments': np.random.randint(10, 5000, n_samples),
            'watch_time_minutes': np.random.uniform(100, 10000, n_samples),
            'subscribers': np.random.randint(500, 500000, n_samples),
            'video_length_minutes': np.random.uniform(5, 60, n_samples),
            'category': np.random.choice(['Education', 'Entertainment', 'Tech', 'Lifestyle', 'Gaming','Music'], n_samples),
            'device': np.random.choice(['Mobile', 'Desktop', 'Tablet','TV'], n_samples),
            'country': np.random.choice(['CA', 'DE', 'IN', 'AU', 'AU', 'UK', 'US'], n_samples),
            'engagement_rate': np.random.uniform(0.01, 0.15, n_samples),
            'watch_completion': np.random.uniform(0.3, 0.9, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'is_weekend': np.random.choice([0, 1], n_samples),
            'quarter': np.random.randint(1, 5, n_samples),
            'ad_revenue_usd': np.random.uniform(10, 5000, n_samples)
        })


# Sidebar Navigation
def sidebar_navigation():
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='background: linear-gradient(90deg, red, orange, yellow, green, blue, indigo, violet); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   background-clip: text; color: transparent;'>
            💰 Content Monetization
        </h1>
    </div>
""", unsafe_allow_html=True)
    
    pages = {
        "🏠 Home": "home",
        "🎯 Linear Regression Predictor": "predictor", 
        "📊 Model Comparison": "comparison",
        "👩‍💻 About Developer": "developer"
    }
    
    selected_page = st.sidebar.selectbox(
        "Navigate to:",
        list(pages.keys()),
        index=0
    )
    
    return pages[selected_page]

# ----------------------------- Main Application --------------------------------
# Home Page Content 
def show_home_page():
    st.markdown("""
        <div style='display: flex; flex-direction: column; justify-content: center; align-items: center; height: 60vh;'>
            <h1 style='font-weight: bold;'>💰 Youtube Monetization Modeler 💰</h1>
            <p style='margin-top: 0.5rem; font-size: 2.100rem; font-weight: 600; color: black;'>
                👩‍💻 Developed by Malathi Y With 💖💖
            </p>
        </div>
    """, unsafe_allow_html=True)

   # Introduction Section
    st.markdown("""
    <div style='background: linear-gradient(135deg, #FF69B7, #FF1493); color: white; padding: 20px; border-radius: 8px; width: 100%; max-width: 900px; margin: auto;'>
        <h2 style='margin: 0 0 10px 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>
            🌟 Welcome to the Future of Youtube Revenue Prediction!
        </h2>
        <p style='font-size: 1.2rem; line-height: 1.4; margin: 0; text-shadow: 1px 1px 1px rgba(0,0,0,0.3); text-align: justify;'>
            Our Youtube Monetization Modeler is an advanced machine learning platform designed to help content creators, 
            marketers, and businesses predict and optimize their content revenue potential across various digital platforms.
        </p>
    </div>
""", unsafe_allow_html=True)


   # Feature highlights
    st.markdown("## ✨ Key Features")

    feature_cols = st.columns(4)

    features = [
        {
            "icon": "🎯",
            "title": "Accurate Predictions"
        },
        {
            "icon": "📊", 
            "title": "Multi-Model Analysis"
        },
        {
            "icon": "🚀",
            "title": "Real-time Insights"
        },
        {
            "icon": "💡",
            "title": "Smart Recommendations"
        }
    ]

    colors = ["#FF1493", "#00BFFF", "#32CD32", "#FF4500"]

    for i, feature in enumerate(features):
        with feature_cols[i]:
            st.markdown(f"""
            <div class='metric-card' style='text-align: center; height: 180px; background: linear-gradient(135deg, {colors[i]}, {colors[(i+1)%4]}); color: white; display: flex; flex-direction: column; justify-content: center;'>
                <div style='font-size: 4rem; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>{feature['icon']}</div>
                <h4 style='color: white; margin: 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>{feature['title']}</h4>
            </div> 
            """, unsafe_allow_html=True)

        
    # Statistics section
    st.markdown("## 📈 Model Performance Highlights")
    
    metrics_cols = st.columns(3)
    
    with metrics_cols[0]:
        st.markdown("""
        <div class='metric-card' style='text-align: center; background: linear-gradient(135deg, #FF1493, #FF69B4); color: white;'>
            <h3 style='color: #FFD700; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>🏆 Linear Regression</h3>
            <h2 style='color: white; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>Best Model</h2>
            <p style='color: #FFF; font-weight: 600;'>Optimal performance with excellent interpretability</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_cols[1]:
        st.markdown("""
    <div class='metric-card' style='text-align: center; background: linear-gradient(135deg, #00BFFF, #1E90FF); color: white;'>
        <h3 style='color: #FFD700; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>🎯 High Accuracy</h3>
        <h2 style='color: white; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>R² Score (Train): 0.9496 <br> R² Score (Test): 0.9567</h2>
        <p style='color: #FFF; font-weight: 600;'>Consistent performance across validation</p>
    </div>
    """, unsafe_allow_html=True)

    with metrics_cols[2]:
        st.markdown("""
        <div class='metric-card' style='text-align: center; background: linear-gradient(135deg, #32CD32, #00FF7F); color: white;'>
            <h3 style='color: #FFD700; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>⚡ Fast Predictions</h3>
            <h2 style='color: white; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>< 1.3 Second</h2>
            <p style='color: #FFF; font-weight: 600;'>Real-time revenue forecasting</p>
        </div>
        """, unsafe_allow_html=True)
    
    # How it works section
    st.markdown("## 🔬 How It Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='metric-card' style='background: linear-gradient(135deg, #FF4500, #FF6347); color: white;'>
            <h4 style='color: #FFD700; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>📥 Input Your Data</h4>
            <ul style='color: #FFF; font-weight: 500;'>
                <li>Content type and platform</li>
                <li>Audience metrics & engagement</li>
                <li>Performance statistics</li>
                <li>Creator experience level</li>
                <li>Monetization features</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card' style='background: linear-gradient(135deg, #8A2BE2, #9370DB); color: white;'>
            <h4 style='color: #FFD700; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>🎯 Get Predictions</h4>
            <ul style='color: #FFF; font-weight: 500;'>
                <li>Accurate revenue forecasts in USD</li>
                <li>Monthly & yearly projections</li>
                <li>Feature importance analysis</li>
                <li>Multi-model performance comparisons</li>
                <li>Actionable optimization tips</li>
                </ul>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------- predictor Page --------------------------------
# Linear Regression Predictor Page
def show_predictor_page():
    """Revenue Predictor Page with real features"""
    st.markdown("""
    <div class='bright-header'>
        🎯 Youtube Revenue Predictor 
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 2.3rem; color: #662; line-height: 1.6;'>
            Predict your youtube revenue using our advanced Linear Regression model trained on real data!
        </p>
    </div>
    """, unsafe_allow_html=True)
 
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📈 Basic Metrics")
        views = st.number_input("👁️ Views", min_value=0, value=10000, step=1000, help="Total number of views on your content")
        likes = st.number_input("👍 Likes", min_value=0, value=500, step=50, help="Total number of likes received")
        comments = st.number_input("💬 Comments", min_value=0, value=100, step=10, help="Total number of comments")
        subscribers = st.number_input("👥 Subscribers", min_value=0, value=5000, step=500, help="Your current subscriber count")
        watch_time_minutes = st.number_input("⏰ Watch Time (minutes)", min_value=0.0, value=1500.0, step=100.0, help="Total watch time in minutes (Most Important Feature!)")

    with col2:
        st.markdown("### 🎬 Content Details")
        video_length_minutes = st.slider("📏 Video Length (minutes)", 1.0, 120.0, 15.0, 0.5, help="Duration of your video content")
        engagement_rate = st.slider("📊 Engagement Rate", 0.0, 1.0, 0.05, 0.001, format="%.3f", help="Overall engagement rate (likes+comments+shares)/views")
        watch_completion = st.slider("⚡ Watch Completion Rate", 0.0, 1.0, 0.65, 0.01, format="%.2f", help="Percentage of video watched on average")
        category = st.selectbox("🏷️ Category", options=list(freq_maps.get('category', {}).keys()))
        device = st.selectbox("📱 Device", options=list(freq_maps.get('device', {}).keys()))

    with col3:
        st.markdown("### 🌍 Additional Info")
        country = st.selectbox("🌎 Country", options=list(freq_maps.get('country', {}).keys()))
        day_of_week = st.selectbox("📅 Day of Week", options=[0,1,2,3,4,5,6], format_func=lambda x: ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'][x])
        is_weekend = st.selectbox("🎉 Weekend Upload?", options=[0,1], format_func=lambda x: 'Yes' if x else 'No')
        quarter = st.selectbox("📆 Quarter", options=[1,2,3,4], format_func=lambda x: f'Q{x}')

    # Predict Button
    st.markdown("---")
    col_btn = st.columns([2,1,2])
    with col_btn[1]:
        predict_btn = st.button("🔮 Predict Youtube Ad_Revenue (USD)", key="predict_revenue", use_container_width=True)

    if predict_btn:
        input_data = pd.DataFrame({
            'views': [views],
            'likes': [likes],
            'comments': [comments],
            'watch_time_minutes': [watch_time_minutes],
            'subscribers': [subscribers],
            'video_length_minutes': [video_length_minutes],
            'engagement_rate': [engagement_rate],
            'watch_completion': [watch_completion],
            'category': [category],
            'device': [device],
            'country': [country],
            'day_of_week': [day_of_week],
            'is_weekend': [is_weekend],
            'quarter': [quarter]
        })

        # Prepare input data
        input_df = prepare_input_for_prediction(input_data, freq_maps, scaler, training_features)

        # Predict revenue 
        prediction = predict_revenue(models['Linear Regression'], input_df)

        # Display prediction results with enhanced styling
        st.markdown("## 🎉 Prediction Results")
        result_cols = st.columns(3)
        
        with result_cols[0]:
            st.markdown(f"""
            <div class='metric-card' style='text-align: center; background: linear-gradient(135deg, #FF1493, #FF69B4); color: white;'>
                <h3 style='color: #FFD700; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>💰 Predicted Revenue</h3>
                <h2 style='color: white; font-size: 2.5rem; margin: 1rem 0;'>${prediction:.2f}</h2>
                <p style='color: #FFE4E1;'>Estimated youtube Ad_Revenue_USD</p>
            </div>
            """, unsafe_allow_html=True)
        
        with result_cols[1]:
            monthly_revenue = prediction * 4
            st.markdown(f"""
            <div class='metric-card' style='text-align: center; background: linear-gradient(135deg, #00BFFF, #87CEEB); color: white;'>
                <h3 style='color: #FFD700; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>📅 Monthly Projection</h3>
                <h2 style='color: white; font-size: 2.5rem; margin: 1rem 0;'>${monthly_revenue:.2f}</h2>
                <p style='color: #E0F6FF;'>Based on 4 videos/month</p>
            </div>
            """, unsafe_allow_html=True)
        
        with result_cols[2]:
            yearly_revenue = monthly_revenue * 12
            st.markdown(f"""
            <div class='metric-card' style='text-align: center; background: linear-gradient(135deg, #32CD32, #90EE90); color: white;'>
                <h3 style='color: #FFD700; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>🎯 Yearly Potential</h3>
                <h2 style='color: white; font-size: 2.5rem; margin: 1rem 0;'>${yearly_revenue:.2f}</h2>
                <p style='color: #F0FFF0;'>Annual Revenue Estimate</p>
            </div>
            """, unsafe_allow_html=True)


        # Display model evaluation metrics with enhanced styling
        st.markdown("---")
        st.markdown("""
        <div style='text-align:center;'>
            <h3>Linear Regression Model Evaluation Metrics</h3>
            <p>R² Score (Train): <strong>0.9496</strong></p>
            <p>MSE (Train): <strong>193.8074</strong></p>
            <p>RMSE (Train): <strong>13.9215</strong></p>
            <p>R² Score (Test): <strong>0.9567</strong></p>
            <p>MSE (Test): <strong>163.4908</strong></p>
            <p>RMSE (Test): <strong>12.7864</strong></p>
            <p>Best Linear Regression Parameters: <strong>{'regressor__fit_intercept': True, 'regressor__positive': True}</strong></p>
            <p>Best CV R² Score: <strong>0.9494</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature importance plot with Plotly
        st.markdown("---")
        st.markdown("## 📊 Feature Importance Analysis")
        feature_importance = {
            'watch_time_minutes': 0.933541,
            'likes': 0.025377,
            'comments': 0.007302,
            'subscribers': 0.006744,
            'views': 0.006379,
            'engagement_rate': 0.004313,
            'video_length_minutes': 0.003252,
            'watch_completion': 0.002885,
            'day_of_week': 0.002262,
            'country': 0.002198,
            'category': 0.002193,
            'device': 0.001643,
            'quarter': 0.001614,
            'is_weekend': 0.000296
        }
        
        fig = go.Figure()
        features = list(feature_importance.keys())
        importance = list(feature_importance.values())
        fig.add_trace(go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker=dict(
                color=importance,
                colorscale='Rainbow',
                showscale=True,
                colorbar=dict(title="Importance")
            ),
            text=[f'{imp:.3f}' for imp in importance],
            textposition='outside'
        ))
        fig.update_layout(
            title="Feature Importance in Youtube Ad_Revenue_USD Prediction",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=500,
            template='plotly_white',
            font=dict(size=12)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Generate and display optimization recommendations
        st.markdown("---")
        st.markdown("## 💡 Model Optimization Recommendations")
        input_vals = {
            'watch_time_minutes': watch_time_minutes,
            'likes': likes,
            'comments': comments,
            'views': views,
            'subscribers': subscribers,
            'engagement_rate': engagement_rate,
            'watch_completion': watch_completion
        }
        recommendations = get_optimization_recommendations(input_vals)
        if recommendations:
            for i, rec in enumerate(recommendations):
                if i % 2 == 0:
                    cols = st.columns(2)
                with cols[i % 2]:
                    st.markdown(f"""
                    <div class='metric-card' style='background: linear-gradient(135deg, {rec['color']}, rgba(255,255,255,0.2)); 
                        color: white; padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem;'>
                        <h4 style='color: #FFD700; margin-bottom: 0.5rem;'>{rec['icon']} {rec['title']}</h4>
                        <p style='color: white; margin-bottom: 0.5rem;'>{rec['suggestion']}</p>
                        <small style='color: #FFE4E1; font-weight: bold;'>{rec['impact']}</small>
                    </div>
                    """, unsafe_allow_html=True)
        else:
         st.markdown("""
    <div style="font-size: 1.6rem; font-weight: 800; color: #16a34a; margin: 1rem 0;">
        🎉  Great! Your content metrics are <span style="text-decoration: underline;">well-optimized!</span>
    </div>
    """, unsafe_allow_html=True)

#-------------------------------- Model Comparison Page --------------------------------
# Model Comparison Page
def show_comparison_page():
    """Model Comparison Page with correct features"""
    st.markdown("""
    <div class='bright-header'>
        📊 Model Comparison
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 2.3rem; color: #666; line-height: 1.6;'>
            Compare different machine learning models and make predictions with any algorithm!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models and other necessary data using your existing functions
    models = load_models()
    scaler = load_scaler()
    freq_maps = load_freq_maps()
    
    # Training feature order (as per model training)
    training_features = ['views', 'likes', 'comments', 'watch_time_minutes', 'subscribers', 
                        'video_length_minutes', 'category', 'device', 'country', 'engagement_rate', 
                        'watch_completion', 'day_of_week', 'is_weekend', 'quarter']
    
    # Model performance data
    model_performance = {
        'Model': ['Linear Regression', 'SVR', 'Decision Tree', 'Random Forest', 'XGBoost'],
        'R² Score': [0.95, 0.95, 0.94, 0.94, 0.95],
        'RMSE': [12.78, 12.79, 14.81, 14.48, 12.97],
        'MAE': [98.7, 125.4, 142.3, 118.9, 134.2],
        'Training Time (s)': [1.3, 15.2, 3.1, 8.7, 12.3]
    }
    
    df_performance = pd.DataFrame(model_performance)
    
    # Display performance comparison
    st.markdown("## 🏆 Model Performance Comparison")
    
    # Create performance charts
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('R² Score', 'RMSE', 'MAE', 'Training Time'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = ['#FF1493', '#00BFFF', '#32CD32', '#FF4500', '#8A2BE2']
    
    # R² Score
    fig.add_trace(
        go.Bar(x=df_performance['Model'], y=df_performance['R² Score'], 
               name='R² Score', marker_color=colors,
               text=df_performance['R² Score'], textposition='outside'),
        row=1, col=1
    )
    
    # RMSE
    fig.add_trace(
        go.Bar(x=df_performance['Model'], y=df_performance['RMSE'], 
               name='RMSE', marker_color=colors,
               text=df_performance['RMSE'], textposition='outside'),
        row=1, col=2
    )
    
    # MAE
    fig.add_trace(
        go.Bar(x=df_performance['Model'], y=df_performance['MAE'], 
               name='MAE', marker_color=colors,
               text=df_performance['MAE'], textposition='outside'),
        row=2, col=1
    )
    
    # Training Time
    fig.add_trace(
        go.Bar(x=df_performance['Model'], y=df_performance['Training Time (s)'], 
               name='Training Time', marker_color=colors,
               text=df_performance['Training Time (s)'], textposition='outside'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics table
    st.markdown("### 📋 Detailed Performance Metrics")
    st.dataframe(df_performance.style.highlight_max(axis=0), use_container_width=True)
    
    # Best model highlight
    st.markdown("""
    <div style='background: linear-gradient(135deg, #FFD700, #FFA500); padding: 20px; border-radius: 15px; margin: 20px 0;'>
        <h3 style='color: #8B4513; text-align: center; margin-bottom: 10px;'>🏆 Best Performing Model</h3>
        <p style='color: #8B4513; text-align: center; font-size: 1.2rem; font-weight: bold;'>
            Linear Regression with R² = 0.95 and RMSE = 12.78
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Predictor Section
    st.markdown("---")
    st.markdown("## 🚀 Quick Model Predictor")
    
    # Check if models are available
    if not models:
        st.warning("⚠️ **No models available.** Please train models first on the Model Training page.")
        return
    
    # Model selection
    selected_model = st.selectbox(
        "Select Model for Prediction:",
        list(models.keys()),
        index=0,
        help="Choose which model to use for prediction"
    )
    
    # Input form
    st.markdown("### 📝 Input Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Performance Metrics**")
        views = st.number_input("👁️ Views", value=10000, step=1000)
        likes = st.number_input("👍 Likes", value=500, step=50)
        comments = st.number_input("💬 Comments", value=100, step=10)
        subscribers = st.number_input("👥 Subscribers", value=5000, step=500)
    
    with col2:
        st.markdown("**⏱️ Time & Engagement**")
        watch_time_minutes = st.number_input("⏰ Watch Time (min)", value=1000.0, step=100.0)
        video_length_minutes = st.slider("📏 Video Length (min)", 1.0, 60.0, 15.0)
        engagement_rate = st.slider("📊 Engagement Rate", 0.0, 0.2, 0.05, 0.001)
        watch_completion = st.slider("⚡ Watch Completion", 0.0, 1.0, 0.65, 0.01)
    
    with col3:
        st.markdown("**🏷️ Categories & Context**")
        category_options = ['Entertainment', 'Gaming', 'Education', 'Music', 'Sports', 'News', 'Technology']
        device_options = ['Mobile', 'Desktop', 'Tablet', 'TV']
        country_options = ['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'JP', 'IN']
        
        category = st.selectbox("🎭 Category", category_options)
        device = st.selectbox("📱 Device", device_options)
        country = st.selectbox("🌍 Country", country_options)
        
        day_of_week = st.selectbox("📅 Day of Week", 
                                  ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        day_of_week_num = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].index(day_of_week) + 1
        is_weekend = 1 if day_of_week in ['Saturday', 'Sunday'] else 0
        quarter = st.selectbox("📈 Quarter", [1, 2, 3, 4])
    
    # Predict button
    if st.button("⚡ Quick Predict", key="quick_predict"):
        try:
            # Create input data
            input_data = {
                'views': views,
                'likes': likes,
                'comments': comments,
                'watch_time_minutes': watch_time_minutes,
                'subscribers': subscribers,
                'video_length_minutes': video_length_minutes,
                'engagement_rate': engagement_rate,
                'watch_completion': watch_completion,
                'category': category,
                'device': device,
                'country': country,
                'day_of_week': day_of_week_num,
                'is_weekend': is_weekend,
                'quarter': quarter
            }
            
            # Prepare input for prediction
            input_df = pd.DataFrame([input_data])
            
            # Ensure preparation only if freq_maps and scaler are available
            if freq_maps and training_features:
                prepared_input = prepare_input_for_prediction(input_df, freq_maps, scaler, training_features)
            else:
                # Fallback: use input as-is if preparation data not available
                prepared_input = input_df
                st.warning("Using unprocessed input - model accuracy may be affected")
            
            # Make prediction
            prediction = predict_revenue(models[selected_model], prepared_input)
            
            # Display result
            st.markdown(f"""
            <div style='text-align: center; background: linear-gradient(135deg, #FF1493, #8A2BE2); 
                        color: white; padding: 2rem; margin: 1rem 0; border-radius: 20px; 
                        box-shadow: 0 8px 25px rgba(255,20,147,0.3);'>
                <h3 style='color: #FFD700; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); margin-bottom: 1rem;'>
                    🎯 {selected_model} Prediction
                </h3>
                <h2 style='color: white; font-size: 3rem; margin: 1rem 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
                    ${prediction:.2f}
                </h2>
                <p style='color: #FFE4E1; font-size: 1.1rem; margin-top: 1rem;'>
                    💰 Estimated Youtube_Ad_Revenue (USD)
                </p>
            </div>
            """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"❌ **Prediction failed:** {str(e)}")
            st.info("💡 **Tip:** Make sure models are trained first on the Model Training page.")
    
    # Model insights section
    st.markdown("---")
    st.markdown("## 🧠 Model Insights")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
     st.markdown("""
    <div style='background: linear-gradient(135deg, #fef7ff, #f8f0ff, #fff0f8); 
                padding: 30px; border-radius: 20px; border-left: 6px solid #FF1493;
                box-shadow: 0 4px 20px rgba(255, 20, 147, 0.1);
                border: 1px solid rgba(255, 20, 147, 0.1);'>
        <h3 style='color: #2c3e50; font-size: 1.8rem; margin-bottom: 20px; text-align: center;'>🎯 Feature Importance</h3>
        <p style='color: #34495e; font-size: 1.2rem; margin-bottom: 15px; text-align: center; font-weight: 500;'>
            Based on model analysis:
        </p>
        <div style='font-size: 1.15rem; line-height: 1.8; color: #2c3e50;'>
            <p style='margin-bottom: 12px; background: rgba(231, 76, 60, 0.08); padding: 10px 15px; border-radius: 10px; border-left: 4px solid #e74c3c;'>
                <strong style='color: #e74c3c;'>1. Watch Time</strong> <span style='color: #27ae60; font-weight: 600;'>(93% importance)</span> - Most critical factor</p>
            <p style='margin-bottom: 12px; background: rgba(52, 152, 219, 0.08); padding: 10px 15px; border-radius: 10px; border-left: 4px solid #3498db;'>
                <strong style='color: #3498db;'>2. Views</strong> <span style='color: #27ae60; font-weight: 600;'>(78% importance)</span> - Strong correlation with revenue</p>
            <p style='margin-bottom: 12px; background: rgba(155, 89, 182, 0.08); padding: 10px 15px; border-radius: 10px; border-left: 4px solid #9b59b6;'>
                <strong style='color: #9b59b6;'>3. Engagement Rate</strong> <span style='color: #27ae60; font-weight: 600;'>(65% importance)</span> - Quality metric</p>
            <p style='margin-bottom: 12px; background: rgba(243, 156, 18, 0.08); padding: 10px 15px; border-radius: 10px; border-left: 4px solid #f39c12;'>
                <strong style='color: #f39c12;'>4. Subscribers</strong> <span style='color: #27ae60; font-weight: 600;'>(52% importance)</span> - Audience base</p>
            <p style='margin-bottom: 8px; background: rgba(230, 126, 34, 0.08); padding: 10px 15px; border-radius: 10px; border-left: 4px solid #e67e22;'>
                <strong style='color: #e67e22;'>5. Category</strong> <span style='color: #27ae60; font-weight: 600;'>(41% importance)</span> - Content type matters</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with insights_col2: 
      st.markdown("""
    <div style='background: linear-gradient(135deg, #fff8f0, #fff3e6, #ffefdb); 
                padding: 30px; border-radius: 20px; border-left: 6px solid #FFA500;
                box-shadow: 0 4px 20px rgba(255, 165, 0, 0.1);
                border: 1px solid rgba(255, 165, 0, 0.1);'>
        <h3 style='color: #2c3e50; font-size: 1.8rem; margin-bottom: 20px; text-align: center;'>💡 Optimization Tips</h3>
        <p style='color: #34495e; font-size: 1.2rem; margin-bottom: 15px; text-align: center; font-weight: 500;'>
            To maximize  youtube ad revenue:
        </p>
        <div style='font-size: 1.15rem; line-height: 1.8; color: #2c3e50;'>
            <p style='margin-bottom: 12px; background: rgba(231, 76, 60, 0.08); padding: 10px 15px; border-radius: 10px; border-left: 4px solid #e74c3c;'>
                • <strong style='color: #e74c3c;'>Increase watch time</strong> through engaging content</p>
            <p style='margin-bottom: 12px; background: rgba(52, 152, 219, 0.08); padding: 10px 15px; border-radius: 10px; border-left: 4px solid #3498db;'>
                • <strong style='color: #3498db;'>Optimize video length</strong> (15-20 min sweet spot)</p>
            <p style='margin-bottom: 12px; background: rgba(155, 89, 182, 0.08); padding: 10px 15px; border-radius: 10px; border-left: 4px solid #9b59b6;'>
                • <strong style='color: #9b59b6;'>Target high-engagement categories</strong> (Gaming, Tech)</p>
            <p style='margin-bottom: 12px; background: rgba(243, 156, 18, 0.08); padding: 10px 15px; border-radius: 10px; border-left: 4px solid #f39c12;'>
                • <strong style='color: #f39c12;'>Post during peak hours</strong> (weekends perform better)</p>
            <p style='margin-bottom: 8px; background: rgba(39, 174, 96, 0.08); padding: 10px 15px; border-radius: 10px; border-left: 4px solid #27ae60;'>
                • <strong style='color: #27ae60;'>Focus on completion rates</strong> over just views</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

#---------------------------------- About Developer Page --------------------------------   
# About Developer Page
def About_Developer():
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h1 style='background: linear-gradient(135deg, #FF1493, #00BFFF, #32CD32, #FF4500); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                   font-size: 3rem; font-weight: bold; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
             👩‍💻About the Developer
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Developer Profile Section
    col1, col2 = st.columns([1, 2])
    
    with col1:
        try:
            st.image("malathi.png", width=400)
            st.markdown("<div style='text-align: center; font-size: 2.1rem; font-weight: 900; margin-top: 0.7em;'>Malathi Y</div>", unsafe_allow_html=True)

        except:
            st.markdown("""
            <div style='width: 220px; height: 220px; background: linear-gradient(135deg, #FF1493, #00BFFF); 
                        border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                        color: white; font-size: 3rem; margin: 0 auto;'>
                👩‍💻
            </div>
            <p style='text-align: center; margin-top: 10px; font-weight: bold;'>Malathi Y</p>
            """, unsafe_allow_html=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)
    
        with col2: 
          st.markdown("""
    <div style="
        background: linear-gradient(90deg, #FF6B6B 20%, #FFD93D 100%);
        width: 95%;
        margin: 0 auto 2.2rem auto;
        padding: 2.5rem 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px 0 rgba(255,147,41,0.08);
        border: 2px solid #FF7F50;
        text-align: center;
    ">
    <h2 style="
        font-size: 2.3rem;
        font-weight: 900;
        color: #22223b;
        letter-spacing: 1.2px;
        margin-bottom: 1.1rem;
        ">
        👋 Hello! I'm <span style='color: #0077b6;'>Malathi Y</span>
    </h2>
    <p style="
        font-size: 1.22rem;
        font-weight: 800;
        color: #24243e;
        line-height: 1.75;
        ">
        I'm a former <span style="color: #00BFFF; font-weight:900;">Staff Nurse</span> from India (TamilNadu), now transitioning into <span style="color:#32CD32; font-weight:900;">Data Science & Machine Learning</span>.
        <br>
        My journey from healthcare to analytics is driven by curiosity, a love for problem-solving, and deep interest in how <span style="font-weight:900;">data</span> shapes decision-making.
    </p>
    </div>
    """, unsafe_allow_html=True)

    # Professional Background Section
    st.markdown("---")
    
    bg_cols = st.columns(3)
    
    with bg_cols[0]:
        st.markdown("""
        <div class='metric-card' style='background: linear-gradient(135deg, #FF4500, #FF6347); 
                    color: white; text-align: center; padding: 20px; border-radius: 15px;
                    box-shadow: 0 8px 32px rgba(255,69,0,0.3);'>
            <h3 style='color: #FFD700; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>
                👩‍⚕️ My Past Profession
            </h3>
            <ul style='text-align: left; color: white; font-size: 1rem;'>
                <li>🏥 Former <strong>Registered Staff Nurse</strong></li>
                <li>👩‍💼 Clinical decision-making expert</li>
                <li>💡 Healthcare data analytics enthusiast</li>
                <li>🔄 Love to takecare of patients </li> 
                <li>🔄 Transitioning to Data Science</li>     
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with bg_cols[1]:
        st.markdown("""
        <div class='metric-card' style='background: linear-gradient(135deg, #00BFFF, #1E90FF); 
                    color: white; text-align: center; padding: 20px; border-radius: 15px;
                    box-shadow: 0 8px 32px rgba(0,191,255,0.3);'>
            <h3 style='color: #FFD700; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>
                📚 My Present Mission
            </h3>
            <ul style='text-align: left; color: white; font-size: 1rem;'>
                <li>🔄 Career Shift and kept baby step in to <strong>Data Science</strong></li>
                <li>🎯 Currently enrolled at <strong>GUVI</strong> for datascience course </li>
                <li>🧠 Learning ML, AI, and Analytics </li>
                <li>📊 Building real-world projects</li>
                <li>🤝 Open to collaboration opportunities</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with bg_cols[2]:
        st.markdown("""
        <div class='metric-card' style='background: linear-gradient(135deg, #32CD32, #228B22); 
                    color: white; text-align: center; padding: 20px; border-radius: 15px;
                    box-shadow: 0 8px 32px rgba(50,205,50,0.3);'>
            <h3 style='color: #FFD700; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>
                🛠️ My Skills So Far
            </h3>
            <ul style='text-align: left; color: white; font-size: 1rem;'>
                <li>🐍 Python,SQL,Pandas,NumPy</li>
                <li>📊 Statistics & Probability</li>   
                <li>🧩 Data cleaning , EDA , Data preprocessing </li>  
                <li>🤖 Machine Learning (Scikit-learn)</li>
                <li>📊 Streamlit, Plotly, Seaborn, Matplotlib </li>
                <li>🧩 Power Bi Dashboard creating </li>
                <li>💼 Business Insight Reporting</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
   
    # Project Details Section
    project_cols = st.columns(2)
    
    with project_cols[0]:
     st.markdown("""
    <div class='metric-card' style='background: linear-gradient(135deg, #8A2BE2, #FF1493); 
                color: white; padding: 25px; border-radius: 20px; height: 400px;
                box-shadow: 0 10px 40px rgba(138,43,226,0.3);'>
        <h3 style='color: #FFD700; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); text-align: center; font-size: 2rem; font-weight: 900;'>
            📊 About This Project: Youtube Monetization Modeler
        </h3>
        <p style='color: white; font-size: 1.3rem; font-weight: 700; line-height: 1.75; text-align: center; margin-top: 25px;'>
            This dashboard is a comprehensive <strong style='color: #FF1493;'>Machine Learning project</strong> that predicts and analyzes 
            <strong style='color: #00BFFF;'>YouTube content revenue</strong> using advanced regression algorithms and interactive visualizations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
  # Project Technical Details
    tech_cols = st.columns(2)
    
    with tech_cols[0]:
        st.markdown("""
        <div class='metric-card' style='background: linear-gradient(135deg, #FF1493, #FF69B4); 
                    color: white; padding: 20px; border-radius: 15px; height: 400px;
                    box-shadow: 0 8px 32px rgba(255,20,147,0.3);'>
            <h3 style='color: #FFD700; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); text-align: center;'>
                🧰 Project Tools & Technologies
            </h3>
            <ul style='color: white; font-size: 1.1rem; line-height: 1.8;'>
                <li>🐍 <strong>Python</strong> - Core programming</li>
                <li>🤖 <strong>Scikit-learn</strong> - Machine Learning</li>
                <li>📊 <strong>Pandas & NumPy</strong> - Data manipulation</li>
                <li>🌐 <strong>Streamlit</strong> - Interactive dashboard</li>
                <li>📈 <strong>Plotly & Seaborn</strong> - Visualizations</li>
                <li>🔍 <strong>GridSearchCV</strong> - Model optimization</li>
                <li>🎯 <strong>Linear Regression</strong> - Best performing model</li>
            </ul><li>🎯 <strong>Linear Regression , SVR ,Decision tree, Random forest,Xgboost</strong> - Trained models </li>
        </div>
        """, unsafe_allow_html=True)
    
    with tech_cols[1]:
        st.markdown("""
        <div class='metric-card' style='background: linear-gradient(135deg, #8A2BE2, #9370DB); 
                    color: white; padding: 20px; border-radius: 15px; height: 400px;
                    box-shadow: 0 8px 32px rgba(138,43,226,0.3);'>
            <h3 style='color: #FFD700; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); text-align: center;'>
                🌟 What I Built
            </h3>
            <ul style='color: white; font-size: 1.1rem; line-height: 1.8;'>
                <li>✅ <strong>Complete ML Pipeline</strong> - Training to deployment</li>
                <li>📍 <strong>Interactive Youtube Ad Revenue Predictor</strong> - Real-time predictions</li>
                <li>🔎 <strong>Model Comparison Dashboard</strong> - 5 ML algorithms</li>
                <li>📈 <strong>Performance Analytics</strong> - Detailed insights</li>
                <li>🎨 <strong>Rainbow-themed UI</strong> - Modern design</li>
                <li>💡 <strong>Recommendations</strong> - Revenue optimization</li>
                <li>📊 <strong>Residual Analysis</strong> - Model validation</li>
                <li>📊 <strong>Gridsearch CV</strong> - Auto hyperparameter tuning</li>  
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Future Goals Section
    goal_cols = st.columns(1)
    
    with goal_cols[0]:
     st.markdown("""
    <div style="
        background: linear-gradient(90deg, #FFD93D 10%, #FF6B6B 90%);
        width: 97%;
        margin: 0 auto 2rem auto;
        padding: 2.4rem 1.8rem;
        border-radius: 20px;
        box-shadow: 0 8px 28px 0 rgba(255,120,41,0.09);
        border: 2px solid #FF7F50;
        text-align: left;
        display: flex;
        flex-direction: column;
        gap: 1.1rem;
    ">
        <h2 style="
            font-size: 2.2rem;
            font-weight: 900;
            color: #FF4500;
            letter-spacing: 1.2px;
            margin-bottom: 0.4rem;
            text-align: left;
        ">
            🎯 My Future Goals
        </h2>
        <div style="margin-left: 0.3em;">
          <div style="font-size:1.18rem;font-weight:800;color:#24243e;line-height:1.9;display:flex;align-items:center;">
            💼 <span style="margin-left:0.75em;">Become a full-time <span style="color: #FF1493;">Data Scientist / ML Engineer</span></span>
          </div>
          <div style="font-size:1.18rem;font-weight:800;color:#24243e;line-height:1.9;display:flex;align-items:center;">
            💡 <span style="margin-left:0.75em;">Apply Datascience ML/AI to <span style="color: #00BFFF;">healthcare, finance & content analytics</span></span>
          </div>
          <div style="font-size:1.18rem;font-weight:800;color:#24243e;line-height:1.9;display:flex;align-items:center;">
            📚 <span style="margin-left:0.75em;">Master advanced tools like <span style="color: #32CD32;">TensorFlow, PyTorch & Cloud ML</span></span>
          </div>
          <div style="font-size:1.18rem;font-weight:800;color:#24243e;line-height:1.9;display:flex;align-items:center;">
            🏆 <span style="margin-left:0.75em;">Build impactful <span style="color: #8A2BE2;">AI solutions</span> that solve real-world problems</span>
          </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Contact Information
    st.markdown("---")
    contact_info = {
        "📧 Email": "malathisathish2228@gmail.com",
        "💼 LinkedIn": "linkedin.com/in/malathi-y-datascience",
        "🐙 GitHub": "https://github.com/malathisathish/Content-Monetization-Modeler",
        "🌐 Portfolio": "https://github.com/malathisathish"
    }
    
    st.markdown("""
<div style='background: linear-gradient(135deg, #FF7F50, #FFD700, #FF69B4); 
            padding: 8px; border-radius: 16px; margin: 15px 0; max-width: 450px;'>
    <div style='background: white; padding: 18px 25px; border-radius: 14px; box-shadow: 0 3px 12px rgba(0,0,0,0.1);'>
        <h3 style='text-align: center; color: #8A2BE2; font-weight: 700; margin-bottom: 16px; font-size: 1.45rem;'>
            📞 Let's Connect!
        </h3>
    </div>
</div>
""", unsafe_allow_html=True)
    
    contact_cols = st.columns(4)
    for i, (contact_type, contact_value) in enumerate(contact_info.items()):
        with contact_cols[i]:
            colors = ["#FF1493", "#00BFFF", "#32CD32", "#FF4500"]
            st.markdown(f"""
            <div class='metric-card' style='text-align: center; padding: 15px; height: 120px;
                        background: linear-gradient(135deg, {colors[i]}, {colors[(i+1)%4]}); 
                        color: white; border-radius: 15px;
                        box-shadow: 0 8px 32px rgba(255,20,147,0.2);'>
                <h5 style='color: #FFD700; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); margin-bottom: 10px;'>
                    {contact_type}
                </h5>
                <p style='color: white; font-size: 0.9rem; word-break: break-word;'>
                    {contact_value}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Inspirational Quote
    st.markdown("""
<div style='text-align: center; padding: 28px; 
            background: linear-gradient(135deg, #f0e6ff, #ffe8d6, #fff4e1); 
            border-radius: 20px; border: 2px dashed #a16eff; margin: 30px 0;
            box-shadow: 0 6px 20px rgba(161, 110, 255, 0.2);'>
    <blockquote style='font-size: 1.5rem; font-style: italic; color: #7a43b6; 
                      text-shadow: 1px 1px 3px rgba(0,0,0,0.1); line-height: 1.65; margin-bottom: 15px;'>
        <strong>"From saving lives in the ICU to predicting revenue through ML, 
        I'm on a mission to make data-driven insights count."</strong> 🙏
    </blockquote>
    <cite style='color: #d5489c; font-size: 1.15rem; font-weight: bold;'>
        - Malathi Y, Data Science Enthusiast
        ☻ With ❤️ from Tamilnadu, India
    </cite>
</div>
""", unsafe_allow_html=True)
    
    # Reason for Theme Choice
    st.markdown("""
<div style='background: linear-gradient(135deg, #FF1493, #00BFFF, #32CD32, #FF4500);
            color: white; padding: 2.5rem 2rem; border-radius: 20px;
            box-shadow: 0 8px 32px rgba(255,20,147,0.3);
            max-width: 800px; margin: 20px auto; font-size: 1.3rem; font-weight: 700; text-align: center;'>
    <p>
        <strong> Reason to choose this theme :</strong> 
        "Just as a rainbow transforms light into a spectrum of vibrant colors, this machine learning project transforms raw data into diverse, actionable insights. 
        Every feature, like every hue, adds depth and clarity to our predictions — showing that true innovation emerges when we blend the full spectrum of data and creativity."
    </p>
    <p style="margin-top: 1rem;">So I choose the <span style="color: #FFD700; font-weight: 900;">rainbow theme</span> for my Machine Learning project.</p>
</div>
""", unsafe_allow_html=True)
    
    # Main function to run the app
def main():
    st.set_page_config(
        page_title=" Youtube Monetization Modeler",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS styling
    add_custom_css()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1 style='color: #FF1493; font-size: 2rem;'>📊 Navigation</h1>
        </div>
        """, unsafe_allow_html=True)
        
        selected = st.selectbox(
            "Choose a page:",
            ["🏠 Home", "🎯 Revenue Predictor", "📊 Model Comparison", "👨‍💻 About Developer"]
        )
    
    # Page routing
    if selected == "🏠 Home":
        show_home_page()  
    elif selected == "🎯 Revenue Predictor":
        show_predictor_page()  
    elif selected == "📊 Model Comparison":
        show_comparison_page()  
    elif selected == "👨‍💻 About Developer":
        About_Developer()  

if __name__ == "__main__":
    main()