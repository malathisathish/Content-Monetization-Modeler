## 🎬 Content Monetization Modeler

An interactive Streamlit-based predictive dashboard that estimates YouTube Ad Revenue for individual videos based on video performance, user engagement, and contextual metadata.
This project is designed to empower content creators, media companies, analysts, and marketers with data-driven insights into digital content monetization.

### 🧠 Problem Statement

 📝 Note : In my point of view Clearly understanding the problem statement is the foundation for why this project exists. Since this is an analytical + predictive project aimed at solving real-world challenges faced by YouTube creators and businesses, a well-defined problem statement becomes the key to building meaningful insights.

I have divided the problem into two structured sections:

#### ⚠️ The Problem – Highlighting the gaps and challenges in predicting YouTube revenue.

#### 🎯💡 The Aim (as the Solution) – How this project addresses those gaps using ML + visualization.

This approach makes it easier to see the logic within the problem, as the solution lies within the problem itself.

#### 🔴 SECTION A: Problem – The Need to Decode YouTube Revenue Patterns

With billions of users and millions of content creators, YouTube has become one of the world’s largest digital platforms. However, predicting how much ad revenue a single video will generate remains a challenge.

##### 📌 Key Challenges:

Revenue Variability Across Content Categories

Different niches (Gaming, Education, Entertainment, etc.) yield different CPM (Cost Per Mille) rates.

Example: An educational video may generate higher ad revenue than a comedy skit with the same views.

Impact of Engagement Metrics

Likes, comments, and watch time play a critical role in algorithm-driven ad placement.

Example: A video with high watch completion % gets better ad exposure.

Device & Geographic Factors

Viewer devices (Mobile, TV, Desktop) and countries impact ad targeting and CPM.

Example: Ads shown in the US have higher revenue than in some developing regions.

Time-Based Trends

Seasonality, weekdays vs weekends, and quarterly patterns affect traffic & ad rates.

Example: Holiday season often boosts ad revenue due to advertiser demand.

Uncertainty in Prediction

Creators struggle with business planning because revenue can swing widely even with similar video stats.

##### ✅ Outcome of the Problem:

Creators lack a reliable predictive tool to estimate earnings.

Businesses cannot plan campaigns without understanding expected ROI.

#### 🟢 SECTION B: Aim – A Smart Dashboard to Predict Ad Revenue

This project aims to build, train, and deploy ML models (Linear Regression baseline + Random Forest/XGBoost extensions) to predict YouTube ad revenue and visualize insights in an interactive Streamlit dashboard.

##### 📌 Core Objectives:

Build a Linear Regression model to estimate ad revenue (baseline model).

Engineer and encode features (views, watch time, subscribers, engagement rate, etc.).

Perform residual analysis to evaluate model fit.

Compare models with Random Forest, SVR, and XGBoost.

Deploy an interactive Streamlit app for creators and businesses to forecast video revenue.

##### 💡 End Goal:
A smart dashboard where users can upload video data → process features → train models → get predicted ad revenue → analyze results interactively.

### 📌 Project Overview

With this dashboard, you can answer:

🏆 Which category of videos generates the most ad revenue?
📲 Does device usage (TV, Mobile, Desktop) affect ad monetization?
🌍 How does country/region influence ad rates?
📈 How much can a new video expect to earn based on its performance metrics?

It connects raw CSV data → preprocessing → ML models → visualization in Streamlit.

### 🎯 Objectives

♦ To predict YouTube ad revenue (USD) using statistical & ML models.

♦ To perform feature engineering on engagement, video length, category, and country.

♦ To train and evaluate Linear Regression, Random Forest, XGBoost.

♦ To visualize residuals, error distributions, and performance metrics.

♦ To build a Streamlit dashboard for interactive exploration.

### 🗂️ Dataset Details

The dataset is derived from YouTube video performance logs.

📊 Features Used:

Numerical Features → views, likes, comments, watch_time_minutes, subscribers, engagement_rate, watch_completion, video_length_minutes, day_of_week, is_weekend, quarter.

Categorical Features → category, device, country.

Target Variable → ad_revenue_usd.

✅ After preprocessing → Encoding (Frequency / Target Encoding) + Scaling → ready for ML models.

### 📁 Folder Structure

📦 content-monetization-modeler
┣ 📁 data
┃ ┣ cleaned_content_monetization_data.csv
┃ ┗ raw_data.csv
┣ 📁 models
┃ ┣ linear_regression_model.pkl
┃ ┣ random_forest_model.pkl
┃ ┗ xgboost_model.pkl
┃ ┗ svr_model.pkl
┃ ┗ decision_tree_model.pkl
┣ 📁 streamlit_app
┃ ┗ app.py
┣ 📄 requirements.txt
┣ 📄 README.md
┣ content_monetization_ipynb

### 📊 Analysis Scenarios & Insights

Video Category: Educational content shows higher CPM → higher ad revenue.

Device Impact: TV & Desktop users yield higher ad rates than Mobile.

Geographic Impact: US, UK, and CA drive higher CPM than India.

Engagement Metrics: Higher watch completion and engagement rate strongly correlate with revenue.

Seasonality: Q4 (holiday season) consistently drives higher revenue due to ad demand.

### 🛠️ Tech Stack

Frontend: Streamlit (interactive dashboard), Plotly (visualizations)

Backend: Python (feature engineering, ML training), Pandas, NumPy

ML Models: Scikit-learn (Linear Regression, Random Forest, SVR), XGBoost

Visualization: Seaborn, Matplotlib, Plotly

Environment: Visual Studio Code / Jupyter Notebook

### ▶️ How to Run the Content Monetization Modeler

1️⃣ Load csv file 

2️⃣ (Optional) Create a Virtual Environment

python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # macOS/Linux


3️⃣ Install Required Dependencies

pip install -r requirements.txt


4️⃣ Run Model Training

python train_models.py


5️⃣ Launch the Streamlit Dashboard

streamlit run streamlit_app/dashboard.py


6️⃣ Open your browser →
http://localhost:8501

### 🔮 Future Enhancements

Deploy as a cloud-based web app (Heroku/AWS).

Add real-time YouTube API integration to fetch live video stats.

Introduce deep learning models (LSTMs/Transformers) for advanced prediction.

Generate automated recommendation reports for creators.

### 🙏 Acknowledgments

YouTube Analytics API – for video performance insights.

Scikit-learn & XGBoost – for machine learning models.

Streamlit – for dashboard development.

Pandas & NumPy – for data wrangling.

Guvi mentors for supporting me during project.

### ✅ Conclusion

This project bridges the gap between video analytics and predictive monetization by combining ML models, feature engineering, and dashboards.

It empowers creators and businesses to:

Plan content strategies based on revenue potential.

Compare video performance across categories, devices, and geographies.

Make data-driven monetization decisions.

✨ “Data turns creativity into strategy — helping creators monetize smarter.”

### 👩‍💻 Author
Malathi.y | Data Science Enthusiast 🎓

💬 Feedback? Contributions? Questions? Let’s connect!
📧 Email: malathisathish2228@gmail.com

💻 GitHub: "https://github.com/malathisathish"

