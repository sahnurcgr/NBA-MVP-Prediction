# NBA-MVP-Prediction
🏀 NBA MVP Prediction Project
An end-to-end machine learning project that predicts the 2024-2025 NBA MVP winner based on historical player data from the last 20 years.

Developed as part of the Introduction to Data Science course at Ankara University (May 2025).

🚀 Overview
The goal of this project is to model the criteria for the NBA MVP award using historical data and predict the most likely candidate for the current season. The project compares different dataset sizes (10 years vs. 20 years) and evaluates multiple algorithms to achieve the highest predictive accuracy.

🛠 Tech Stack
Language: Python 3.x

Libraries: Pandas, Scikit-learn, Joblib, NumPy

Techniques: Web Scraping, Feature Engineering, Data Scaling, Model Serialization

📊 Methodology
Data Collection: Scraped historical NBA player stats from the last 20 seasons.

Data Preprocessing:

Handled missing values using SimpleImputer.

Standardized features using StandardScaler to ensure consistency.

Model Selection & Experiments:

Tested 3 different models: Linear Regression, Random Forest, and [Denediğin diğer model].

Key Finding: Using 20 years of data instead of 10 years significantly improved the model's ability to capture long-term MVP trends.

Final Model: Random Forest Classifier was chosen for its superior performance in ranking candidates.
