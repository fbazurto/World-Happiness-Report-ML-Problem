# World-Happiness-Report-ML-Problem
# 🎯 Project Overview

In this project I selected the World Happiness Report (WHR) dataset to build and evaluate a binary classification logistic regression model that predicts whether a country-year observation has above‑average positive affect.

# **🧠 Problem Description**

Dataset: WHR2018Chapter2OnlineData.csv (World Happiness Report 2018, assorted socio‑economic and well‑being features)


Label: positive_affect_label — 1 if Positive affect > mean (above average), else 0


Problem Type: Supervised, binary classification


Features:
 Log GDP per capita, Social support, Healthy life expectancy at birth, Freedom to make life choices, Perceptions of corruption, Negative affect
 (plus binary indicators for missingness on those columns)


Value Proposition: Identifying the key socio‑economic drivers of positive affect can inform policy-makers, NGOs, or organizations aiming to improve population well-being.


# **🧪 Data Preparation & Exploration**

Inspected column distributions, missing values, and outliers


Dropped irrelevant or error-prone columns: country, year, GINI or quality metrics


Created missingness indicator flags and performed mean imputation for numerical features


Generated binary label based on whether Positive affect exceeds its mean value


Split data into train/test sets (67% train, 33% test) using train_test_split(random_state=1234)



# **🛠️ Modeling & Evaluation Plan**


1. Baseline Model
Model: Logistic Regression, default C=1.0, max_iter=1000


Metrics: Confusion matrix, precision-recall and ROC curves, AUC


2. Hyperparameter Tuning
GridSearchCV over C = [1e-5, 1e-4, ..., 1e4], 5-fold cross-validation


Identify best C; typically C=1 or C=10


3. Final Model

   
Train logistic regression with C=best_C


Evaluate on test set using same metrics


Compare default vs tuned performance


4. Feature Selection

   
Use SelectKBest(f_classif, k=2) to identify top predictors


Retrain and evaluate model on selected features


Observe AUC performance (e.g., ~0.80)



# 🎨 Visualizations & Insights


Precision–Recall Curve: Default vs tuned model (green vs red); compare threshold trade-offs


ROC Curve & AUC: Visual comparison and numeric AUC values for both models


Feature Importance: Highlight which features (e.g. Freedom to choose, Life Ladder) most influence predictions


# 🛠️ Usage Instructions


Prerequisites
Install dependencies (versions indicative):
bash
CopyEdit
pip install pandas numpy scikit-learn matplotlib seaborn

To Run Analysis
Clone the repo


Place WHR2018Chapter2OnlineData.csv under data/


Launch and run the Jupyter notebook notebooks/Lab8_DefineAndSolve.ipynb


This performs:


data loading & cleaning


EDA, feature engineering


train/test split


baseline and tuned models


evaluation plots and metrics


feature selection




# ✅ Summary & Findings


Logistic Regression using key features can effectively classify above-average positive affect (AUC ≈ 0.80)


Tuning C provided marginal improvements


Feature selection affirmed importance of Freedom to make life choices and Life Ladder


Precision-recall and ROC curves show models behave similarly across thresholds



# 📚 References & Acknowledgements


WHR dataset (2018)


scikit-learn: LogisticRegression, GridSearchCV, SelectKBest


Visualization: seaborn, matplotlib


Inspiration from README templates and best practices: catiaspsilva's README template saiprakashspr.medium.com+5github.com+5medium.datadriveninvestor.com+5, freecodecamp README guidance freecodecamp.org

# 📄 License & Contribution


Feel free to explore, modify, or extend it. Pull requests or suggestions welcome!



