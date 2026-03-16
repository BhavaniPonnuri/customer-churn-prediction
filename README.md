# Customer Churn Prediction
#### Customer Churn prediction and Retention Analysis

**Customer Churn Rate is a metric used to measure the percentage of customer discontinuation of a service or product provided within a certain period of time. This is something that companies must pay attention to because the Customer Churn Rate is an obstacle to growth. Aside from inhibiting the growth, loss of customers also means a loss of income (Revenue Churn Rate).**

This project analyzes the customer purchase history to group customers based on their buying behavior and identify who is likely to stop buying. Build a model to predict at-risk customers and estimate how much value each customer brings to the business, helping the company focus retention efforts where they matter most.

This repository contains scripts or source code about how to predict customer churn rate and analysis of retention. RFM method is used here in analysis and segementing the users. 
- Dataset source - https://www.kaggle.com/datasets/vijayuv/onlineretail/data
- This dataset contains all purchases made for an online retail company based in the UK during an eight month period.


### Project Architecture
Raw Transactions (541K rows)
        │
        ▼
   Data Cleaning
   (nulls, cancellations, zero prices, duplicates)
        │
        ▼
 Time-Based Window Split
 ┌──────────────────────────────────┐
 │  Training Window                 │  Prediction Window
 │  Dec 2010 – Aug 2011             │  Sep 2011 – Dec 2011
 │  → RFM Feature Engineering       │  → Churn Label Creation
 └──────────────────────────────────┘
        │                                      │
        ▼                                      ▼
  Customer-Level RFM Table  ←── churn label stamped per customer
        │
        ▼
  Segmentation                    Churn Prediction
  ├── Custom Business Bins         ├── Logistic Regression
  └── GMM Clustering (validation)  ├── Random Forest
                                   ├── XGBoost
                                   └── Gradient Boosting ← Final Model
        │
        ▼
  CLV Estimation + Business Insights

### Tech Stack:
Python, pandas, scikitlearn, xgboost, matplotlib, numpy

### How to Run
1. Clone the repository
git clone https://github.com/BhavaniPonnuri/customer-segmentation-retention.git
cd customer-segmentation-retention
2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
3. Download the dataset
Download OnlineRetail.csv from above Kaggle site and place it in the project root directory.
4. Run the notebook
jupyter notebook online_retail_project.ipynb
Run cells sequentially.

### Key Learnings
1. Target Leakage Detection
Identified and resolved data leakage where recency (a model feature) was directly used to derive the churn label — resulting in artificial 100% model scores. Fixed using temporal train/test splitting.
2. Temporal Validation
Implemented time-based window splitting — the production-standard approach for churn and event-prediction models — separating feature calculation and label creation into non-overlapping time periods.
3. Segmentation Method Selection
Evaluated K-Means (failed due to RFM skewness and L-shaped distribution), log transformation (insufficient improvement), DBSCAN (unsuitable due to density variation and high noise rate), and GMM before settling on custom business-defined bins as primary method — validated by GMM independently.
4. Class Imbalance Assessment
Assessed actual class distribution (67/33) before applying SMOTE — correctly concluded mild imbalance did not warrant synthetic oversampling. Used scale_pos_weight instead.
5. Recall-First Model Selection
Finalised Gradient Boosting over XGBoost based on False Negative minimisation — correctly identifying that in churn prediction, missing a real churner (FN) is more costly than a false alarm (FP).
