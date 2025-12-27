# Student Dropout Prediction (XGBoost Classifier)

This project contains the complete code for training a Machine Learning model to predict student dropout rates. The solution utilizes XGBoost, a powerful gradient boosting framework, to classify students based on various academic and demographic features.

# 🚀 Key Features

The pipeline is engineered for accuracy and efficiency within a Kaggle environment, leveraging robust preprocessing and state-of-the-art tree-based algorithms.

* Gradient Boosting Machine: Utilizes the XGBClassifier for high-performance classification, capable of capturing complex non-linear relationships in student data.

* Standardized Preprocessing: Implements StandardScaler to normalize feature distributions, ensuring optimal convergence for the model.

* Robust Evaluation: Includes comprehensive model assessment using precision, recall, and f1-score metrics via classification_report.


# 📈 Methodology

The core of this solution is a supervised classification pipeline that maps student attributes to a binary target variable (dropout vs. non-dropout).

## 1. Data Preprocessing

To prepare the raw CSV data for the model, the following transformation steps are applied:

Feature Selection: Irrelevant columns such as unique identifiers (id) and target labels are separated from the feature set.

Train-Validation Split: The training data is split (75% Train / 25% Validation) to create a hold-out set for unbiased performance evaluation.

Feature Scaling: All numerical features are transformed using StandardScaler (z= 
σ
x−μ
​	
 ), ensuring that features with larger ranges do not dominate the learning process.
 

## 2. Model Architecture (XGBoost)

The predictive model is built on the XGBoost (Extreme Gradient Boosting) framework:

Algorithm: A scalable implementation of gradient boosted decision trees.

Optimization: It iteratively builds new trees to correct the errors of previous trees, optimizing a differentiable loss function.

Configuration: The model is initialized with a fixed random_state=42 to ensure reproducible results across different runs.

## 3. Training & Inference

Training: The model is fitted on the scaled training features (X_train) and corresponding labels (y_train).

Validation: Performance is verified on the validation set (X_val) using standard classification metrics.

Inference: The trained model predicts labels for the unseen test dataset, which is also scaled using the fitted scaler parameters.

Output Generation: Predictions are mapped back to their corresponding IDs and saved as submission.csv.

# 🛠️ Tech Stack
Core: Python 3

Machine Learning: XGBoost, Scikit-learn

Data Handling: Pandas, NumPy

Evaluation: Scikit-learn Metrics (Classification Report)

# 🏃 Running the Project

## 1. Dependencies

This script is designed to run in a Kaggle Notebook or a standard Python environment with the following libraries installed:

Bash
pip install pandas numpy scikit-learn xgboost
## 2. Dataset

This model was trained on the Student Dropout Prediction dataset. The data consists of:

train.csv: Training features and labels.

test.csv: Unseen test data for submission generation.

The code expects these files to be located in the /kaggle/input/students-drop-out-prediction/ directory.

## 3. Notebook Review

The provided code serves as an end-to-end pipeline:

Data Loading: Reads training and testing CSV files into Pandas DataFrames.

Preprocessing: Splits data and applies standard scaling.

Modeling: Trains the XGBoost Classifier.

Evaluation: Prints a detailed classification report.

Submission: Generates the final submission.csv file for the leaderboard.
