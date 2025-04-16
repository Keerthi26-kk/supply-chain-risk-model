Real-Time Supply Chain Disruption Detection using Machine Learning

✔︎Overview

This project presents a machine learning-based approach for predicting potential supply chain disruptions using historical and operational data. It leverages the power of XGBoost for classification, integrates data cleaning, feature engineering, and model evaluation, and sets the foundation for real-time decision-making in logistics.

✔︎ Problem Statement

Supply chain disruptions can lead to significant financial and operational losses. This project aims to:

● Detect early signs of disruption based on key supply chain indicators.
● Train a binary classification model (disruption: 0 = No Disruption, 1 = Disruption).
● Use real-time or batch data to continuously assess risk levels.

✔︎ Dataset

The dataset includes features such as:

➢ Availability — stock availability percentage.
➢ Stock levels — current stock count.
➢ Lead times — time taken from order to delivery.
➢ Number of products sold — transactional volume.
➢ Revenue generated — financial indicator.
➢ One-hot encoded logistics-related fields like Transportation modes and Routes.

✔︎ Target variable:

disruption — binary label generated based on operational thresholds and domain logic.

✔︎ Technologies Used

∙ Python 3.12
∙ Pandas, NumPy – Data manipulation
∙ scikit-learn – Preprocessing, training, evaluation
∙ XGBoost – Classification model
∙ Matplotlib, Seaborn – Visualization

✔︎ Data Preprocessing

Key preprocessing steps:

Handled missing values and verified feature data types.
Converted all categorical features using One-Hot Encoding.
Scaled features using StandardScaler.
Ensured binary classification feasibility by validating class balance.

Disruption condition logic:

df['disruption'] = (
    (df['Availability'] < 20) &
    (df['Stock levels'] < 20) &
    (df['Lead times'] > df['Lead times'].median())
).astype(int)
To ensure model training stability, additional synthetic disruption=1 samples were appended to the dataset if class imbalance was detected.

✔︎ Model Training

We used XGBoostClassifier with GridSearchCV for hyperparameter tuning.

Hyperparameters:

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

✔︎ Model Training Pipeline:

Split data into 80% training and 20% testing sets.
Trained XGBoost with cross-validation.
Selected best model based on accuracy.

✔︎ Evaluation Metrics

Confusion Matrix
Classification Report (Precision, Recall, F1-score)
ROC AUC Score
Feature Importance Visualization
In scenarios where the target class (disruption=1) was missing, the ROC AUC and feature importance were not computable — additional steps were taken to balance the dataset.

✔︎ Key Results

Model Accuracy: 100% on synthetic-balanced data
Important Features: Availability, Stock levels, Lead times
ROC AUC: Interpreted when both classes are present in test data.

✔︎ Future Enhancements

Integrate real-time streaming data pipelines using Kafka or Spark Streaming.
Deploy model using FastAPI or Flask for prediction APIs.
Add model monitoring (drift detection, explainability) via MLflow or EvidentlyAI.
Handle dynamic retraining in response to concept drift.

✔︎ Folder Structure

📦 SupplyChainDisruptionDetection
 ┣ 📜 data.csv
 ┣ 📜 model_training.ipynb
 ┣ 📜 README.md
 ┗ 📜 requirements.txt
🔗 How to Run

Clone the repo
Install dependencies:
pip install -r requirements.txt
Run the Jupyter notebook:
jupyter notebook model_training.ipynb

✔︎ Acknowledgements

This project was developed as part of a supply chain analytics initiative to enable proactive disruption management using machine learning techniques.
