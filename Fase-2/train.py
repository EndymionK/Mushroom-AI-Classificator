# train.py

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Paths
TRAIN_CSV_PATH = '../train.csv'
MODEL_SAVE_PATH = 'xgb_model.pkl'

# Load training data
df_train = pd.read_csv(TRAIN_CSV_PATH)

# Drop 'id' column and separate features and target
df_train = df_train.drop(columns=['id'])
X = df_train.drop(columns=['class'])
y = df_train['class']

# Encode categorical features
le = LabelEncoder()
y = le.fit_transform(y)

# Split data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define and train the model
model = XGBClassifier(
    alpha=0.1,
    subsample=0.8,
    colsample_bytree=0.6,
    objective='binary:logistic',
    max_depth=14,
    min_child_weight=7,
    gamma=1e-6,
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

# Save the trained model
with open(MODEL_SAVE_PATH, 'wb') as f:
    pickle.dump(model, f)

print(f"Model trained and saved to {MODEL_SAVE_PATH}")
