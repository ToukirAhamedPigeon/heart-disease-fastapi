# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv("data/heart.csv") 

print(df.columns.tolist())

print(df.head())

X = df.drop("target", axis=1)
y = df["target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
import os
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/heart_model.joblib")
print("Model saved at model/heart_model.joblib")
