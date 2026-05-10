import pickle
import warnings
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

# Find data.csv regardless of where the script is run from
base_dir = os.path.dirname(os.path.abspath(__file__))
possible_paths = [
    os.path.join(base_dir, 'data', 'data.csv'),
    os.path.join(base_dir, '..', 'data', 'data.csv'),
    'data/data.csv',
    '../data/data.csv'
]

df = None
for path in possible_paths:
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"Loaded data from: {path}")
        break

if df is None:
    raise FileNotFoundError("data.csv not found in any expected location")

X = df[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']]
y = df['RiskLevel'].map({'high risk': 0, 'low risk': 1, 'mid risk': 2})

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

model_path = os.path.join(base_dir, 'model.bin')
pickle.dump(model, open(model_path, 'wb'))
print(f"Model saved to: {model_path}")
print(f"Done: {type(model)}")