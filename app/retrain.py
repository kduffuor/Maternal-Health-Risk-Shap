import pickle
import warnings
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

df = pd.read_csv('data/data.csv')
X = df[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']]
y = df['RiskLevel'].map({'high risk': 0, 'low risk': 1, 'mid risk': 2})

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

pickle.dump(model, open('model.bin', 'wb'))
print('Done:', type(model))