import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

# 1. Load Dataset
data = pd.read_csv("diabetes.csv")

# 2. Standarisasi Data
X = data.drop(columns=['Outcome'])
y = data['Outcome']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Memisahkan Data Training dan Data Testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Membuat Model Naïve Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# 5. Simpan Model dan Scaler dengan Pickle
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)  # Simpan model Naïve Bayes

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)  # Simpan objek StandardScaler

print("✅ Model dan Scaler telah berhasil disimpan sebagai 'diabetes_model.pkl' dan 'scaler.pkl'.")
