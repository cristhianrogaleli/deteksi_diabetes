import numpy as np
from sklearn.naive_bayes import GaussianNB

# Load Data Training
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

# 4. Membuat Model Naïve Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Simpan model
import pickle
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)
