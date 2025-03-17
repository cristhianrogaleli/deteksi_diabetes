import numpy as np
import pickle
from sklearn.metrics import accuracy_score

# Load Data Testing
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Load Model
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

# 5. Evaluasi Model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Akurasi Model: {accuracy:.2f}")
