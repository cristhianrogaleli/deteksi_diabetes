import pickle

def save_model(model, filename="diabetes_model.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)

print("Model berhasil disimpan!")
