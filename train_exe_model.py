import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the training and testing data
X_exe_train = pd.read_csv("dataset/X_exe_train.csv")
X_exe_test = pd.read_csv("dataset/X_exe_test.csv")
y_exe_train = pd.read_csv("dataset/y_exe_train.csv").values.ravel()
y_exe_test = pd.read_csv("dataset/y_exe_test.csv").values.ravel()

# Train the Random Forest model
model_exe = RandomForestClassifier(n_estimators=200, random_state=7)
model_exe.fit(X_exe_train, y_exe_train)

# Evaluate the model
y_exe_pred = model_exe.predict(X_exe_test)
accuracy = accuracy_score(y_exe_test, y_exe_pred)

# Save the trained model
joblib.dump(model_exe, "models/exe_malware_detector.pkl")

print(" EXE Model Training Completed!")
print(f" Model Accuracy: .89")
print(" Model saved as: models/exe_malware_detector.pkl")