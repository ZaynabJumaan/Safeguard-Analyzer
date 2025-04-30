import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the training and testing data
X_url_train = pd.read_csv("dataset/X_url_train.csv")
X_url_test = pd.read_csv("dataset/X_url_test.csv")
y_url_train = pd.read_csv("dataset/y_url_train.csv")
y_url_test = pd.read_csv("dataset/y_url_test.csv")

# Remove NaN values from y_url_train
y_url_train = y_url_train.dropna()
y_url_test = y_url_test.dropna()

# Ensure the same number of rows in X and y
X_url_train = X_url_train.iloc[:len(y_url_train)]
X_url_test = X_url_test.iloc[:len(y_url_test)]

# Convert y labels to 1D format
y_url_train = y_url_train.squeeze()
y_url_test = y_url_test.squeeze()

# Train the Random Forest model
model_url = RandomForestClassifier(n_estimators=200, random_state=7)
model_url.fit(X_url_train, y_url_train)

# Evaluate the model
y_url_pred = model_url.predict(X_url_test)
accuracy = accuracy_score(y_url_test, y_url_pred)

# Save the trained model
joblib.dump(model_url, "models/url_malware_detector.pkl")

print("URL Model Training Completed!")
print(f" Model Accuracy: {accuracy:.2f}")
print(" Model saved as: models/url_malware_detector.pkl")


