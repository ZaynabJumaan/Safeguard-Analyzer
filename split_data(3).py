import pandas as pd
from sklearn.model_selection import train_test_split

# Load preprocessed datasets
url_data = pd.read_csv("dataset/processed_urls.csv")
exe_data = pd.read_csv("dataset/processed_programs.csv")

# ------ STEP 1: SPLIT URL DATA ------
X_url = url_data.drop(columns=['type'])  # Features
y_url = url_data['type']  # Target labels

# Split into 80% training and 20% testing
X_url_train, X_url_test, y_url_train, y_url_test = train_test_split(X_url, y_url, test_size=0.2, random_state=42)

# Save split data
X_url_train.to_csv("dataset/X_url_train.csv", index=False)
X_url_test.to_csv("dataset/X_url_test.csv", index=False)
y_url_train.to_csv("dataset/y_url_train.csv", index=False)
y_url_test.to_csv("dataset/y_url_test.csv", index=False)

# ------ STEP 2: SPLIT EXECUTABLE DATA ------
X_exe = exe_data.drop(columns=['legitimate'])  # Features
y_exe = exe_data['legitimate']  # Target labels

# Split into 80% training and 20% testing
X_exe_train, X_exe_test, y_exe_train, y_exe_test = train_test_split(X_exe, y_exe, test_size=0.2, random_state=42)

# Save split data
X_exe_train.to_csv("dataset/X_exe_train.csv", index=False)
X_exe_test.to_csv("dataset/X_exe_test.csv", index=False)
y_exe_train.to_csv("dataset/y_exe_train.csv", index=False)
y_exe_test.to_csv("dataset/y_exe_test.csv", index=False)

print(" Data Splitting Completed Successfully!")
print(" Training and Testing datasets saved in 'dataset' folder.")
