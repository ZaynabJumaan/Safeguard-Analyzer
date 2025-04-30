import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

os.makedirs("dataset", exist_ok=True)

url_data = pd.read_csv("dataset/urls.csv")
exe_data = pd.read_csv("dataset/programs.csv", delimiter='|')

# ------ STEP 1: PROCESS URL DATA ------
url_data['type'] = url_data['type'].map({'benign': 0, 'phishing': 1, 'defacement': 2})

url_data['url_length'] = url_data['url'].apply(len)
url_data['num_dots'] = url_data['url'].apply(lambda x: x.count('.'))
url_data['num_slashes'] = url_data['url'].apply(lambda x: x.count('/'))
url_data['has_ip'] = url_data['url'].apply(lambda x: 1 if any(part.isdigit() for part in x.split('.')) else 0)

url_data = url_data.drop(columns=['url'])

# ------ STEP 2: PROCESS EXECUTABLE DATA ------
exe_data = exe_data.drop(columns=['Name', 'md5'], errors='ignore')

scaler = StandardScaler()
exe_features = exe_data.drop(columns=['legitimate'])
exe_data[exe_features.columns] = scaler.fit_transform(exe_features)

try:
    joblib.dump(scaler, "dataset/scaler.pkl")
    print("Scaler saved as: dataset/scaler.pkl")
except Exception as e:
    print(f"Error saving scaler: {e}")
    exit()

# ------ STEP 3: SAVE THE PREPROCESSED DATA ------
url_data.to_csv("dataset/processed_urls.csv", index=False)
exe_data.to_csv("dataset/processed_programs.csv", index=False)

print("Data Preprocessing Completed Successfully!")
print("Processed URL dataset saved as: dataset/processed_urls.csv")
print("Processed Executable dataset saved as: dataset/processed_programs.csv")