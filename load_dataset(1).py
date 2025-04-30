import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate  

url_data = pd.read_csv("dataset/urls.csv")
exe_data = pd.read_csv("dataset/programs.csv", delimiter='|')

missing_url = url_data.isnull().sum()
missing_exe = exe_data.isnull().sum()

url_class_distribution = url_data['type'].value_counts()

exe_class_distribution = exe_data['legitimate'].value_counts()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.barplot(x=url_class_distribution.index, y=url_class_distribution.values, palette="viridis")
plt.title("URL Dataset - Class Distribution")
plt.xlabel("Type")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
sns.barplot(x=exe_class_distribution.index, y=exe_class_distribution.values, palette="rocket")
plt.title("Executable Dataset - Class Distribution")
plt.xlabel("Legitimate (1=Benign, 0=Malware)")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

num_features_url = url_data.shape[1] - 1  
num_features_exe = exe_data.shape[1] - 1  

data_summary = [
    ["URLs", url_data.shape[0], url_data.shape[1], missing_url.sum(), url_class_distribution.to_dict(), num_features_url],
    ["Executables", exe_data.shape[0], exe_data.shape[1], missing_exe.sum(), exe_class_distribution.to_dict(), num_features_exe]
]

print(" Dataset Summary:\n")
print(tabulate(data_summary, headers=["Dataset", "Total Rows", "Total Columns", "Missing Values", "Class Distribution", "Number of Features"], tablefmt="fancy_grid"))

print(f"\n Number of features in URL dataset: {num_features_url}")
print(f" Number of features in Executable dataset: {num_features_exe}")
