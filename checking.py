import pandas as pd
import os

os.makedirs("dataset", exist_ok=True)

exe_data = pd.read_csv("dataset/programs.csv", delimiter='|')

exe_data = exe_data[['Name', 'legitimate']]

exe_data.to_csv("dataset/checking_programs.csv", index=False)

print("Data Preprocessing Completed Successfully!")
