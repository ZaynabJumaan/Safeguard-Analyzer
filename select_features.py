from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE, SelectFromModel, mutual_info_classif
import pandas as pd
import numpy as np

exe_data = pd.read_csv("dataset/processed_programs.csv")

X_exe = exe_data.drop(columns=["legitimate"])
y_exe = exe_data["legitimate"].astype(int)

estimator_exe = DecisionTreeClassifier()
rfe_exe = RFE(estimator_exe, n_features_to_select=15)
rfe_exe.fit(X_exe, y_exe)
selected_features_exe_rfe = X_exe.columns[rfe_exe.support_]

sfm_exe = SelectFromModel(estimator_exe, threshold="mean")
sfm_exe.fit(X_exe, y_exe)
selected_features_exe_sfm = X_exe.columns[sfm_exe.get_support()]

mi_scores_exe = mutual_info_classif(X_exe, y_exe)
selected_features_exe_mi = X_exe.columns[np.argsort(mi_scores_exe)[-15:]]
print("Selected Features for EXE dataset (SelectFromModel):", selected_features_exe_sfm)
