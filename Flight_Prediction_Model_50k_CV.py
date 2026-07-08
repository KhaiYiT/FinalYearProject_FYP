import pandas as pd

from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Read the balanced_2022 parquet file
df = pd.read_parquet("balanced_2022(50k).parquet")

# Check the shape
print("\nOriginal Dataset shape: ", df.shape)


# =====================================================================================================================================
# Data Preprocessing
# =====================================================================================================================================

print("\nData is Cleaning ......")

# feature selection
target = "ArrDel15"
features = [
    "Origin", "Dest", "CRSDepTime", "CRSElapsedTime", "Distance", 
    "Quarter", "Month", "DayofMonth", "DayOfWeek", "Marketing_Airline_Network", 
    "Operating_Airline", "OriginState", "DestState", "DepTimeBlk", "CRSArrTime", "DistanceGroup" 
]
df_new = df[features + [target]]

# Data Cleaning Part (Remove rows with missing values)
rows_before = df_new.shape[0]
df_new = df_new.dropna()
rows_after = df_new.shape[0]

print("Cleaned Dataset Shape: ", df_new.shape)
print("Rows removed: ", rows_before - rows_after, "rows")
print("Data cleaning completed.\n")

# Define the features and target variable
X = df_new[features]
y = df_new[target].astype(int)

# encoding categorical variables
print("Data is Encoding ......")

X_encoded = pd.get_dummies(X, drop_first=True)
X_scaled = StandardScaler().fit_transform(X_encoded)

print("Dataset Shape After Encoding: ", X_encoded.shape)
print("Data encoding completed.\n")

# Evaluation metrics
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# number of folds
cv_folds_list = [2, 3, 5, 10]

# Store all model results
results = []


# =====================================================================================================================================
# Data Mining
# =====================================================================================================================================

for cv_folds in cv_folds_list:

    print("Number of folds:", cv_folds)
    
    # Logistic Regression (lr)
    print("Training Logistic Regression......")
    
    lr_model = LogisticRegression(max_iter=10000, solver='saga', random_state=42)
    lr_result = cross_validate(lr_model, X_scaled, y, cv=cv_folds, scoring=scoring, n_jobs=1)
    
    # Calculate average result
    lr_accuracy = lr_result["test_accuracy"].mean()
    lr_precision = lr_result["test_precision"].mean()
    lr_recall = lr_result["test_recall"].mean()
    lr_f1 = lr_result["test_f1"].mean()
    lr_auc = lr_result["test_roc_auc"].mean()
    
    # Decision Tree (dt)
    print("Training Decision Tree......")
    
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_result = cross_validate(dt_model, X_encoded, y, cv=cv_folds, scoring=scoring, n_jobs=1)
    
    # Calculate average result
    dt_accuracy = dt_result["test_accuracy"].mean()
    dt_precision = dt_result["test_precision"].mean()
    dt_recall = dt_result["test_recall"].mean()
    dt_f1 = dt_result["test_f1"].mean()
    dt_auc = dt_result["test_roc_auc"].mean()
    
    # Random Forest (rf)
    print("Training Random Forest......\n")
    rf_model = RandomForestClassifier(n_estimators=300, max_depth=30, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=1)
    rf_result = cross_validate(rf_model, X_encoded, y, cv=cv_folds, scoring=scoring, n_jobs=1)
    
    # Calculate average result
    rf_accuracy = rf_result["test_accuracy"].mean()
    rf_precision = rf_result["test_precision"].mean()
    rf_recall = rf_result["test_recall"].mean()
    rf_f1 = rf_result["test_f1"].mean()
    rf_auc = rf_result["test_roc_auc"].mean()

    # Store results
    results.append(["Logistic Regression", cv_folds, lr_accuracy, lr_precision, lr_recall, lr_f1, lr_auc])
    results.append(["Decision Tree", cv_folds, dt_accuracy, dt_precision, dt_recall, dt_f1, dt_auc])
    results.append(["Random Forest", cv_folds, rf_accuracy, rf_precision, rf_recall, rf_f1, rf_auc])

print("Model training completed.\n")


# =====================================================================================================================================
# Results Comparison
# =====================================================================================================================================

results_df = pd.DataFrame(results, columns=["Model", "CV Fold", "Accuracy", "Precision", "Recall", "F1-score", "AUC"])
print(results_df)