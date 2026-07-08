import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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

print("Dataset Shape After Encoding: ", X_encoded.shape)
print("Data encoding completed.\n")

# train-test split sizes
test_size_list = [0.1, 0.2, 0.3, 0.4]

# Store all model results
results = []


# =====================================================================================================================================
# Data Mining
# =====================================================================================================================================

for test_size in test_size_list:

    train_size = 1 - test_size
    split_ratio = str(int(train_size * 100)) + ":" + str(int(test_size * 100))

    print("Train-Test Split:", split_ratio)

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=test_size, random_state=42, stratify=y)

    # Logistic Regression (lr)
    print("Training Logistic Regression......")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr_model = LogisticRegression(max_iter=10000, solver="saga", random_state=42)
    lr_model.fit(X_train_scaled, y_train)

    lr_pred = lr_model.predict(X_test_scaled)
    lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]

    #record results
    lr_accuracy = accuracy_score(y_test, lr_pred)
    lr_precision = precision_score(y_test, lr_pred)
    lr_recall = recall_score(y_test, lr_pred)
    lr_f1 = f1_score(y_test, lr_pred)
    lr_auc = roc_auc_score(y_test, lr_prob)

    # Decision Tree (dt)
    print("Training Decision Tree......")

    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)

    dt_pred = dt_model.predict(X_test)
    dt_prob = dt_model.predict_proba(X_test)[:, 1]

    #record result 
    dt_accuracy = accuracy_score(y_test, dt_pred)
    dt_precision = precision_score(y_test, dt_pred)
    dt_recall = recall_score(y_test, dt_pred)
    dt_f1 = f1_score(y_test, dt_pred)
    dt_auc = roc_auc_score(y_test, dt_prob)

    # Random Forest (rf)
    print("Training Random Forest......\n")

    rf_model = RandomForestClassifier(n_estimators=300, max_depth=30, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=1)
    rf_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]

    #record result
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_precision = precision_score(y_test, rf_pred)
    rf_recall = recall_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf_prob)

    # Store results
    results.append(["Logistic Regression", split_ratio, lr_accuracy, lr_precision, lr_recall, lr_f1, lr_auc])
    results.append(["Decision Tree", split_ratio, dt_accuracy, dt_precision, dt_recall, dt_f1, dt_auc])
    results.append(["Random Forest", split_ratio, rf_accuracy, rf_precision, rf_recall, rf_f1, rf_auc])

print("Model training completed.\n")


# =====================================================================================================================================
# Results Comparison
# =====================================================================================================================================

results_df = pd.DataFrame(results, columns=["Model", "Train-Test Split", "Accuracy", "Precision", "Recall", "F1-score", "AUC"])
print(results_df)