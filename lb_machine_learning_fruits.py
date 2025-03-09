import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Binarizer, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from pandas.plotting import scatter_matrix
from sklearn.metrics import accuracy_score

try:
    file_data = pd.read_csv("datasets/fruit_prices_2020.csv")
except Exception as e:
    print(f"Error reading the file: {e}")
    exit(1)

print(
    f"\nHead first rows: \n{file_data.head()}",
    f"\nInformation about dataset: \n{file_data.info()}",
    f"\nDescribe statistics: \n{file_data.describe()}"
)

file_data = file_data.dropna()

scaler = MinMaxScaler()
standard_scaler = StandardScaler()
normalizer = Normalizer()
binarizer = Binarizer(threshold=1.0)

# choose only numeric columns from table
num_cols = file_data.select_dtypes(include=[np.number]).columns
print("\nNumeric columns:", list(num_cols))

# scaler data
scaled = scaler.fit_transform(file_data[num_cols])

# standard data
standardized = standard_scaler.fit_transform(file_data[num_cols])

# normalized data
normalized = Normalizer().fit_transform(file_data[num_cols])

# binarized data
binarized = binarizer.fit_transform(file_data[num_cols])

plt.figure(figsize=(12,6))
sns.heatmap(file_data[num_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Теплова карта кореляційної матриці")
plt.show()

scatter_matrix(file_data[num_cols], figsize=(12, 10), diagonal="kde")
plt.suptitle("Матриця розсіювання")
plt.show()

file_data['Category'] = pd.cut(
    file_data['RetailPrice'],
    bins=[0, 1.5, 3, file_data['RetailPrice'].max()],
    labels=['Cheap', 'Moderate', 'Expensive']
)

if 'Category' in file_data.columns:
    X = file_data.drop("Category", axis=1).select_dtypes(include=[np.number])
    y = file_data["Category"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_logistic_regression = LogisticRegression(max_iter=1000)
    model_logistic_regression.fit(X_train, y_train)

    y_predicted_lr = model_logistic_regression.predict(X_test)

    print("\nLogistic Regression Model:confusion_matrix ", confusion_matrix(y_test, y_predicted_lr))
    print("\nLogistic Regression Model:classification_report ", classification_report(y_test, y_predicted_lr))

    random_forest_model = RandomForestClassifier()
    random_forest_model.fit(X_train, y_train)

    y_predicted = random_forest_model.predict(X_test)

    print("\nAccuracy: ", accuracy_score(y_test, y_predicted))
else:
    print("\nDataset does not have 'Category', classification cannot be done.")
