# =====================> Snippet code by Saad Mohamed Saad <=====================

import pandas as pd

# Load the dataset
myDataset = pd.read_csv('lung cancer.csv')

# Display the first few rows of the dataset
print("Before cleaning:")
print(myDataset)

# Check for missing values and handle them
missing_values = myDataset.isnull().sum()
print("\nMissing values before handling:")
print(missing_values)

# Drop rows with missing values
myDataset = myDataset.dropna()

# Check for duplicate rows and remove them
duplicate_rows = myDataset.duplicated().sum()
print("\nNumber of duplicate rows:", duplicate_rows)
myDataset = myDataset.drop_duplicates()

# Display the cleaned dataset
print("\nAfter cleaning:")
print(myDataset.head())

# Save the cleaned dataset to a new CSV file
myDataset.to_csv('Cleaned_lung cancer.csv', index=False)

print("\nDataset cleaning completed!")



# =====================> Snippet code by Wasim Mahmoud <=====================

import pandas as pd
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import LabelEncoder

# Load the data
data = pd.read_csv('lung cancer.csv')

# Separate features and target variable
X = data.drop('LUNG_CANCER', axis=1)
y = data['LUNG_CANCER']

# Encode the "Gender" feature
le = LabelEncoder()
X["GENDER"] = le.fit_transform(X["GENDER"])

# Feature selection using chi-square test
chi2_selector = SelectKBest(chi2, k=10)

# Fit the selector
chi2_selector.fit(X, y)

# Get selected features
selected_features = X.columns[chi2_selector.get_support(indices=True)]

print("Selected features using chi-square test:", selected_features)

# =====================> Snippet code by Nehad Mohamed <=====================

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the data assuming it's in a CSV file named 'lung_cancer.csv'
data = pd.read_csv('lung cancer.csv')

# Separate features and target variable
X = data.drop('LUNG_CANCER', axis=1)  # All columns except 'Lung Cancer'
y = data['LUNG_CANCER']

# Handle categorical features (e.g., Gender) using label encoding
categorical_features = ['GENDER']  # Add other categorical features if needed
le = LabelEncoder()
for feature in categorical_features:
    X[feature] = le.fit_transform(X[feature])

# Create a StandardScaler object
scaler = StandardScaler()

# Fit the scaler on the features (now numerical)
scaler.fit(X)

# Transform the features using the fitted scaler
X_scaled = scaler.transform(X)

# Combine the scaled features and target variable
data_scaled = pd.concat([pd.DataFrame(X_scaled, columns=X.columns), y], axis=1)

# Print a sample of the scaled data
print(data_scaled.head())