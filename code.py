# =====================> Snippt code by Saad Mohamed Saad <=====================

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