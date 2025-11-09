import pandas as pd

# Load the dataset from the data folder
try:
    df = pd.read_csv('data/census.csv')
except FileNotFoundError:
    print("Error: 'data/census.csv' not found. Make sure you are in the correct directory.")
    exit()

# --- Initial Inspection ---
print("--- 1. First 5 Rows of the Dataset ---")
print(df.head())
print("\n" + "="*50 + "\n")

# --- Check for inconsistencies in column names (like leading spaces) ---
print("--- 2. Original Column Names ---")
print(df.columns)
# Clean the column names by removing leading/trailing whitespace
df.columns = df.columns.str.strip()
print("\n--- Cleaned Column Names ---")
print(df.columns)
print("\n" + "="*50 + "\n")


# --- Get a High-Level Summary ---
print("--- 3. Data Types and Non-Null Counts ---")
# .info() is great for seeing the data type of each column and if there are missing values.
df.info()
print("\n" + "="*50 + "\n")

# --- Summary Statistics for Numerical Data ---
print("--- 4. Descriptive Statistics for Numerical Columns ---")
# .describe() gives you the count, mean, standard deviation, min, max, etc.
print(df.describe())
print("\n" + "="*50 + "\n")

# --- Understand the Target Variable ---
print("--- 5. Distribution of the 'salary' Column (Our Target) ---")
# This tells us how many people are in each income bracket. It's a classification problem.
print(df['salary'].value_counts())
print("\n" + "="*50 + "\n")

# --- Identify Missing Values ---
print("--- 6. Checking for '?' as Missing Values ---")
# In this dataset, missing values are marked with ' ?'. Let's count them.
for col in df.columns:
    # We convert the column to string type to safely check for the '?' character
    missing_count = (df[col].astype(str).str.strip() == '?').sum()
    if missing_count > 0:
        print(f"-> Column '{col}' has {missing_count} missing values.")