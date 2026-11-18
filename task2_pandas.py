import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------- 1. Load data ----------
# If the task gave you a dataset, save it as data.csv in this folder
# and set TARGET to the column you want to predict.
TARGET = "purchased"

try:
    df = pd.read_csv("data.csv")
    print("Loaded data.csv")
except FileNotFoundError:
    # No file given, so build a small messy dataset to practice on
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "age": rng.integers(18, 65, n).astype(float),
        "salary": rng.normal(50000, 12000, n).round(0),
        "city": rng.choice(["Mumbai", "Delhi", "Pune", "Chennai"], n),
        "gender": rng.choice(["Male", "Female"], n),
        "purchased": rng.integers(0, 2, n),
    })
    # Make it messy: missing values and outliers
    df.loc[rng.choice(n, 15, replace=False), "age"] = np.nan
    df.loc[rng.choice(n, 10, replace=False), "city"] = np.nan
    df.loc[rng.choice(n, 5, replace=False), "salary"] = 900000   # outliers
    df.to_csv("data.csv", index=False)
    print("Created sample data.csv")

if TARGET not in df.columns:
    TARGET = df.columns[-1]

# ---------- 2. Inspect ----------
print("\nShape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# ---------- 3. Handle missing values ----------
# Numbers: fill with median (not affected by extreme values)
# Text: fill with the most common value (mode)
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(exclude=np.number).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# ---------- 4. Remove outliers (IQR method) ----------
# Skip the target column so we never delete a class label
num_cols = [c for c in df.select_dtypes(include=np.number).columns if c != TARGET]
before = len(df)
for col in num_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
print(f"\nOutlier rows removed: {before - len(df)}")

# ---------- 5. Encode categorical features ----------
# Models need numbers, so turn text columns into 0/1 columns (one-hot encoding)
cat_cols = df.select_dtypes(exclude=np.number).columns
df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

# ---------- 6. Split into train and test sets ----------
X = df.drop(columns=[TARGET])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nTrain size:", X_train.shape, "| Test size:", X_test.shape)

# ---------- 7. Save results ----------
df.to_csv("cleaned_data.csv", index=False)
X_train.assign(**{TARGET: y_train}).to_csv("train.csv", index=False)
X_test.assign(**{TARGET: y_test}).to_csv("test.csv", index=False)
print("\nSaved: cleaned_data.csv, train.csv, test.csv")
print("\nCleaned data preview:\n", df.head())