import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # saves charts to files without opening windows
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 1. Load the cleaned data from Task 2 ----------
df = pd.read_csv("cleaned_data.csv")
os.makedirs("charts", exist_ok=True)
print("Data loaded:", df.shape)

# ---------- 2. Correlation heatmap ----------
# Shows how strongly each pair of features moves together (+1 / -1 = strong)
plt.figure(figsize=(9, 7))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("charts/correlation_heatmap.png", dpi=150)
plt.close()
print("Saved charts/correlation_heatmap.png")

# ---------- 3. Distribution curves ----------
# Histogram + smooth KDE curve shows the shape of each numeric feature
for col in ["age", "salary"]:
    if col in df.columns:
        plt.figure(figsize=(7, 4))
        sns.histplot(df[col], kde=True, color="steelblue")
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(f"charts/distribution_{col}.png", dpi=150)
        plt.close()
        print(f"Saved charts/distribution_{col}.png")

# ---------- 4. Visual correlations (pairplot) ----------
# Scatter plots of every pair of key features
cols = [c for c in ["age", "salary", "purchased"] if c in df.columns]
if len(cols) >= 2:
    sns.pairplot(df[cols], hue="purchased" if "purchased" in cols else None)
    plt.savefig("charts/pairplot.png", dpi=120)
    plt.close()
    print("Saved charts/pairplot.png")

print("\nAll charts saved in the 'charts' folder.")