import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df_train = pd.read_csv("LAB_2/smartphone_battery_drain_dataset.csv")

print(df_train.head(5))
print(df_train.columns)
print(df_train.info())
print(df_train.describe())

print(" " * 200 + "\n")

print(f"Количество дубликатов = {df_train.duplicated().sum()}\n")
df = df_train.drop_duplicates()
print(f"Пропущенные знчения \n{df.isnull().sum()}\n")
print(f"Размер датасета = {len(df)}")


df_clean = df.dropna(subset=["CPU_Usage_%", "Battery_Temperature_C"])

stats = df_clean.groupby("Usage_Mode")[["CPU_Usage_%", "Battery_Temperature_C"]].agg(
    ["count", "mean", "median", "std", "min", "max"]
)
print(stats)

df.dropna(subset=["CPU_Usage_%", "Battery_Temperature_C"], inplace=True)
print(f"\nРазмер датасета = {len(df)}")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("Числовые признаки:", numeric_cols)

plt.figure(figsize=(15, 8))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 4, i)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower, upper)


df = pd.get_dummies(df, columns=["Network_Type", "Usage_Mode"], prefix=["Net", "Mode"])

df["Charging_State"] = df["Charging_State"].map({"Charging": 1, "Not Charging": 0})


freq_encoding = df["App_Running"].value_counts().to_dict()
df["App_Running_freq"] = df["App_Running"].map(freq_encoding)
df.drop("App_Running", axis=1, inplace=True)


df["Intensity"] = df["CPU_Usage_%"] * df["RAM_Usage_MB"] / 1000
df["Total_Battery_Drain"] = df["Screen_On_Time_min"] * df["Battery_Drop_Per_Hour"] / 60
temp_threshold = df["Battery_Temperature_C"].median()
df["High_Temperature"] = (df["Battery_Temperature_C"] > temp_threshold).astype(int)


print("\nФинальный размер датасета:", df.shape)
print("\nОписание финальных признаков (первые 5 строк):")
print(df.head())


df.to_csv("LAB_2/smartphone_battery_processed.csv", index=False)
