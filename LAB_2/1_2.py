import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#  1. Первичный анализ данных (EDA)
df_train = pd.read_csv("LAB_2/smartphone_battery_drain_dataset.csv")

with open("LAB_2/DEBUG.txt", "w", encoding="utf-8") as f:

    f.write(" 1. ПЕРВИЧНЫЙ АНАЛИЗ (EDA) \n")
    f.write(f"Исходный размер датасета: {df_train.shape}\n")
    f.write("\n Первые 5 строк:\n")
    f.write(df_train.head(5).to_string() + "\n")

    f.write("\n Столбцы и типы данных:\n")
    df_train.info(buf=f)

    f.write("\n Статистическое описание:\n")
    f.write(df_train.describe().to_string() + "\n\n")

    # 2. Очитска данных и обработка выбросов
    df = df_train.drop_duplicates().copy()
    f.write(f" 2. ОЧИСТКА \n")
    f.write(f"Количество найденных дубликатов: {df_train.duplicated().sum()}\n")

    f.write("\n Пропущенные значения \n")
    f.write(df.isnull().sum().to_string() + "\n")

    # Ответ на вопрос почему я решил удалить пропуски
    df_clean = df.dropna(subset=["CPU_Usage_%", "Battery_Temperature_C"])
    stats = df_clean.groupby("Usage_Mode")[
        ["CPU_Usage_%", "Battery_Temperature_C"]
    ].agg(["count", "mean", "median", "std", "min", "max"])
    f.write("\n Статистика по группам (Usage_Mode) или почему я удалил пропуски\n")
    f.write(stats.to_string() + "\n")

    df.dropna(subset=["CPU_Usage_%", "Battery_Temperature_C"], inplace=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    f.write("\n --- 2 Обработка выбросов --- \n")

    df_before = df.copy()
    df_before["Data_State"] = "Before"

    # 2. Температура
    df["Battery_Temperature_C"] = df["Battery_Temperature_C"].clip(lower=0)

    # Проценты
    for pct_col in ["CPU_Usage_%", "Brightness_Level_%"]:
        df[pct_col] = df[pct_col].clip(0, 100)

    # Физические величины (не могут быть отрицательными)
    df["RAM_Usage_MB"] = df["RAM_Usage_MB"].clip(lower=0)
    df["Screen_On_Time_min"] = df["Screen_On_Time_min"].clip(lower=0)
    df["Battery_Drop_Per_Hour"] = df["Battery_Drop_Per_Hour"].clip(lower=0)

    df_after = df.copy()
    df_after["Data_State"] = "After"

    df_compare = pd.concat([df_before, df_after])

    plt.figure(figsize=(18, 12))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(2, 3, i)
        sns.boxplot(data=df_compare, x="Data_State", y=col, palette="Set2")
        plt.title(f"Comparison: {col}")
        plt.xlabel("")

    plt.tight_layout()
    plt.show()

    # 3 Кодирую признаки
    f.write(" 3. КОДИРОВАНИЕ ПРИЗНАКОВ \n")
    df = pd.get_dummies(
        df, columns=["Network_Type", "Usage_Mode"], prefix=["Net", "Mode"]
    )
    df["Charging_State"] = df["Charging_State"].map({"Charging": 1, "Not Charging": 0})

    freq_encoding = df["App_Running"].value_counts().to_dict()
    df["App_Running_freq"] = df["App_Running"].map(freq_encoding)
    df.drop("App_Running", axis=1, inplace=True)

    # 4 Создание новых столбов
    df["Intensity"] = (df["CPU_Usage_%"] * df["RAM_Usage_MB"]) / 1000
    df["Total_Battery_Drain"] = (df["Screen_On_Time_min"] / 60) * df[
        "Battery_Drop_Per_Hour"
    ]
    temp_threshold = df["Battery_Temperature_C"].median()
    df["High_Temperature"] = (df["Battery_Temperature_C"] > temp_threshold).astype(int)

    f.write("\n4. НОВЫЕ ПРИЗНАКИ \n")
    f.write(f"Созданы признаки: Intensity, Total_Battery_Drain, High_Temperature\n")
    f.write(f"Итоговая размерность: {df.shape}\n")
    f.write(df.head(5).to_string() + "\n")

df.to_csv("LAB_2/smartphone_battery_processed.csv", index=False)

print("READY " * 100)
