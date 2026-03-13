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

    # Ответ на вопрос почему я решил удалить выбросы
    df_clean = df.dropna(subset=["CPU_Usage_%", "Battery_Temperature_C"])
    stats = df_clean.groupby("Usage_Mode")[["CPU_Usage_%", "Battery_Temperature_C"]].agg(
        ["count", "mean", "median", "std", "min", "max"]
    )
    f.write("\n Статистика по группам (Usage_Mode) или почему я удалил пропуски\n")
    f.write(stats.to_string() + "\n")

    df.dropna(subset=["CPU_Usage_%", "Battery_Temperature_C"], inplace=True)


    # Обработка выбросов(Ящик с усами)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    f.write("\n ОБРАБОТКА ВЫБРОСОВ\n")
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
        
        # Кол-во выбросов
        outliers_count = ((df[col] < lower) | (df[col] > upper)).sum()
        f.write(f"Признак '{col}': обработано выбросов -> {outliers_count}\n")
        
        df[col] = df[col].clip(lower, upper)

    # 3 обрабатываю категориальные выбросы
    f.write(" 3. КОДИРОВАНИЕ ПРИЗНАКОВ \n")
    df = pd.get_dummies(df, columns=["Network_Type", "Usage_Mode"], prefix=["Net", "Mode"])
    df["Charging_State"] = df["Charging_State"].map({"Charging": 1, "Not Charging": 0})
    
    freq_encoding = df["App_Running"].value_counts().to_dict()
    df["App_Running_freq"] = df["App_Running"].map(freq_encoding)
    df.drop("App_Running", axis=1, inplace=True)

    # 4 Создание новых столбов
    df["Intensity"] = (df["CPU_Usage_%"] * df["RAM_Usage_MB"]) / 1000
    df["Total_Battery_Drain"] = (df["Screen_On_Time_min"] / 60) * df["Battery_Drop_Per_Hour"]
    temp_threshold = df["Battery_Temperature_C"].median()
    df["High_Temperature"] = (df["Battery_Temperature_C"] > temp_threshold).astype(int)

    f.write("\n4. НОВЫЕ ПРИЗНАКИ \n")
    f.write(f"Созданы признаки: Intensity, Total_Battery_Drain, High_Temperature\n")
    f.write(f"Итоговая размерность: {df.shape}\n")

df.to_csv("LAB_2/smartphone_battery_processed.csv", index=False)

print("READY " * 100)