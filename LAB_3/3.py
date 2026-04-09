import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =================================================================
# 1, 2. ПОДГОТОВКА И ЭКСПЕРТНЫЙ АНАЛИЗ ДАННЫХ
# =================================================================
path = "LAB_3/smartphone_battery_processed.csv"
df = pd.read_csv(path).drop_duplicates().dropna()

target = "Battery_Drop_Per_Hour"

# Оставляем только физически обоснованные признаки (убрали Total_Battery_Drain)
cols_to_keep = [
    "App_Running",
    "Screen_On_Time_min",
    "CPU_Usage_%",
    "Battery_Temperature_C",
    target,
    "Network_Type",
    "Brightness_Level_%",
    "RAM_Usage_MB",
    "Charging_State",
    "Usage_Mode",
    "Intensity",
    "High_Temperature",
]
df = df[cols_to_keep]

# --- КОРРЕЛЯЦИЯ ---
plt.figure(figsize=(8, 5))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap="coolwarm")
plt.title("Корреляция признаков")
plt.show()

# =================================================================
# 4. ЭКСПЕРИМЕНТ 1: КОДИРОВАНИЕ ДО РАЗДЕЛЕНИЯ
# =================================================================
print("--- Эксперимент 1 (Кодирование до разделения) ---")
df_exp1 = pd.get_dummies(
    df, columns=["App_Running", "Network_Type", "Charging_State", "Usage_Mode"]
)
X1 = df_exp1.drop(columns=[target])
y1 = df_exp1[target]
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X1, y1, test_size=0.2, random_state=42
)

model1 = LinearRegression().fit(X_train1, y_train1)
print(f"R2: {r2_score(y_test1, model1.predict(X_test1)):.4f}")

# =================================================================
# 5. ЭКСПЕРИМЕНТ 2: ПРАВИЛЬНЫЙ PIPELINE (БЕЗ УТЕЧЕК)
# =================================================================
print("\n--- Эксперимент 2 (Кодирование после разделения) ---")
X = df.drop(columns=[target])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

cat_cols = ["App_Running", "Network_Type", "Charging_State", "Usage_Mode"]
# Только те числовые признаки, которые мы реально можем знать ДО разряда батареи
num_cols = [
    "Screen_On_Time_min",
    "Battery_Temperature_C",
    "Intensity",
    "High_Temperature",
]

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X_train[cat_cols])
scaler = StandardScaler().fit(X_train[num_cols])


def transform_data(data):
    cat = pd.DataFrame(
        ohe.transform(data[cat_cols]),
        columns=ohe.get_feature_names_out(),
        index=data.index,
    )
    num = pd.DataFrame(
        scaler.transform(data[num_cols]), columns=num_cols, index=data.index
    )
    return pd.concat([num, cat], axis=1)


X_train_f = transform_data(X_train)
X_test_f = transform_data(X_test)

model = LinearRegression().fit(X_train_f, y_train)
pred = model.predict(X_test_f)

print(f"R2:  {r2_score(y_test, pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, pred):.4f}")

# =================================================================
# 6. ВАЖНОСТЬ ПРИЗНАКОВ
# =================================================================
importance = pd.DataFrame(
    {"Feature": X_train_f.columns, "Weight": model.coef_}
).sort_values(by="Weight", ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(
    x="Weight",
    y="Feature",
    data=importance.head(10),
    hue="Feature",  # Теперь явно привязываем цвет к признаку
    palette="viridis",
    legend=False,  # Отключаем легенду, чтобы она не перекрывала график
)
plt.title("Топ-10 важных признаков")
plt.show()
# =================================================================
# 7. ПРЕДСКАЗАНИЕ ДЛЯ НОВЫХ ДАННЫХ
# =================================================================
new_phone = pd.DataFrame(
    {
        "CPU_Usage_%": [100.0],
        "Brightness_Level_%": [100.0],
        "RAM_Usage_MB": [4500.0],
        "Screen_On_Time_min": [120.0],
        "Battery_Temperature_C": [42.0],
        "App_Running": ["YouTube"],
        "Network_Type": ["5G"],
        "Charging_State": ["Discharging"],
        "Usage_Mode": ["Gaming"],
    }
)

# Вычисляем производные признаки только на основе входных параметров
new_phone["Intensity"] = (new_phone["CPU_Usage_%"] * new_phone["RAM_Usage_MB"]) / 1000
new_phone["High_Temperature"] = (new_phone["Battery_Temperature_C"] > 35).astype(int)

# Трансформируем и предсказываем
res = model.predict(transform_data(new_phone))
print(f"\nПрогноз для нового телефона: {res[0]:.2f} %/час")
