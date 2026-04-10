import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

path = "LAB_3/smartphone_battery_processed.csv"
df = pd.read_csv(path)

target = "Battery_Drop_Per_Hour"

# Оставляем физически значимые признаки
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

plt.figure(figsize=(8, 5))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap="coolwarm")
plt.title("Корреляция признаков")
plt.show()

# Разделение на признаки и таргет
X = df.drop(columns=[target])
y = df[target]

# Разделение на Train/Test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Списки столбцов для обработки
cat_cols = ["App_Running", "Network_Type", "Charging_State", "Usage_Mode"]
num_cols = [
    "Screen_On_Time_min",
    "Battery_Temperature_C",
    "Intensity",
    "High_Temperature",
]

# Обучение препроцессоров
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X_train[cat_cols])
scaler = StandardScaler().fit(X_train[num_cols])


# Функция трансформации
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

# 2. ОБУЧЕНИЕ МОДЕЛЕЙ (ЛАБОРАТОРНАЯ 4)
results = []


# Функция для записи результатов
def log_result(name, params, model):
    pred = model.predict(X_test_f)
    results.append(
        {
            "Model": name,
            "Params": params,
            "R2": r2_score(y_test, pred),
            "MSE": mean_squared_error(y_test, pred),
            "MAE": mean_absolute_error(y_test, pred),
        }
    )


# --- БАЗОВЫЙ УРОВЕНЬ: Линейная регрессия (из Лабы 3) ---
lr_model = LinearRegression().fit(X_train_f, y_train)
log_result("Linear Regression (L3)", "Default", lr_model)

# --- RANDOM FOREST (Ручной подбор гиперпараметров) ---
rf_sets = [
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 200, "max_depth": 20},
]

for i, params in enumerate(rf_sets):
    rf = RandomForestRegressor(**params, random_state=42).fit(X_train_f, y_train)
    log_result(f"Random Forest v{i+1}", str(params), rf)

# --- XGBOOST (Ручной подбор гиперпараметров) ---
xgb_sets = [
    {"n_estimators": 100, "learning_rate": 0.05},
    {"n_estimators": 100, "learning_rate": 0.2},
    {"n_estimators": 300, "learning_rate": 0.1},
]

for i, params in enumerate(xgb_sets):
    xgb = XGBRegressor(**params, random_state=42).fit(X_train_f, y_train)
    log_result(f"XGBoost v{i+1}", str(params), xgb)

print("\nТаблица сравнения результатов:")
print(pd.DataFrame(results).to_string(index=False))
