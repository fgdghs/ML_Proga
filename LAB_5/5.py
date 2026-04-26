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
import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV


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

# --- 1. ПОДГОТОВКА ДАННЫХ И СЕТКИ ---
# (Предполагается, что X_train_f, X_test_f, y_train, y_test уже в памяти)

# Общая сетка параметров для всех методов
search_spaces = {
    "n_estimators": [100, 200, 300],
    "learning_rate": (0.01, 0.1, "log-uniform"),  # Для Байеса можно задавать диапазоны
    "max_depth": [3, 5, 7, 10],
    "subsample": (0.6, 1.0, "uniform"),
}

# Для Grid и Random преобразуем диапазоны в списки, чтобы избежать ошибок
simple_param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7, 10],
    "subsample": [0.6, 0.8, 1.0],
}

results_list = []


def run_experiment(name, search_obj):
    print(f"Запуск {name}...")
    start_time = time.time()
    search_obj.fit(X_train_f, y_train)
    end_time = time.time() - start_time

    best_model = search_obj.best_estimator_
    y_pred = best_model.predict(X_test_f)

    # Сбор метрик
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    results_list.append(
        {
            "Method": name,
            "Time_sec": end_time,
            "R2": r2,
            "MSE": mse,
            "MAE": mae,
            "Best_Params": search_obj.best_params_,
        }
    )
    print(f"Завершено. Время: {end_time:.2f}с, R2: {r2:.4f}")


# --- 2. ВЫПОЛНЕНИЕ ПОИСКА ---

# A. Поиск по решетке
run_experiment(
    "Grid Search",
    GridSearchCV(
        XGBRegressor(random_state=42), simple_param_grid, cv=3, n_jobs=-1, scoring="r2"
    ),
)

# B. Случайный поиск (15 итераций)
run_experiment(
    "Random Search",
    RandomizedSearchCV(
        XGBRegressor(random_state=42),
        simple_param_grid,
        n_iter=15,
        cv=3,
        n_jobs=-1,
        scoring="r2",
        random_state=42,
    ),
)

# C. Байесовский поиск (15 итераций)
run_experiment(
    "Bayesian Search",
    BayesSearchCV(
        XGBRegressor(random_state=42),
        search_spaces,
        n_iter=15,
        cv=3,
        n_jobs=-1,
        scoring="r2",
        random_state=42,
    ),
)

# --- 3. ВИЗУАЛИЗАЦИЯ (Пункты 2 и 3 задания) ---
# --- 3. ВЫВОД РЕЗУЛЬТАТОВ (Оптимальные параметры + Метрики) ---
df_res = pd.DataFrame(results_list)

print("\n" + "=" * 50)
print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА МЕТОДОВ ПОДБОРА")
print("=" * 50)

# Выводим основные метрики
print(df_res[["Method", "Time_sec", "R2", "MAE"]].to_string(index=False))

print("\n" + "=" * 50)
print("НАЙДЕННЫЕ ОПТИМАЛЬНЫЕ ГИПЕРПАРАМЕТРЫ:")
print("=" * 50)

# Проходим циклом по результатам и печатаем параметры для каждой модели отдельно
for index, row in df_res.iterrows():
    print(f"\nМетод: {row['Method']}")
    print(f"Лучший R2: {row['R2']:.4f}")
    print("-" * 20)
    for param, value in row["Best_Params"].items():
        print(f"  {param}: {value}")
