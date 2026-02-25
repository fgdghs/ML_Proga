import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("LAB_1//data.csv")

x_top = 10
y_top = 10

x_down = -10
y_down = -10

filtered_df = df[
    (df["x"] >= x_down) & (df["x"] <= x_top) & (df["y"] >= y_down) & (df["y"] <= y_top)
]

plt.figure(figsize=(8, 6))
plt.scatter(filtered_df["x"], filtered_df["y"], alpha=0.5, s=10, color="r")

plt.xlabel("x")
plt.ylabel("y")
plt.grid(
    True,
    which="major",
    axis="both",
    linestyle="--",
    color="gray",
    linewidth=0.5,
    alpha=0.7,
)
plt.axis("equal")
plt.show()
