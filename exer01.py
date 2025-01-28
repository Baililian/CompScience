import matplotlib.pyplot as plt
import numpy as np

x = [] 
y = [] 
for i in range(1, 8, 1):  # START, END, INCREMENT
    enter_x = int(input(f"Enter Number for X (index {i}): "))
    enter_y = int(input(f"Enter Number for Y (index {i}): "))
    x.append(enter_x)  # APPEND TO THE X LIST 
    y.append(enter_y)  # APPEND TO THE Y LIST 
print("\nTABLE X AND Y")
print(f"X Values: {x}")
print(f"Y Values: {y}")

# CALCULATE MEANS OF X & Y
mean_x = sum(x) / len(x)
mean_y = sum(y) / len(y)
print("\nCalculated Means:")
print(f"Mean of X: {mean_x:.2f}")
print(f"Mean of Y: {mean_y:.2f}")

# CALCULATE SLOPE (m) AND INTERCEPT (b)
n = len(x)
squared_x = sum([xi ** 2 for xi in x])  
xy_sum = sum([xi * yi for xi, yi in zip(x, y)])  
denominator = ((n * squared_x) - (sum(x)) ** 2)

m = ((n * xy_sum) - (sum(x) * sum(y))) / denominator
b = ((sum(y) * squared_x) - (sum(x) * xy_sum)) / denominator
print("\nRegression Line:")
print(f"Slope (m): {m:.2f}")
print(f"Intercept (b): {b:.2f}")

# Predicted y values y = mx + b
predicted_y = [(m * xi) + b for xi in x]
print("\nPredicted Y Values:")
for xi, yi in zip(x, predicted_y):
    print(f"For X = {xi}, Predicted Y = {yi:.2f}")

# Plotting the data points and regression line
plt.scatter(x, y, color='#0f0', label='Data points')
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='#f00', label='Regression Line')
plt.title("Linear Regression", fontsize=15)
plt.xlabel("X-values", fontsize=12)
plt.ylabel("Y-values", fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.5)
plt.show()
