# k-Nearest Neighbors (K-NN) Algorithm

import matplotlib.pyplot as plt

class KNN:
    def __init__(self):
        self.count_col = int(input("How many columns (excluding label column): "))
        self.col = []  # Store column names
        self.data = []  # Store rows of data
        self.k = int(input("Enter the value of K (number of neighbors): "))

        for i in range(self.count_col):
            col_data = input(f"Enter Name for Column {i + 1}: ")
            self.col.append(col_data)

        self.col.append("Label")  # Last column is the label
        self.col.append("Distance")  # Column for computed distances

        count_rows = int(input("How many rows of data: "))

        for i in range(count_rows):
            row = []
            print(f"\nEntering data for row {i + 1}:")
            for col in self.col[:-1]:  # Excluding Distance column
                value = float(input(f"Enter value for {col}: ")) if col != "Label" else input(f"Enter label for {col}: ")
                row.append(value)
            row.append(None)  # Placeholder for Distance
            self.data.append(row)

        print("\nCollected column names:", self.col)
        self.print_data()

    def print_data(self):
        print("\nStored Data (Sorted by Distance):")
        print("bri | sat | Label | Distance")
        for row in self.data:
            print(f"{row[0]:.2f} | {row[1]:.2f} | {row[2]} | {row[3]}")

    def euclidean_distance(self, point1, point2):
        # Calculate Euclidean distance between two points (ignoring labels).
        return sum((a - b) ** 2 for a, b in zip(point1, point2)) ** 0.5

    def get_k_nearest_neighbors(self, test_point):
        # Find K-nearest neighbors for a given test point.
        distances = []

        for row in self.data:
            dist = self.euclidean_distance(test_point, row[:-2])  # Ignore Label & Distance
            distances.append([row[0], row[1], row[2], round(dist, 2)])  # Store data with computed distance

        distances.sort(key=lambda x: x[3])  # Sort by Distance (ascending)
        return distances[:self.k]  # Return K closest neighbors

    def predict(self, test_point):
        """Predict the label of a given test point using KNN."""
        neighbors = self.get_k_nearest_neighbors(test_point)
        label_counts = {}  # Manual label counting

        for _, _, label, _ in neighbors:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        # Majority voting (choose label with highest count)
        predicted_label = max(label_counts, key=label_counts.get)
        return predicted_label, neighbors

    def draw_graph(self, test_point, neighbors, prediction):
        """Displays data points and test points as a scatter plot."""
        plt.figure(figsize=(7, 5))

        # Plot stored data points
        for row in self.data:
            x, y, label, _ = row
            color = "red" if label == "red" else "blue"
            plt.scatter(x, y, c=color, label=label, edgecolors="black")

        # Plot test point
        plt.scatter(test_point[0], test_point[1], c="green", label="Test Point", edgecolors="black", marker="x", s=100)

        # Draw lines to K-nearest neighbors
        for x, y, _, _ in neighbors:
            plt.plot([test_point[0], x], [test_point[1], y], "k--", linewidth=0.8)

        plt.title(f"Predicted Class: {prediction}")
        plt.xlabel("bri")
        plt.ylabel("sat")
        plt.legend()
        plt.grid()
        plt.show()


# Create an instance and predict a new sample
knn = KNN()

# Input test sample
test_sample = [float(input(f"Enter value for {col}: ")) for col in knn.col[:-2]]  # Ignore Label & Distance column
prediction, nearest_neighbors = knn.predict(test_sample)

# Display updated data with computed distances
print("\nUpdated Data with Computed Distances (Sorted by Distance):")
print("bri | sat | Label | Distance")
for row in nearest_neighbors:
    print(f"{row[0]:.2f} | {row[1]:.2f} | {row[2]} | {row[3]:.2f}")

print("\nPredicted class:", prediction)

# Draw the graph
knn.draw_graph(test_sample, nearest_neighbors, prediction)
