# k-MEANS CLUSTERING
import random

def euclidean_d(p1, p2):
    # Computes the Euclidean distance between two points.
    return sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5

def initialize_centroids(data, k):
    # Selects k random points as initial centroids.
    return random.sample(data, k)

def assign_clusters(data, centroids):
    # Assigns each data point to the nearest centroid.
    clusters = [[] for _ in range(len(centroids))]
    for point in data:
        distances = [euclidean_d(point, centroid) for centroid in centroids]
        nearest_centroid = distances.index(min(distances))
        clusters[nearest_centroid].append(point)
    return clusters

def compute_new_centroids(clusters, centroids):
    # Computes the new centroids as the mean of assigned points.
    new_centroids = []
    for i, cluster in enumerate(clusters):
        if cluster:  # Avoid division by zero
            new_centroid = [sum(dim) / len(cluster) for dim in zip(*cluster)]
        else:
            new_centroid = centroids[i]  # Keep previous centroid if cluster is empty
        new_centroids.append(new_centroid)
    return new_centroids

def kmeans(data, k, max_iters=100, tolerance=1e-6):
    # Runs the K-Means clustering algorithm.
    centroids = initialize_centroids(data, k)
    
    for _ in range(max_iters):
        clusters = assign_clusters(data, centroids)
        new_centroids = compute_new_centroids(clusters, centroids)

        # Check for convergence
        if all(euclidean_d(new, old) < tolerance for new, old in zip(new_centroids, centroids)):
            break
        
        centroids = new_centroids

    return centroids, clusters

# Example 
data = [[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]]
k = 2

final_centroids, clustered_data = kmeans(data, k)

# Display results
print("Final Centroids:")
for centroid in final_centroids:
    print([f"{coord:.2f}" for coord in centroid])

print("\nCluster Assignments:")
for i, cluster in enumerate(clustered_data):
    print(f"Cluster {i}: {[point for point in cluster]}")
