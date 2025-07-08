import numpy as np
import matplotlib.pyplot as plt

def generate_laplace_noise(mean=0.0, scale=1.0, size=1):
    """
    Generate Laplace noise.
    
    Parameters:
    - mean: The mean of the Laplace distribution.
    - scale: The scale parameter (b) of the Laplace distribution.
    - size: The number of samples to generate.
    
    Returns:
    - A numpy array of Laplace noise samples.
    """
    return np.random.laplace(loc=mean, scale=scale, size=size)

def normalize(data, range=(-1, 1)):
    """
    Normalize the data to fall within the listed range.
    
    Parameters:
    - data: Input data as a numpy array.
    - range: A tuple specifying the desired range for normalization.
    
    Returns:
    - Normalized data.
    """
    min_value = np.min(data, axis=0)
    max_value = np.max(data, axis=0)
    return (data - min_value) / (max_value - min_value) * (range[1]-range[0]) + range[0]

def generate_starting_centroids(k, dim):
    """
    Generates k centroids with coordinates in the range [-1, 1] using the sphere packing algorithm described in arxiv.org/pdf/1504.05998
    """
    sphere_radius = 1/k
    prev_fail_radius = 1.0
    prev_success_radius = 0.0
    best_centroids = []
    for i in range(15):
        # print(f"Attempt {i+1}: Trying sphere radius {sphere_radius}; prev success {prev_success_radius}, prev fail {prev_fail_radius}")
        centroids = []
        for j in range(k*5+100):
            c = np.random.uniform(-1.0+sphere_radius, 1.0-sphere_radius, dim)
            if all(np.linalg.norm(c - np.array(existing_c)) >= sphere_radius*2 for existing_c in centroids):
                centroids.append(c)
                if len(centroids) == k:
                    best_centroids = centroids
                    prev_success_radius = sphere_radius
                    sphere_radius = (sphere_radius + prev_fail_radius) / 2
                    break
        else:
            prev_fail_radius = sphere_radius
            sphere_radius = (sphere_radius + prev_success_radius) / 2
    return best_centroids

def k_means(data, steps, starting_centroids):
    centroids = starting_centroids.copy()
    for i in range(steps):
        # Assign clusters based on the closest centroid
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        clusters = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = np.array([data[clusters == k].mean(axis=0) for k in range(len(centroids))])
        
        centroids = new_centroids
    return centroids

def DPLloyd(data, steps, starting_centroids, eps):
    """
    
    Parameters:
    - data: Input data points.
    - steps: Number of iterations.
    - starting_centroids: Initial centroids for clustering.
    - eps: Privacy budget for differential privacy.
    
    Returns:
    - A list of centroids
    """
    eps_per_step = eps / (2*steps) # We add noise twice per step, once at cluster assignment and once at centroid update
    centroids = starting_centroids.copy()
    for i in range(steps):
        # Assign clusters based on the closest centroid
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        clusters = np.argmin(distances, axis=1)
        counts = np.bincount(clusters, minlength=len(centroids))
        # print(counts)
        privatized_counts = counts + generate_laplace_noise(0.0, 2.0/eps_per_step, len(centroids))  # Add Laplace noise for differential privacy
        # print(privatized_counts)

        # Update centroids
        new_centroids = np.array([(data[clusters == k].sum(axis=0)+generate_laplace_noise(0.0, 2.0/eps_per_step, data.shape[1]))/privatized_counts[k] if privatized_counts[k] >= 1 else np.random.rand(data.shape[1])*2-1 for k in range(len(centroids))])
        
        centroids = new_centroids
    return centroids

def infer_labels(centroids, data, keys):
    """
    Infer labels for the centroids based on the data points.

    Parameters:
    - centroids: The centroids obtained from clustering.
    - data: The original data points.
    - keys: The actual labels for the data points.

    Returns:
    - A list of inferred labels for the centroids.
    """
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    found_clusters = np.argmin(distances, axis=1)
    
    inferred_labels = []
    for i in range(len(centroids)):
        cluster_keys = keys[found_clusters == i]
        if len(cluster_keys) > 0:
            inferred_labels.append(np.bincount(cluster_keys).argmax())
        else:
            inferred_labels.append(-1)  # No points assigned to this centroid
    return inferred_labels

def check_accuracy(raw_centroids, data, clusters):
    """
    Check the accuracy of the clustering by comparing the centroids with the actual data points.

    Parameters:
    - raw_centroids: The centroids obtained from clustering.
    - data: The original data points.
    - clusters: The cluster assignments for each data point.

    Returns:
    - A float representing the accuracy of the clustering.
    """
    centroids = raw_centroids[infer_labels(raw_centroids, data, clusters)]

    correct = 0
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    found_clusters = np.argmin(distances, axis=1)
    for i in range(len(data)):
        if found_clusters[i] == clusters[i]:
            correct += 1
    if len(centroids) == 2 and correct < len(data) / 2:
        correct = len(data) - correct  # If we have two clusters, we can easily handle the case where they are swapped
        print("Swapped clusters detected. This should never happen, due to the infer_labels function, so if you're seeing this, something has gone wrong.")
    return correct / len(data)

if __name__ == "__main__":
    # Example usage
    data = normalize(np.random.rand(500, 2)+np.array([generate_laplace_noise(4.0, 2.0, 2) for i in range(250)] + [generate_laplace_noise(-1.0, 1.0, 2) for i in range(250)]), (-1, 1))
    classes = [0 for _ in range(250)] + [1 for _ in range(250)]
    starting_centroids = generate_starting_centroids(2, 2)
    centroids = starting_centroids.copy()
    DP_centroids = starting_centroids.copy()
    eps = 2.0
    steps = 4
    for i in range(steps):
        DP_centroids = DPLloyd(data, steps=1, starting_centroids=DP_centroids, eps=eps/steps)
        centroids = k_means(data, steps=1, starting_centroids=centroids)
        plt.scatter(data[:250, 0], data[:250, 1], c='blue', label='Data Points (class 1)')
        plt.scatter(data[250:, 0], data[250:, 1], c='blue', label='Data Points (class 2)')
        plt.scatter(DP_centroids[:, 0], DP_centroids[:, 1], c='red', label='DPLLoyd Centroids', marker="X")
        plt.scatter(centroids[:, 0], centroids[:, 1], c='green', label='Non-private Centroids', marker="X")
        plt.legend()
        plt.show()
        print(f"Centroids accuracy: {check_accuracy(centroids, data, classes)}")
        print(f"DP centroids accuracy: {check_accuracy(DP_centroids, data, classes)}")