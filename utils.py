import random
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

def normalize(data):
    """
    Normalize the data to fall within the range [-1, 1].
    
    Parameters:
    - data: Input data as a numpy array.
    
    Returns:
    - Normalized data.
    """
    min_value = np.min(data, axis=0)
    max_value = np.max(data, axis=0)
    return (data - min_value) / (max_value - min_value) * 2 - 1

def generate_starting_centroids(k, dim):
    """
    Generates k centroids with coordinates in the range [-1, 1] using the sphere packing algorithm described in arxiv.org/pdf/1504.05998
    """
    sphere_radius = 1/k
    prev_fail_radius = 1.0
    prev_success_radius = 0.0
    best_centroids = []
    for i in range(15):
        print(f"Attempt {i+1}: Trying sphere radius {sphere_radius}; prev success {prev_success_radius}, prev fail {prev_fail_radius}")
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

def DPLloyd(data, steps, starting_centroids):
    """
    Placeholder for DPLloyd algorithm.
    This function is not implemented but serves as a placeholder.
    
    Parameters:
    - data: Input data points.
    - steps: Number of iterations.
    - starting_centroids: Initial centroids for clustering.
    
    Returns:
    - None
    """
    centroids = starting_centroids.copy()
    for i in range(steps):
        # Assign clusters based on the closest centroid TODO add DP
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        clusters = np.argmin(distances, axis=1)

        # Update centroids TODO add DP
        new_centroids = np.array([data[clusters == k].mean(axis=0) for k in range(len(centroids))])
        
        centroids = new_centroids
    raise NotImplementedError("DPLloyd algorithm is not implemented yet.")
    return centroids

if __name__ == "__main__":
    # Example usage
    data = normalize(np.random.rand(100, 2)+np.array([generate_laplace_noise(4.0, 2.0, 2) for i in range(50)] + [generate_laplace_noise(-1.0, 1.0, 2) for i in range(50)]))  # 100 points in 2D
    starting_centroids = generate_starting_centroids(5, 2)
    centroids = k_means(data, steps=10, starting_centroids=starting_centroids)
    plt.scatter(data[:, 0], data[:, 1], c='blue', label='Data Points')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', label='Centroids', marker="s")
    plt.legend()
    plt.show()