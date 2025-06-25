import numpy as np
import matplotlib.pyplot as plt
from rice import get_rice_dataset, split_dataset
from kmeans import normalize, generate_starting_centroids, k_means, DPLloyd, check_accuracy

def lira_attack():
    pass

if __name__ == "__main__":
    data = get_rice_dataset()
    train, test = split_dataset(data, train_size=0.5)

    train_keys = np.array([0 if k == "Cammeo" else 1 for k in train[:, -1]])
    train = normalize(train[:, :-1], (-1, 1)).astype(np.float64)
    test_keys = np.array([0 if k == "Cammeo" else 1 for k in test[:, -1]])
    test = normalize(test[:, :-1], (-1, 1)).astype(np.float64)
    
    print("Rice dataset loaded successfully!")

    for i in range(10):

        centroids = generate_starting_centroids(2, train.shape[1])
        # print(train.shape, train[:5], train.dtype)

        nonprivate_centroids = centroids.copy()
        nonprivate_train_accs = []
        for step in range(10):
            nonprivate_centroids = k_means(train, steps=1, starting_centroids=nonprivate_centroids)
            print(f"Train accuracy of non-private k-means after step {step}: {check_accuracy(nonprivate_centroids, train, train_keys)}")
            print(f"Test accuracy of non-private k-means after step {step}: {check_accuracy(nonprivate_centroids, train, train_keys)}")
            nonprivate_train_accs.append(check_accuracy(nonprivate_centroids, train, train_keys))

        eps = 0.25
        DP_steps = 5
        eps_per_step = eps / DP_steps
        private_centroids = centroids.copy()
        private_train_accs = []
        for step in range(DP_steps):
            private_centroids = DPLloyd(train, steps=1, starting_centroids=private_centroids, eps=eps_per_step)
            print(f"Train accuracy of private k-means with epsilon={eps} after step {step}: {check_accuracy(private_centroids, train, train_keys)}")
            print(f"Test accuracy of private k-means with epsilon={eps} after step {step}: {check_accuracy(private_centroids, train, train_keys)}")
            private_train_accs.append(check_accuracy(private_centroids, train, train_keys))

        plt.plot(nonprivate_train_accs, c='blue')
        plt.plot(private_train_accs, c='orange')
    plt.show()