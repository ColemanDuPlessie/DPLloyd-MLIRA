import numpy as np
import matplotlib.pyplot as plt
from rice import get_rice_dataset, split_dataset
from kmeans import normalize, generate_starting_centroids, k_means, DPLloyd, check_accuracy

def lira_attack(train_data, test_data, centroids, label=""):
    """
    returns a 1D array of the confidence of each training sample being classified as the first class
    using the LIRA attack.
    """
    distances = np.linalg.norm(train_data[:, np.newaxis] - centroids, axis=2)
    advantages = np.array([1-min(x) for x in distances]) # TODO this is not correct in the case where the model has failed to converge and a point is assigned incorrectly; this may very well be the central case in which LIRA attacks are functional, so rectifying it is a priority
    sorted_train_advantages = np.sort(advantages)

    distances = np.linalg.norm(test_data[:, np.newaxis] - centroids, axis=2)
    advantages = np.array([1-min(x) for x in distances]) # TODO this is not correct in the case where the model has failed to converge and a point is assigned incorrectly; this may very well be the central case in which LIRA attacks are functional, so rectifying it is a priority
    sorted_test_advantages = np.sort(advantages)

    max_diff = 0
    max_diff_point = None
    train_pointer = 0
    test_pointer = 0
    while train_pointer < sorted_train_advantages.shape[0] and test_pointer < sorted_test_advantages.shape[0]:
        if sorted_train_advantages[train_pointer] < sorted_test_advantages[test_pointer]:
            train_pointer += 1
        else:
            test_pointer += 1
        diff = test_pointer - train_pointer
        if diff > max_diff:
            max_diff = diff
            max_diff_point = (sorted_train_advantages[train_pointer] + sorted_test_advantages[test_pointer]) / 2

    plt.plot(sorted_train_advantages, [i for i in range(sorted_train_advantages.shape[0])], label=f'Train ({label})')
    plt.plot(sorted_test_advantages, [i for i in range(sorted_test_advantages.shape[0])], label=f'Test ({label})')
    plt.plot([max_diff_point, max_diff_point], [0, sorted_train_advantages.shape[0]-1], c='red', linestyle='--', label=f'Best Cutoff ({label})')

if __name__ == "__main__":
    data = get_rice_dataset()
    
    print("Rice dataset loaded successfully!")

    for i in range(10):
        train, test = split_dataset(data, train_size=0.5)

        train_keys = np.array([0 if k == "Cammeo" else 1 for k in train[:, -1]])
        train = normalize(train[:, :-1], (-1, 1)).astype(np.float64)
        test_keys = np.array([0 if k == "Cammeo" else 1 for k in test[:, -1]])
        test = normalize(test[:, :-1], (-1, 1)).astype(np.float64)


        centroids = generate_starting_centroids(2, train.shape[1])

        nonprivate_centroids = centroids.copy()
        nonprivate_train_accs = []
        for step in range(10):
            nonprivate_centroids = k_means(train, steps=1, starting_centroids=nonprivate_centroids)
            print(f"Train accuracy of non-private k-means after step {step}: {check_accuracy(nonprivate_centroids, train, train_keys)}")
            print(f"Test accuracy of non-private k-means after step {step}: {check_accuracy(nonprivate_centroids, train, train_keys)}")
            nonprivate_train_accs.append(check_accuracy(nonprivate_centroids, train, train_keys))

            
        # lira_attack(train, test, nonprivate_centroids)

        eps = 0.5
        DP_steps = 5
        eps_per_step = eps / DP_steps
        private_centroids = centroids.copy()
        private_train_accs = []
        for step in range(DP_steps):
            private_centroids = DPLloyd(train, steps=1, starting_centroids=private_centroids, eps=eps_per_step)
            print(f"Train accuracy of private k-means with epsilon={eps} after step {step}: {check_accuracy(private_centroids, train, train_keys)}")
            print(f"Test accuracy of private k-means with epsilon={eps} after step {step}: {check_accuracy(private_centroids, train, train_keys)}")
            private_train_accs.append(check_accuracy(private_centroids, train, train_keys))
        
        lira_attack(train, test, nonprivate_centroids, label="nonprivate")
        lira_attack(train, test, private_centroids, label="private")

        plt.legend()
        plt.show()

        # plt.plot(nonprivate_train_accs, c='blue')
        # plt.plot(private_train_accs, c='orange')
    # plt.show()