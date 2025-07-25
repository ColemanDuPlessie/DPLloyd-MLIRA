import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets.rice import get_rice_dataset, split_rice_dataset, NUM_RICE_SAMPLES
from datasets.iris import get_iris_dataset, split_iris_dataset, NUM_IRIS_SAMPLES
from kmeans import generate_starting_centroids, k_means, DPLloyd, check_accuracy

def get_kmeans_confidences(data, centroids):
    """
    Returns a 1D array of a confidence metric denoting the confidence that each sample in the data belongs to the training set.
    """
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    return np.array([1-min(x) for x in distances])  # High numbers indicate that the sample is close to a centroid, which means it is likely in the training set.

def lira_attack(train_advantages, test_advantages, label=""):
    """
    returns two 1D binary arrays whose elements are True if the relevant sample is classified as being in the training set
    and False if it is classified as being in the test set, according to the LIRA.
    """
    sorted_train_advantages = np.sort(train_advantages)
    sorted_test_advantages = np.sort(test_advantages)

    max_diff = 0
    max_diff_point = min(sorted_train_advantages[0], sorted_test_advantages[0])-0.001  # Start with a threshold that classifies everything as being in the training set.
    max_diff_train_pointer = 0
    max_diff_test_pointer = 0
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
            try:
                max_diff_point = (sorted_train_advantages[train_pointer] + sorted_test_advantages[test_pointer]) / 2
            except IndexError:
                max_diff_point = sorted_train_advantages[train_pointer] if train_pointer < sorted_train_advantages.shape[0] else sorted_test_advantages[test_pointer]
            max_diff_train_pointer = train_pointer
            max_diff_test_pointer = test_pointer

    if label != "":
        plt.plot(sorted_train_advantages, [i for i in range(sorted_train_advantages.shape[0])], label=f'Train ({label})')
        plt.plot(sorted_test_advantages, [i for i in range(sorted_test_advantages.shape[0])], label=f'Test ({label})')
        plt.plot([max_diff_point, max_diff_point], [0, sorted_train_advantages.shape[0]-1], c='red', linestyle='--', label=f'Best Cutoff ({label})')
    
    train_detected = [x > max_diff_point for x in train_advantages]
    test_detected = [x > max_diff_point for x in test_advantages]

    return (train_detected, test_detected, max_diff, max_diff_train_pointer, max_diff_test_pointer)

if __name__ == "__main__":
    DATASET_TO_USE = "iris" # Either "rice" or "iris"

    if DATASET_TO_USE == "iris":
        data = get_iris_dataset()
        print("Iris dataset loaded successfully!")
    else:
        data = get_rice_dataset()
        print("Rice dataset loaded successfully!")

    nonprivate_success_rates = []
    private_success_rates = []
    nonprivate_train_accs = []
    private_train_accs = []

    PRINT_DURING_LOOP = False

    for i in tqdm(range(5000)):
        if DATASET_TO_USE == "iris":
            TRAIN_SET_SIZE = 0.5
            train, train_keys, test, test_keys = split_iris_dataset(data, train_size=TRAIN_SET_SIZE)
            centroids = generate_starting_centroids(3, train.shape[1])
        else:
            TRAIN_SET_SIZE = 0.5
            train, train_keys, test, test_keys = split_rice_dataset(data, train_size=TRAIN_SET_SIZE)
            centroids = generate_starting_centroids(2, train.shape[1])

        nonprivate_centroids = centroids.copy()
        for step in range(5):
            nonprivate_centroids = k_means(train, steps=1, starting_centroids=nonprivate_centroids)
            if PRINT_DURING_LOOP:
                print(f"Train accuracy of non-private k-means after step {step}: {check_accuracy(nonprivate_centroids, train, train_keys)}")
                print(f"Test accuracy of non-private k-means after step {step}: {check_accuracy(nonprivate_centroids, train, train_keys)}")
        nonprivate_train_accs.append(check_accuracy(nonprivate_centroids, train, train_keys))

        eps = 1.0
        DP_steps = 5
        eps_per_step = eps / DP_steps
        private_centroids = centroids.copy()
        for step in range(DP_steps):
            private_centroids = DPLloyd(train, steps=1, starting_centroids=private_centroids, eps=eps_per_step)
            if PRINT_DURING_LOOP:
                print(f"Train accuracy of private k-means with epsilon={eps} after step {step}: {check_accuracy(private_centroids, train, train_keys)}")
                print(f"Test accuracy of private k-means with epsilon={eps} after step {step}: {check_accuracy(private_centroids, train, train_keys)}")
        private_train_accs.append(check_accuracy(private_centroids, train, train_keys))
        
        nonprivate_attack_guesses = lira_attack(get_kmeans_confidences(train, nonprivate_centroids), get_kmeans_confidences(test, nonprivate_centroids)) # , label="nonprivate")
        nonprivate_attack_train_acc = np.mean(nonprivate_attack_guesses[0])
        nonprivate_attack_test_acc = np.mean(nonprivate_attack_guesses[1])
        nonprivate_attack_success_rate = (TRAIN_SET_SIZE*nonprivate_attack_train_acc) + (1-TRAIN_SET_SIZE)*(1-nonprivate_attack_test_acc)
        nonprivate_success_rates.append(nonprivate_attack_success_rate)

        private_attack_guesses = lira_attack(get_kmeans_confidences(train, private_centroids), get_kmeans_confidences(test, private_centroids)) # , label="private")
        private_attack_train_acc = np.mean(private_attack_guesses[0])
        private_attack_test_acc = np.mean(private_attack_guesses[1])
        private_attack_success_rate = (TRAIN_SET_SIZE*private_attack_train_acc) + (1-TRAIN_SET_SIZE)*(1-private_attack_test_acc)
        private_success_rates.append(private_attack_success_rate)
    
        if PRINT_DURING_LOOP:
            print(f"Accuracies of LIRA on...\n Non-private K-means model: {nonprivate_attack_success_rate}\n Private K-means model: {private_attack_success_rate}")

        # plt.legend()
        # plt.show()

    print(f"Average success rate of LIRA on non-private K-means model: {np.mean(nonprivate_success_rates)}")
    print(f"Average success rate of LIRA on private K-means model: {np.mean(private_success_rates)}")
    print(f"Average train accuracy of non-private K-means model: {np.mean(nonprivate_train_accs)}")
    print(f"Average train accuracy of private K-means model: {np.mean(private_train_accs)}")