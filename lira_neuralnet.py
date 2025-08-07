import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from neuralnet import get_CIFAR10, train_model, test, device

def get_confidences(data, model):
    """
    Returns a 1D array of a confidence metric denoting the confidence that each sample in the data belongs to the training set.
    """
    ans = np.array([])
    for batch in data:
        confidences = np.max(model(batch[0].to(device)).detach().cpu().numpy(), axis=0)
        ans = np.append(ans, confidences)
    return ans # High numbers indicate the model is highly confident about the sample's classification, which means it is likely in the training set.

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
    train_loader, test_loader = get_CIFAR10()
    print("CIFAR-10 loaded successfully!")

    nonprivate_success_rates = []
    private_success_rates = []
    nonprivate_train_accs = []
    private_train_accs = []

    PRINT_DURING_LOOP = False
    SAVE_TRAINED_MODELS = True
    SAVE_NUMBER_OFFSET = 0 # Used for adding additional models when some have already been trained.

    for i in tqdm(range(64)):
        TRAIN_SET_SIZE = 0.5
        train_loader, test_loader = get_CIFAR10(train_set_size=TRAIN_SET_SIZE, test_set_is_leftover_train=(TRAIN_SET_SIZE < 1.0))
        
        nonprivate_model = train_model(train_loader, epochs=2, private=False) # train non-private model
        nonprivate_train_accs.append(test(nonprivate_model, test_loader))

        eps = 1.0
        private_model = train_model(train_loader, epochs=100, eps=1.0) # train private model
        private_train_accs.append(test(private_model, test_loader))
        
        nonprivate_attack_guesses = lira_attack(get_confidences(train_loader, nonprivate_model), get_confidences(test_loader, nonprivate_model)) # , label="nonprivate")
        nonprivate_attack_train_acc = np.mean(nonprivate_attack_guesses[0])
        nonprivate_attack_test_acc = np.mean(nonprivate_attack_guesses[1])
        nonprivate_attack_success_rate = (TRAIN_SET_SIZE*nonprivate_attack_train_acc) + (1-TRAIN_SET_SIZE)*(1-nonprivate_attack_test_acc)
        nonprivate_success_rates.append(nonprivate_attack_success_rate)

        private_attack_guesses = lira_attack(get_confidences(train_loader, private_model), get_confidences(test_loader, private_model)) # , label="private")
        private_attack_train_acc = np.mean(private_attack_guesses[0])
        private_attack_test_acc = np.mean(private_attack_guesses[1])
        private_attack_success_rate = (TRAIN_SET_SIZE*private_attack_train_acc) + (1-TRAIN_SET_SIZE)*(1-private_attack_test_acc)
        private_success_rates.append(private_attack_success_rate)

        if SAVE_TRAINED_MODELS:
            torch.save(nonprivate_model.state_dict(), f"models/nonprivate_resnet18_{i+SAVE_NUMBER_OFFSET}.pt")
            torch.save(private_model.state_dict(), f"models/private_resnet18_{i+SAVE_NUMBER_OFFSET}.pt")
    
        if PRINT_DURING_LOOP:
            print(f"Accuracies of LIRA on...\n Non-private K-means model: {nonprivate_attack_success_rate}\n Private K-means model: {private_attack_success_rate}")

        # plt.legend()
        # plt.show()

    print(f"Average success rate of LIRA on non-private K-means model: {np.mean(nonprivate_success_rates)}")
    print(f"Average success rate of LIRA on private K-means model: {np.mean(private_success_rates)}")
    print(f"Average train accuracy of non-private K-means model: {np.mean(nonprivate_train_accs)}")
    print(f"Average train accuracy of private K-means model: {np.mean(private_train_accs)}")