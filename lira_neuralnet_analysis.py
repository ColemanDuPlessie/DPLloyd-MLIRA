from bisect import bisect_left
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
from tqdm import tqdm
import torch
from torchvision import models
from opacus.validators import ModuleValidator
from neuralnet import get_CIFAR10, device

def get_confidences(data, model):
    """
    Returns a 1D array of a confidence metric denoting the confidence that each sample in the data belongs to the training set.
    """
    ans = np.array([])
    for batch in tqdm(data):
        prediction = torch.softmax(model(batch[0].to(device)), dim=1).detach().cpu().numpy()
        confidences = np.max(prediction, axis=1)
        ans = np.concatenate((ans, confidences), axis=0)
    return ans # High numbers indicate the model is highly confident about the sample's classification, which means it is likely in the training set.

def generate_results(num_models=128, private=True):
    ans = []
    for i in tqdm(range(num_models)):
        if os.path.exists(f"models/confidences_model_{'private' if private else 'nonprivate'}_{i}.npy"):
            out = np.load(f"models/confidences_model_{'private' if private else 'nonprivate'}_{i}.npy", allow_pickle=True)
            ans.append((out[0], out[1], out[2]))
        else:
            train_loader, test_loader, train_datapoints = get_CIFAR10(train_set_size=0.5, test_set_is_leftover_train=True, train_datapoints=np.load(f"models/train_set_{i}.npy"))

            model_path = ("models/" + ("private" if private else "nonprivate") + "_resnet18_" + str(i) + ".pt")
            state_dict = torch.load(model_path, map_location=device)
            model = models.resnet18(num_classes=10)
            model = ModuleValidator.fix(model).to(device)
            model.load_state_dict(state_dict)

            out = (get_confidences(train_loader, model), get_confidences(test_loader, model), train_datapoints)
            to_save = np.array(out)
            np.save(f"models/confidences_model_{'private' if private else 'nonprivate'}_{i}.npy", to_save)
            ans.append(out)
    return ans

def brute_force_lira_attack(train_advantages, test_advantages):
    """
    returns two 2D binary arrays whose elements are True if the relevant sample is classified as being in the training set
    and False if it is classified as being in the test set, according to the LIRA.
    """
    sorted_train_advantages = np.sort(train_advantages)
    sorted_test_advantages = np.sort(test_advantages)

    max_diff = -len(train_advantages)-len(test_advantages)
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
                train_point = sorted_train_advantages[train_pointer-1] if train_pointer != 0 else None
                test_point = sorted_test_advantages[test_pointer-1] if test_pointer != 0 else None
                if train_point is None:
                    if test_point is None:
                        pass # We are on the first iteration
                    else:
                        max_diff_point = test_point + 0.0000001
                else:
                    if test_point is None:
                        max_diff_point = train_point + 0.0000001
                    else:
                        max_diff_point = (sorted_train_advantages[train_pointer-1] + sorted_test_advantages[test_pointer-1]) / 2
            except IndexError:
                max_diff_point = sorted_train_advantages[train_pointer] if train_pointer < sorted_train_advantages.shape[0] else sorted_test_advantages[test_pointer]
            max_diff_train_pointer = train_pointer
            max_diff_test_pointer = test_pointer
    
    train_detected = [x > max_diff_point for x in train_advantages]
    test_detected = [x > max_diff_point for x in test_advantages]

    return (train_detected, test_detected, max_diff_point, max_diff, max_diff_train_pointer, max_diff_test_pointer)

def guessed_threshold_lira_attack(train_advantages, test_advantages, train_frac=0.5):
    """
    returns two 2D binary arrays whose elements are True if the relevant sample is classified as being in the training set
    and False if it is classified as being in the test set, according to the LIRA.
    """
    sorted_advantages = np.sort(np.concatenate((train_advantages, test_advantages), axis=0))

    classified = int((1-train_frac) * sorted_advantages.shape[0])
    threshold = (sorted_advantages[classified-1]+sorted_advantages[classified])/2 # Avoids fencepost errors: if we want the first 5 elements, we want elements 0, 1, 2, 3, 4, not element 5
    
    train_detected = [x > threshold for x in train_advantages]
    test_detected = [x > threshold for x in test_advantages]

    return (train_detected, test_detected, threshold)

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / (2 * stddev))**2)

def find_gaussian_overlap(m1,m2,std1,std2): # TODO generalize this to arbitrary dimensions
  a = 1/(2*std1**2) - 1/(2*std2**2)
  b = m2/(std2**2) - m1/(std1**2)
  c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
  return np.roots([a,b,c])

def gaussian_lira_attack(train_advantages, test_advantages):
    """
    returns two 2D binary arrays whose elements are True if the relevant sample is classified as being in the training set
    and False if it is classified as being in the test set, according to the LIRA.
    """
    train_gaussian = scipy.stats.norm.fit(np.sort(train_advantages))
    test_gaussian = scipy.stats.norm.fit(np.sort(test_advantages))

    plot = False
    if plot:
        plt.plot([i/len(train_advantages-1) for i in np.arange(len(train_advantages))], gaussian(np.sort(train_advantages), 1, *train_gaussian), label="Train Gaussian Fit")
        plt.plot([i/len(test_advantages-1) for i in np.arange(len(test_advantages))], gaussian(np.sort(test_advantages), 1, *test_gaussian), label="Test Gaussian Fit")
        plt.hist(train_advantages, bins=30, alpha=0.5, label="Train Advantages")
        plt.hist(test_advantages, bins=30, alpha=0.5, label="Test Advantages")
        plt.legend()
        plt.show()
    
    threshold = find_gaussian_overlap(train_gaussian[0], test_gaussian[0], train_gaussian[1], test_gaussian[1])
    print(threshold)

    ans = None
    max_count = -1
    for t in threshold:
        train_detected = [x > t for x in train_advantages]
        test_detected = [x > t for x in test_advantages]
        if sum(train_detected) + (len(test_detected) - sum(test_detected)) > max_count:
            ans = (train_detected, test_detected, t)
            max_count = sum(train_detected) + (len(test_detected) - sum(test_detected))
    return ans

def approximate_bilira_attack(train_advantages1, test_advantages1, train_set1, train_advantages2, test_advantages2, train_set2):
    """
    Uses a hasty approximation that assumes the dividing line between in and out points runs orthogonally through the line between the average in point's confidence and the average out point's confidence. It is trivial to come up with examples where this is not true, but those examples may not be common or strong in practice, so hopefully this is a reasonable approximation.
    """
    train_advantages = []
    test_advantages = []
    for i in range(max((max(train_set1)+1), (max(train_set2)+1))):
        if i in train_set1 and i in train_set2:
            train_advantages.append((train_advantages1[train_set1.index(i)], train_advantages2[train_set2.index(i)]))
        elif i not in train_set1 and i not in train_set2:
            test_advantages.append((test_advantages1[i - bisect_left(train_set1, i)], test_advantages2[i - bisect_left(train_set2, i)]))
    sorted_train_advantages = np.sort(np.array(train_advantages), axis=0)
    sorted_test_advantages = np.sort(np.array(test_advantages), axis=0)

    train_avg = np.mean(sorted_train_advantages, axis=0)
    test_avg = np.mean(sorted_test_advantages, axis=0)
    diff = train_avg-test_avg
    unit_diff = diff / np.linalg.norm(diff)
    sorted_train_advantages = np.dot(sorted_train_advantages, unit_diff)
    sorted_test_advantages = np.dot(sorted_test_advantages, unit_diff)

    # print(sorted_train_advantages)
    # print(sorted_train_advantages.shape)
    return gaussian_lira_attack(sorted_train_advantages, sorted_test_advantages)

if __name__ == "__main__":
    """
    dummy_train = np.array([1.59066522, 1.32747686, 1.53802216, 1.72594547, 1.26588047, 2.3016057 ])
    dummy_test  = np.array([2.35630631, 1.23455882, 1.53139615, 1.89570451])

    atk = lira_attack(dummy_train, dummy_test)
    train_acc = np.mean(atk[0])
    test_acc = 1.0-np.mean(atk[1])
    atk_success_rate = (0.6*train_acc) + (1-0.6)*(test_acc)
    print(f"Attack success rate: {atk_success_rate*100:.2f}% (Train acc: {train_acc*100:.2f}%, Test acc: {test_acc*100:.2f}%, Threshold: {atk[2]:.4f})")

    simplified_atk = gaussian_lira_attack(dummy_train, dummy_test)
    train_acc = np.mean(simplified_atk[0])
    test_acc = 1.0-np.mean(simplified_atk[1])
    atk_success_rate = (0.6*train_acc) + (1-0.6)*(test_acc)
    print(f"Attack success rate: {atk_success_rate*100:.2f}% (Train acc: {train_acc*100:.2f}%, Test acc: {test_acc*100:.2f}%, Threshold: {simplified_atk[2]:.4f}))")

    bi_atk = approximate_bilira_attack(dummy_train, dummy_test)
    train_acc = np.mean(bi_atk[0])
    test_acc = 1.0-np.mean(bi_atk[1])
    atk_success_rate = ((bi_atk[3]*train_acc) + (bi_atk[4])*(test_acc))/(bi_atk[3]+bi_atk[4])
    print(f"Attack success rate: {atk_success_rate*100:.2f}% (Train acc: {train_acc*100:.2f}%, Test acc: {test_acc*100:.2f}%, Threshold: {bi_atk[2]:.4f})), Train point count: {bi_atk[3]}, Test point count: {bi_atk[4]}")
    """
    TRAIN_SET_SIZE = 0.5
    
    train_count = 0
    test_count = 0
    train_data = []
    test_data = []
    accs = []
    alt_accs = []
    two_point_accs = []
    
    ans = generate_results(num_models=256, private=False)
    print("Data generated...")
    for sample in tqdm(range(128)):
        train_idxs = [i for i in range(len(ans)) if sample in ans[i][2]]
        test_idxs = [i for i in range(len(ans)) if sample not in ans[i][2]]
        if len(train_idxs) < 1 or len(test_idxs) < 1:
            continue
        train_confidences = np.array([ans[i][0][list(ans[i][2]).index(sample)] for i in train_idxs])
        test_confidences = np.array([ans[i][1][[j for j in range(50000) if j not in ans[i][2]].index(sample)] for i in test_idxs])
        # print(train_confidences.shape, test_confidences.shape, train_confidences, test_confidences)

        train_idxs2 = [i for i in range(len(ans)) if sample+10000 in ans[i][2]]
        test_idxs2 = [i for i in range(len(ans)) if sample+10000 not in ans[i][2]]
        if len(train_idxs2) < 1 or len(test_idxs2) < 1:
            continue
        train_confidences2 = np.array([ans[i][0][list(ans[i][2]).index(sample+10000)] for i in train_idxs2])
        test_confidences2 = np.array([ans[i][1][[j for j in range(50000) if j not in ans[i][2]].index(sample+10000)] for i in test_idxs2])

        train_count += len(train_idxs)
        test_count += len(test_idxs)
        train_data.extend(train_confidences)
        test_data.extend(test_confidences)

        atk = brute_force_lira_attack(train_confidences, test_confidences)
        train_acc = np.mean(atk[0])
        test_acc = 1.0-np.mean(atk[1])
        atk_success_rate = ((len(train_idxs)*train_acc) + (len(test_idxs)*test_acc)) / (len(train_idxs) + len(test_idxs))
        accs.append(atk_success_rate)
        # print(ans[0][0].shape, ans[0][1].shape)
        # print(f"Attack success rate: {atk_success_rate*100:.2f}% (Train acc: {train_acc*100:.2f}%, Test acc: {test_acc*100:.2f}%)")

        atk = gaussian_lira_attack(train_confidences, test_confidences)
        train_acc = np.mean(atk[0])
        test_acc = 1.0-np.mean(atk[1])
        atk_success_rate = ((len(train_idxs)*train_acc) + (len(test_idxs)*test_acc)) / (len(train_idxs) + len(test_idxs))
        alt_accs.append(atk_success_rate)
        # print(ans[0][0].shape, ans[0][1].shape)
        # print(f"Attack success rate: {atk_success_rate*100:.2f}% (Train acc: {train_acc*100:.2f}%, Test acc: {test_acc*100:.2f}%)")

        atk = approximate_bilira_attack(train_confidences, test_confidences, train_idxs, train_confidences2, test_confidences2, train_idxs2)
        train_acc = np.mean(atk[0])
        test_acc = 1.0-np.mean(atk[1])
        atk_success_rate = (train_acc*len(atk[0])+test_acc*len(atk[1]))/(len(atk[0])+len(atk[1]))
        two_point_accs.append(atk_success_rate)
        # print(ans[0][0].shape, ans[0][1].shape)
        # print(f"Simplified two point attack success rate: {atk_success_rate*100:.2f}% (Train acc: {train_acc*100:.2f}%, Test acc: {test_acc*100:.2f}%)")

    print(f"Average attack success rate: {sum(accs)/len(accs)}, average gaussian attack success rate: {sum(alt_accs)/len(alt_accs)}, Average simplified two point success rate: {sum(two_point_accs)/len(two_point_accs)}")
    print("Brute Force Accuracies: " + str(accs))
    print("Gaussian Accuracies: " + str(alt_accs))
    print("Simple Two-point Accuracies: " + str(two_point_accs))