import numpy as np
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
        train_loader, test_loader, train_datapoints = get_CIFAR10(train_set_size=0.5, test_set_is_leftover_train=True, train_datapoints=np.load(f"models/train_set_{i}.npy"))

        model_path = ("models/" + ("private" if private else "nonprivate") + "_resnet18_" + str(i) + ".pt")
        state_dict = torch.load(model_path, map_location=device)
        model = models.resnet18(num_classes=10)
        model = ModuleValidator.fix(model).to(device)
        model.load_state_dict(state_dict)

        out = (get_confidences(train_loader, model), get_confidences(test_loader, model), train_datapoints)
        ans.append(out)
    return ans

def lira_attack(train_advantages, test_advantages):
    """
    returns two 2D binary arrays whose elements are True if the relevant sample is classified as being in the training set
    and False if it is classified as being in the test set, according to the LIRA.
    """ # TODO rewrite this eventually.
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

def alt_lira_attack(train_advantages, test_advantages, train_frac=0.5):
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

if __name__ == "__main__":
    """
    dummy_train = np.array([1.59066522, 1.32747686, 1.53802216, 1.72594547, 1.26588047, 2.3016057 ])
    dummy_test  = np.array([2.35630631, 1.23455882, 1.53139615, 1.89570451])

    atk = lira_attack(dummy_train, dummy_test)
    train_acc = np.mean(atk[0])
    test_acc = 1.0-np.mean(atk[1])
    atk_success_rate = (0.6*train_acc) + (1-0.6)*(test_acc)
    print(f"Attack success rate: {atk_success_rate*100:.2f}% (Train acc: {train_acc*100:.2f}%, Test acc: {test_acc*100:.2f}%, Threshold: {atk[2]:.4f})")

    simplified_atk = alt_lira_attack(dummy_train, dummy_test, 0.6)
    train_acc = np.mean(simplified_atk[0])
    test_acc = 1.0-np.mean(simplified_atk[1])
    atk_success_rate = (0.6*train_acc) + (1-0.6)*(test_acc)
    print(f"Attack success rate: {atk_success_rate*100:.2f}% (Train acc: {train_acc*100:.2f}%, Test acc: {test_acc*100:.2f}%, Threshold: {simplified_atk[2]:.4f}))")
    """
    TRAIN_SET_SIZE = 0.5
    
    accs = []
    alt_accs = []
    
    ans = generate_results(num_models=64, private=False)
    print("Data generated...")
    for sample in tqdm(range(32)):
        train_idxs = [i for i in range(len(ans)) if sample in ans[i][2]]
        test_idxs = [i for i in range(len(ans)) if sample not in ans[i][2]]
        if len(train_idxs) < 1 or len(test_idxs) < 1:
            continue
        train_confidences = np.array([ans[i][0][list(ans[i][2]).index(sample)] for i in train_idxs])
        test_confidences = np.array([ans[i][1][[j for j in range(50000) if j not in ans[i][2]].index(sample)] for i in test_idxs])
        print(train_confidences.shape, test_confidences.shape, train_confidences, test_confidences)

        atk = lira_attack(train_confidences, test_confidences)
        train_acc = np.mean(atk[0])
        test_acc = 1.0-np.mean(atk[1])
        atk_success_rate = ((len(train_idxs)*train_acc) + (len(test_idxs)*test_acc)) / (len(train_idxs) + len(test_idxs))
        accs.append(atk_success_rate)
        print(ans[0][0].shape, ans[0][1].shape)
        print(f"Attack success rate: {atk_success_rate*100:.2f}% (Train acc: {train_acc*100:.2f}%, Test acc: {test_acc*100:.2f}%)")

        atk = alt_lira_attack(train_confidences, test_confidences)
        train_acc = np.mean(atk[0])
        test_acc = 1.0-np.mean(atk[1])
        atk_success_rate = ((len(train_idxs)*train_acc) + (len(test_idxs)*test_acc)) / (len(train_idxs) + len(test_idxs))
        alt_accs.append(atk_success_rate)
        print(ans[0][0].shape, ans[0][1].shape)
        print(f"Attack success rate: {atk_success_rate*100:.2f}% (Train acc: {train_acc*100:.2f}%, Test acc: {test_acc*100:.2f}%)")

    print(f"Average attack success rate: {sum(accs)/len(accs)}, average alt attack success rate: {sum(alt_accs)/len(alt_accs)}")
