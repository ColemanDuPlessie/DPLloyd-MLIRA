import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torchvision import models
from opacus.validators import ModuleValidator
from neuralnet import get_CIFAR10, train_model, test, device
from lira_neuralnet import get_confidences

def get_confidences(data, model): # TODO this is a kludge, we should not be overriding the import here, but I want to have the tqdm for testing purposes
    """
    Returns a 1D array of a confidence metric denoting the confidence that each sample in the data belongs to the training set.
    """
    ans = np.array([])
    for batch in tqdm(data):
        prediction = model(batch[0].to(device)).detach().cpu().numpy()
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

        out = (get_confidences(train_loader, model), get_confidences(test_loader, model))
        ans.append(out)
    return ans

def lira_attack(train_advantages, test_advantages):
    """
    returns two 2D binary arrays whose elements are True if the relevant sample is classified as being in the training set
    and False if it is classified as being in the test set, according to the LIRA.
    """
    pass # TODO

if __name__ == "__main__":
    ans = generate_results(num_models=1, private=False)
    print(ans[0][0].shape, ans[0][1].shape)