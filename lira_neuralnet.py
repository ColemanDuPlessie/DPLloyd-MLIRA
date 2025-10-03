import numpy as np
from tqdm import tqdm
import torch
from neuralnet import get_CIFAR10, train_model, test, device

if __name__ == "__main__":
    train_loader, test_loader = get_CIFAR10()
    print("CIFAR-10 loaded successfully!")

    nonprivate_train_accs = []
    private_train_accs = []

    PRINT_DURING_LOOP = False
    SAVE_TRAINED_MODELS = True
    SAVE_NUMBER_OFFSET = 0 # Used for adding additional models when some have already been trained.

    for i in tqdm(range(64)):
        TRAIN_SET_SIZE = 0.5
        train_loader, test_loader, train_indices = get_CIFAR10(train_set_size=TRAIN_SET_SIZE, test_set_is_leftover_train=(TRAIN_SET_SIZE < 1.0))
        
        nonprivate_model = train_model(train_loader, epochs=2, private=False) # train non-private model
        nonprivate_train_accs.append(test(nonprivate_model, test_loader))

        eps = 1.0
        private_model = train_model(train_loader, epochs=100, eps=1.0) # train private model
        private_train_accs.append(test(private_model, test_loader))

        if SAVE_TRAINED_MODELS:
            np.save(f"models/train_set_{i+SAVE_NUMBER_OFFSET}.npy", train_indices)
            torch.save(nonprivate_model.state_dict(), f"models/nonprivate_resnet18_{i+SAVE_NUMBER_OFFSET}.pt")
            torch.save(private_model.state_dict(), f"models/private_resnet18_{i+SAVE_NUMBER_OFFSET}.pt")
    
        del nonprivate_model
        del private_model
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"Average train accuracy of non-private K-means model: {np.mean(nonprivate_train_accs)}")
    print(f"Average train accuracy of private K-means model: {np.mean(private_train_accs)}")
