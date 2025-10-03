import numpy as np
import torch
from neuralnet import get_CIFAR10, train_model, test
from sys import argv

if __name__ == "__main__":
    train_loader, test_loader = get_CIFAR10()
    print("CIFAR-10 loaded successfully!")

    SAVE_TRAINED_MODELS = True
    SAVE_NUMBER = int(argv[1]) # Used for adding additional models when some have already been trained.

    TRAIN_SET_SIZE = 0.5
    train_loader, test_loader, train_indices = get_CIFAR10(train_set_size=TRAIN_SET_SIZE, test_set_is_leftover_train=(TRAIN_SET_SIZE < 1.0))
    
    nonprivate_model = train_model(train_loader, epochs=2, private=False) # train non-private model

    eps = 1.0
    private_model = train_model(train_loader, epochs=100, eps=eps) # train private model
    

    if SAVE_TRAINED_MODELS:
        np.save(f"models/train_set_{SAVE_NUMBER}.npy", train_indices)
        torch.save(nonprivate_model.state_dict(), f"models/nonprivate_resnet18_{SAVE_NUMBER}.pt")
        torch.save(private_model.state_dict(), f"models/private_resnet18_{SAVE_NUMBER}.pt")



