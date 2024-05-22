# import packages
import pandas as pd
import numpy as np
import resnet
import resnet2
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, balanced_accuracy_score

# import torch
import torch
from torch.utils.data import TensorDataset, DataLoader

# parameters
root = "/Users/leluwy/Desktop/ETH/AICenterProjects/our_dataset/"
sampling_rate = 256
overall_duration = 180
sample_duration = 10
diff_levels = 9
nb_classes = 3

# define model
model_spec = 'resnet2'  # chose between resnet and resnet2
num_epochs = 50

# main
if __name__ == '__main__':
    # read_data()

    # read data from file
    X = np.load(root + "X_numpy.npy")
    y = np.load(root + "y_numpy.npy")

    # transpose X
    # X = np.transpose(X, axes=(0, 2, 1))

    # data shape
    print("data shape: ", X.shape)
    print("data shape: ", y.shape)

    # class distribution
    print("class 0: ", np.sum(y == 0))
    print("class 1: ", np.sum(y == 1))
    print("class 2: ", np.sum(y == 2))

    weights = [np.sum(y == 0), np.sum(y == 1), np.sum(y == 2)]

    # train-test-split (174 = 2 test size *3 data points per paragraph * 29 paragraphs)
    X_train = X[174:]
    y_train = y[174:]
    X_test = X[:174]
    y_test = y[:174]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=174, random_state=0, shuffle=False)
    print("number of test_samples: ", len(y_test))

    # test class distribution
    print("class 0: ", np.sum(y_test == 0))
    print("class 1: ", np.sum(y_test == 1))
    print("class 2: ", np.sum(y_test == 2))

    # convert to pytorch tensors
    tensor_x_train = torch.Tensor(X_train)
    tensor_y_train = torch.Tensor(y_train)
    tensor_x_test = torch.Tensor(X_test)
    tensor_y_test = torch.Tensor(y_test)

    # convert to DataLoader
    train_loader = TensorDataset(tensor_x_train, tensor_y_train)
    train_loader = DataLoader(train_loader, batch_size=64, shuffle=True)

    test_loader = TensorDataset(tensor_x_test, tensor_y_test)
    test_loader = DataLoader(test_loader, batch_size=64, shuffle=True)

    # model
    if model_spec == 'resnet':
        model = resnet.ResNetClassifier(X_train.shape, nb_classes)
    if model_spec == 'resnet2':
        model = resnet2.ResNetClassifier(X_train.shape, nb_classes)

    print("model summary: ", model)

    # training
    sample_weight = compute_class_weight(class_weight="balanced", classes=np.array([0, 1, 2]), y=y_train)

    if model_spec == 'resnet':
        resnet.train(model=model, train_loader=train_loader, val_loader=test_loader, num_epochs=num_epochs, lr=0.001,
                    sample_weight=sample_weight, root=root)
        torch.save(model.state_dict(), root + "final_model.pth")

    if model_spec == 'resnet2':
        resnet2.train(model=model, train_loader=train_loader, val_loader=test_loader, num_epochs=num_epochs, lr=0.001,
                    sample_weight=sample_weight, root=root)
        torch.save(model.state_dict(), root + "final_model.pth")

    # extract features
    # extract_features(model, X, y)

    print("DONE")
