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
model_spec = resnet2  # chose between resnet and resnet2
num_epochs = 50


def extract_features(model, X, y):
    # create a data loader for X
    tensor_X = torch.Tensor(X)
    tensor_y = torch.Tensor(y)
    X_loader = TensorDataset(tensor_X, tensor_y)
    X_loader = DataLoader(X_loader, batch_size=256)

    # prediction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()  # changes how different layers behave, e.g. dropout

    correct_val = 0
    total_val = 0
    f1 = 0
    acc_balanced = 0
    total_batches = 0
    features_list = []

    with torch.no_grad():
        for inputs, targets in X_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # moves the data to the same device as the model
            total_batches += 1

            outputs, features = model(inputs)
            features_list.append(features.cpu().numpy())

            _, predicted = torch.max(outputs.data, 1)
            total_val += targets.size(0)
            correct_val += (predicted == targets).sum().item()
            f1 += f1_score(targets, predicted, average='weighted')
            acc_balanced += balanced_accuracy_score(targets, predicted)
            total_batches += 1

    f1 = f1 / total_batches
    acc_balanced = acc_balanced / total_batches

    print("accuracy:", f1)
    print("weighted accuracy:", acc_balanced)

    features_np = np.stack(features_list[:-1], axis=0)
    num_samples, batch_size, num_features = features_np.shape
    combined_features = np.reshape(features_np, (num_samples * batch_size, num_features))

    print("feature shape: ", combined_features.shape)

    # save features
    np.save(root + "features", combined_features)


def read_data():
    # subject numbers
    subj_nums = [1595, 1105, 1106, 1241, 1271, 1314, 1323, 1337, 1372, 1417, 1434, 1544, 1547, 1953, 1629, 1716, 1717,
                 1744, 1868, 1892, 1030]

    # samples per subject per diff_level
    samples_per_subj = int(overall_duration / sample_duration)  # 18

    # initialize dataset
    X = np.zeros((0, sampling_rate * sample_duration, 4))  # shape of 3D matrix X should be (3138, 2560, 4)
    y = np.zeros(0)

    # Read the CSV file using Pandas
    for k in range(len(subj_nums)):
        print("read_subj:", k + 1)
        for i in range(diff_levels):

            # missing data for subject 1629, 1744, 1868
            if k == 14 and (i == 5 or i == 8):
                continue
            if k == 17 and (i == 6 or i == 7 or i == 8):
                continue
            if k == 18 and (i == 7 or i == 8):
                continue

            # read csv file
            csv_file = root + "EEG/" + str(subj_nums[k]) + "/eeg_data_level_" + str(i + 1) + ".csv"
            label_file = root + "Labels/" + str(subj_nums[k]) + ".csv"
            data = pd.read_csv(csv_file)
            y_data = pd.read_csv(label_file)

            # append y
            y_add = np.array(y_data)[:, i + 1]

            # change labels to three class classification
            mask_0 = (y_add <= 3)
            mask_1 = ((y_add >= 4) & (y_add <= 6))
            mask_2 = (y_add >= 7)
            y_add[mask_0] = 0
            y_add[mask_1] = 1
            y_add[mask_2] = 2

            # exclude the timestamp
            np_data = np.array(data)[:, 1:]

            if len(np_data > sampling_rate * overall_duration):
                np_data = np_data[:sampling_rate * overall_duration]  # cut the signal at 180sec

            # append data point to the matrix X
            for j in range(samples_per_subj):
                start_np = j * sampling_rate * sample_duration
                append_data_point = np_data[start_np:start_np + sampling_rate * sample_duration, :]
                if len(append_data_point) == sampling_rate * sample_duration:  # in case when signal < 180sec
                    y = np.append(y, y_add[j])
                    X = np.concatenate((X, np.expand_dims(append_data_point, axis=0)), axis=0)

    # impute missing data
    n_samples = X.shape[0]
    X_2d = X.reshape((n_samples, -1))
    impute = KNNImputer(n_neighbors=50)
    X_2d = impute.fit_transform(X_2d)

    # scaling
    scaler = StandardScaler()
    scaler.fit_transform(X_2d)

    X = X_2d.reshape(X.shape)

    # display data
    print("data: ", X)

    # save matrix
    np.save(root + "X_numpy", X)
    np.save(root + "y_numpy", y)

    print("Data read")


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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=174, random_state=0, shuffle=False)
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
    if model_spec == resnet:
        model = resnet.ResNetClassifier(X_train.shape, nb_classes)
    if model_spec == resnet2:
        model = resnet2.ResNetClassifier(X_train.shape, nb_classes)
    else:
        print("Model not specified")

    print("model summary: ", model)

    # training
    sample_weight = compute_class_weight(class_weight="balanced", classes=np.array([0, 1, 2]), y=y_train)

    if model_spec == resnet:
        resnet.train(model=model, train_loader=train_loader, val_loader=test_loader, num_epochs=num_epochs, lr=0.001,
                    sample_weight=sample_weight, root=root)
        torch.save(model.state_dict(), root + "final_model.pth")

    if model_spec == resnet2:
        resnet2.train(model=model, train_loader=train_loader, val_loader=test_loader, num_epochs=num_epochs, lr=0.001,
                    sample_weight=sample_weight, root=root)
        torch.save(model.state_dict(), root + "final_model.pth")

    # extract features
    # extract_features(model, X, y)

    print("DONE")
