# import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import f1_score, balanced_accuracy_score


class ResNetClassifier(nn.Module):
    def __init__(self, input_shape, nb_classes):
        super(ResNetClassifier, self).__init__()

        n_feature_maps = 32  # hidden dimension

        dropout_rate = 0.4

        # block 1
        self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=n_feature_maps, kernel_size=8, padding='same')
        self.bn1 = nn.BatchNorm1d(n_feature_maps)
        self.dropout1 = nn.Dropout(p=dropout_rate)

        self.conv2 = nn.Conv1d(in_channels=n_feature_maps, out_channels=n_feature_maps, kernel_size=5, padding='same')
        self.bn2 = nn.BatchNorm1d(n_feature_maps)
        self.dropout2 = nn.Dropout(p=dropout_rate)

        self.conv3 = nn.Conv1d(in_channels=n_feature_maps, out_channels=n_feature_maps, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm1d(n_feature_maps)
        self.shortcut1 = nn.Conv1d(in_channels=input_shape[1], out_channels=n_feature_maps, kernel_size=1,
                                   padding='same')

        # block 2
        self.conv4 = nn.Conv1d(in_channels=n_feature_maps, out_channels=n_feature_maps * 2, kernel_size=8,
                               padding='same')
        self.bn4 = nn.BatchNorm1d(n_feature_maps * 2)
        self.dropout3 = nn.Dropout(p=dropout_rate)

        self.conv5 = nn.Conv1d(in_channels=n_feature_maps * 2, out_channels=n_feature_maps * 2, kernel_size=5,
                               padding='same')
        self.bn5 = nn.BatchNorm1d(n_feature_maps * 2)
        self.dropout4 = nn.Dropout(p=dropout_rate)

        self.conv6 = nn.Conv1d(in_channels=n_feature_maps * 2, out_channels=n_feature_maps * 2, kernel_size=3,
                               padding='same')
        self.bn6 = nn.BatchNorm1d(n_feature_maps * 2)
        self.shortcut2 = nn.Conv1d(in_channels=n_feature_maps, out_channels=n_feature_maps * 2, kernel_size=1,
                                   padding='same')

        # block 3
        self.conv7 = nn.Conv1d(in_channels=n_feature_maps * 2, out_channels=n_feature_maps * 2, kernel_size=8,
                               padding='same')
        self.bn7 = nn.BatchNorm1d(n_feature_maps * 2)
        self.dropout5 = nn.Dropout(p=dropout_rate)

        self.conv8 = nn.Conv1d(in_channels=n_feature_maps * 2, out_channels=n_feature_maps * 2, kernel_size=5,
                               padding='same')
        self.bn8 = nn.BatchNorm1d(n_feature_maps * 2)
        self.dropout6 = nn.Dropout(p=dropout_rate)

        self.conv9 = nn.Conv1d(in_channels=n_feature_maps * 2, out_channels=n_feature_maps * 2, kernel_size=3,
                               padding='same')
        self.bn9 = nn.BatchNorm1d(n_feature_maps * 2)

        # final layer
        self.gap_layer = nn.AdaptiveAvgPool1d(1)  # Global Averaging Pooling layer over the 64 channels, 1280 length
        self.fc = nn.Linear(1280, nb_classes)  # FC

    def forward(self, x):
        # block 1
        conv_x = F.relu(self.bn1(self.conv1(x)))
        conv_x = self.dropout1(conv_x)

        conv_y = F.relu(self.bn2(self.conv2(conv_x)))
        conv_y = self.dropout2(conv_y)

        conv_z = F.relu(self.bn3(self.conv3(conv_y)))
        shortcut_y = self.bn1(self.shortcut1(x))
        output_block_1 = F.relu(conv_z + shortcut_y)

        # block 2
        conv_x = F.relu(self.bn4(self.conv4(output_block_1)))
        conv_x = self.dropout3(conv_x)

        conv_y = F.relu(self.bn5(self.conv5(conv_x)))
        conv_y = self.dropout4(conv_y)

        conv_z = F.relu(self.bn6(self.conv6(conv_y)))
        shortcut_y = self.bn4(self.shortcut2(output_block_1))
        output_block_2 = F.relu(conv_z + shortcut_y)

        # block 3
        conv_x = F.relu(self.bn7(self.conv7(output_block_2)))
        conv_x = self.dropout5(conv_x)

        conv_y = F.relu(self.bn8(self.conv8(conv_x)))
        conv_y = self.dropout6(conv_y)

        conv_z = F.relu(self.bn9(self.conv9(conv_y)))
        shortcut_y = self.bn7(output_block_2)
        output_block_3 = F.relu(conv_z + shortcut_y)

        # final layer
        gap_layer = self.gap_layer(output_block_3.transpose(1, 2)).transpose(1, 2)  # apply GAP
        gap_layer = gap_layer.view(gap_layer.size(0), -1)
        output = self.fc(torch.squeeze(gap_layer))
        return F.softmax(output, dim=1), gap_layer


def train(model, train_loader, val_loader, sample_weight, root, num_epochs=50, lr=0.001):

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(sample_weight).float())  # use  weighted cross entropy loss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use gpu if possible
    model.to(device)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()  # changes to train mode
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        # batch
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()  # resets the stored gradients

            outputs, _ = model(inputs)

            loss = criterion(outputs, targets.long())
            loss.backward()  # computes the gradient
            optimizer.step()  # perform parameter update

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

        train_losses.append(train_loss / len(train_loader))
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)

        model.eval()

        val_loss = 0.0
        correct_val = 0
        total_val = 0
        f1 = 0
        acc_balanced = 0
        total_batches = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                total_batches += 1

                outputs, _ = model(inputs)
                loss = criterion(outputs, targets.long())

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()

                f1 += f1_score(targets, predicted, average='weighted')
                acc_balanced += balanced_accuracy_score(targets, predicted)  # weighted accuracy

        val_losses.append(val_loss / len(val_loader))
        val_accuracy = 100 * correct_val / total_val
        val_accuracies.append(val_accuracy)
        f1 = f1 / total_batches
        acc_balanced = acc_balanced / total_batches

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracy:.2f}"
            f"%, Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracy:.2f}%, weighted f1-score: {f1},"
            f"weighted accuracy: {acc_balanced}")

        # model checkpoint
        torch.save(model.state_dict(), root + str(epoch) + "_model.pth")

    return train_losses, val_losses, train_accuracies, val_accuracies
