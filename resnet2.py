# import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score


class ResNetClassifier(nn.Module):
    def __init__(self, input_shape, nb_classes):
        super(ResNetClassifier, self).__init__()

        n_feature_maps = 32

        # BLOCK 1
        self.bn1 = nn.BatchNorm1d(input_shape[1])
        self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=n_feature_maps, kernel_size=32, padding='same')
        self.dropout1 = nn.Dropout(p=0.5)

        self.conv2 = nn.Conv1d(in_channels=n_feature_maps, out_channels=n_feature_maps, kernel_size=32, padding='same')
        self.bn2 = nn.BatchNorm1d(n_feature_maps)
        self.dropout2 = nn.Dropout(p=0.5)

        self.shortcut1 = nn.Conv1d(in_channels=input_shape[1], out_channels=n_feature_maps, kernel_size=1)

        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # BLOCK 2
        self.bn3 = nn.BatchNorm1d(n_feature_maps)
        self.conv3 = nn.Conv1d(in_channels=n_feature_maps, out_channels=n_feature_maps * 2, kernel_size=16, padding='same')
        self.dropout3 = nn.Dropout(p=0.5)

        self.bn4 = nn.BatchNorm1d(n_feature_maps * 2)
        self.conv4 = nn.Conv1d(in_channels=n_feature_maps * 2, out_channels=n_feature_maps * 2, kernel_size=16, padding='same')
        self.dropout4 = nn.Dropout(p=0.5)

        self.shortcut2 = nn.Conv1d(in_channels=n_feature_maps, out_channels=n_feature_maps * 2, kernel_size=1)

        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # BLOCK 3
        self.bn5 = nn.BatchNorm1d(n_feature_maps * 2)
        self.conv5 = nn.Conv1d(in_channels=n_feature_maps * 2, out_channels=n_feature_maps * 2 * 2, kernel_size=8, padding='same')
        self.dropout5 = nn.Dropout(p=0.5)

        self.bn6 = nn.BatchNorm1d(n_feature_maps * 2 * 2)
        self.conv6 = nn.Conv1d(in_channels=n_feature_maps * 2 * 2, out_channels=n_feature_maps * 2 * 2, kernel_size=8, padding='same')
        self.dropout6 = nn.Dropout(p=0.5)

        self.shortcut3 = nn.Conv1d(in_channels=n_feature_maps * 2, out_channels=n_feature_maps * 2 * 2, kernel_size=1)

        self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # FINAL
        self.gap_layer = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(n_feature_maps * 2 * 2, 256)
        self.dropout7 = nn.Dropout(p=0.25)

        self.fc2 = nn.Linear(256, 128)
        self.dropout8 = nn.Dropout(p=0.25)

        self.fc3 = nn.Linear(128, 128)
        self.dropout9 = nn.Dropout(p=0.25)

        self.output_layer = nn.Linear(128, nb_classes)

    def forward(self, x):
        # BLOCK 1

        conv_x = self.conv1(F.relu(self.bn1(x)))

        conv_x = self.dropout1(conv_x)

        conv_y = self.conv2(F.relu(self.bn2(conv_x)))
        conv_y = self.dropout2(conv_y)

        shortcut_y = self.shortcut1(self.bn1(x))

        output_block_1 = F.relu(conv_y + shortcut_y)
        output_block_1 = self.maxpool1(output_block_1)

        # BLOCK 2
        conv_x = self.conv3(F.relu(self.bn3(output_block_1)))
        conv_x = self.dropout3(conv_x)

        conv_y = self.conv4(F.relu(self.bn4(conv_x)))
        conv_y = self.dropout4(conv_y)

        shortcut_y = self.shortcut2(self.bn3(output_block_1))
        output_block_2 = F.relu(conv_y + shortcut_y)
        output_block_2 = self.maxpool2(output_block_2)

        # BLOCK 3
        conv_x = self.conv5(F.relu(self.bn5(output_block_2)))
        conv_x = self.dropout5(conv_x)

        conv_y = self.conv6(F.relu(self.bn6(conv_x)))
        conv_y = self.dropout6(conv_y)

        shortcut_y = self.shortcut3(self.bn5(output_block_2))
        output_block_3 = F.relu(conv_y + shortcut_y)
        output_block_3 = self.maxpool3(output_block_3)

        # FINAL
        gap_layer = self.gap_layer(output_block_3)

        output_1 = F.relu(self.fc1(torch.squeeze(gap_layer)))
        output_1 = self.dropout7(output_1)

        output_2 = F.relu(self.fc2(output_1))
        output_2 = self.dropout8(output_2)

        output_3 = F.relu(self.fc3(output_2))
        output_3 = self.dropout9(output_3)

        final_output = self.output_layer(output_3)

        return F.softmax(final_output, dim=1)


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
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        # batch
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets.long())
            loss.backward()
            optimizer.step()

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

                outputs = model(inputs)
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