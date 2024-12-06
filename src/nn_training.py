import json
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def load_data(features_file, labels_file):
    with open(features_file, 'r') as f:
        features_data = json.load(f)
    with open(labels_file, 'r') as f:
        labels_data = json.load(f)

    X, y = [], []
    max_features_length = 0

    for key, features in tqdm(features_data.items(), desc=f"processing {features_file}"):
        left_features = features.get('left', {})
        right_features = features.get('right', {})
        
        combined_features = []
        for hand_features in [left_features, right_features]:
            combined_features += hand_features.get('distances', [])
            combined_features += hand_features.get('angles', [])
            combined_features += hand_features.get('mass_center_distances', [])
        
        max_features_length = max(max_features_length, len(combined_features))

    for key, features in tqdm(features_data.items(), desc=f"processing {features_file}"):
        if key in labels_data:
            label = labels_data[key][0]  # using the first part of the label
            left_features = features.get('left', {})
            right_features = features.get('right', {})

            combined_features = []
            for hand_features in [left_features, right_features]:
                combined_features += hand_features.get('distances', [])
                combined_features += hand_features.get('angles', [])
                combined_features += hand_features.get('mass_center_distances', [])
            
            # pad combined_features to max_features_length with zeros
            padded_features = combined_features + [0] * (max_features_length - len(combined_features))
            
            X.append(padded_features)
            y.append(label)

    return np.array(X), np.array(y)

class PoseNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(PoseNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100):
    model.to(device)
    best_model_wts = None
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"epoch {epoch+1}/{num_epochs}")
        
        # train phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f"train Loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}")
        
        # validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        print(f"val loss: {val_loss:.4f} acc: {val_acc:.4f}")

        # save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()
    
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    data_dir = 'annotations'
    splits = ['train', 'val', 'test']
    datasets = {}

    for split in splits:
        features_file = os.path.join(data_dir, f"{split}/features.json")
        labels_file = os.path.join(data_dir, f"{split}/labels.json")
        print(f"loading {split} dataset...")
        X, y = load_data(features_file, labels_file)
        datasets[split] = (X, y)

    X_train, y_train = datasets['train']
    X_val, y_val = datasets['val']
    X_test, y_test = datasets['test']

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    y_test_enc = label_encoder.transform(y_test)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train_enc, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val_enc, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test_enc, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    input_size = X_train.shape[1]
    num_classes = len(label_encoder.classes_)

    model = PoseNN(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("training the neural network...")
    model = train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=25)

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "best_nn.pth")
    torch.save({"model_state_dict": model.state_dict(), "label_encoder": label_encoder}, model_path)
    print(f"model saved to {model_path}")

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_corrects = sum(np.array(all_preds) == np.array(all_labels))
    test_acc = test_corrects / len(test_loader.dataset)
    print(f"test acc: {test_acc:.4f}")
    print("\ntest:")
    print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))
