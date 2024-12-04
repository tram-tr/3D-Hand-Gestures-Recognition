import json
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from tqdm import tqdm
import os
import pickle

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

def train(X_train, y_train, X_val, y_val):
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(probability=True, random_state=42))
    ])

    print("training SVM model...")
    pipeline.fit(X_train, y_train_enc)

    y_val_pred = pipeline.predict(X_val)
    print("val:")
    print(classification_report(y_val_enc, y_val_pred, target_names=label_encoder.classes_))

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "svm_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump((pipeline, label_encoder), f)
    print(f"model saved to {model_path}")

    return pipeline, label_encoder

def evaluate_model(model, X_test, y_test, label_encoder):
    y_test_enc = label_encoder.transform(y_test)
    y_test_pred = model.predict(X_test)

    print("test:")
    print(classification_report(y_test_enc, y_test_pred, target_names=label_encoder.classes_))

if __name__ == '__main__':
    data_dir = 'annotations'
    splits = ['train', 'val', 'test']
    datasets = {}

    for split in splits:
        features_file = os.path.join(data_dir, f"{split}/features.json")
        labels_file = os.path.join(data_dir, f"{split}/labels.json")
        print(f"Loading {split} dataset...")
        X, y = load_data(features_file, labels_file)
        datasets[split] = (X, y)

    X_train, y_train = datasets['train']
    X_val, y_val = datasets['val']
    X_test, y_test = datasets['test']

    clf, label_encoder = train(X_train, y_train, X_val, y_val)

    evaluate_model(clf, X_test, y_test, label_encoder)
