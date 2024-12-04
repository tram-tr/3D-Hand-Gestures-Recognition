import json
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
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
        ('clf', XGBClassifier(use_label_encoder=False, eval_metric='mse', random_state=42))
    ])

    param_grid = {
        'clf__n_estimators': [50, 100, 200],
        'clf__learning_rate': [0.01, 0.1, 0.2],
        'clf__max_depth': [3, 5, 7],
        'clf__subsample': [0.8, 1.0],
        'clf__colsample_bytree': [0.8, 1.0]
    }

    print('hyperparameter tuning...')
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        scoring='accuracy', 
        cv=5, 
        verbose=2, 
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train_enc)
    best_model = grid_search.best_estimator_
    y_val_pred = best_model.predict(X_val)
    print("val:")
    print(classification_report(y_val_enc, y_val_pred, target_names=label_encoder.classes_))
    print(f"best hyperparameters: {grid_search.best_params_}")

    model_name = 'best_xgboost.pkl'
    model_dir = 'models'
    save_path = os.join(model_dir, model_name)
    with open(save_path, "wb") as f:
        pickle.dump((best_model, label_encoder), f)
    print(f"model saved to {save_path}")

    return best_model, label_encoder

def evaluate_model(model, X_test, y_test, label_encoder):
    y_test_enc = label_encoder.transform(y_test)
    y_test_pred = model.predict(X_test)
    print("test:")
    print(classification_report(y_test_enc, y_test_pred, target_names=label_encoder.classes_))
    return y_test_pred

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

    clf, label_encoder = train(X_train, y_train, X_val, y_val)

    y_test_pred = evaluate_model(clf, X_test, y_test, label_encoder)


    
