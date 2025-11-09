import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score


from .data import process_data


def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    
    f1 = f1_score(y, preds)
    precision = precision_score(y, preds)
    recall = recall_score(y, preds)
    return precision, recall, f1


def inference(model, X):
    
    preds = model.predict(X)
    return preds




def save_model(artifact, file_path):

    joblib.dump(artifact, file_path)


def load_model(file_path):

    return joblib.load(file_path)


# IN ml/model.py
# REPLACE your old function with this one.

def performance_on_categorical_slice(
    model, data, feature, slice_value, label, encoder, lb, process_data_func, categorical_features
):
    
    
    # Create a slice of the data for the current class/value
    slice_df = data[data[feature] == slice_value]

    # Process this slice using the provided function and fitted encoders
    X_slice, y_slice, _, _ = process_data_func(
        slice_df,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Handle cases where the slice is empty (e.g., contains only unknown categories)
    if X_slice.shape[0] == 0:
        return 0.0, 0.0, 0.0

    # Make predictions on the slice
    preds_slice = inference(model, X_slice)

    # Compute metrics for the slice.
    # zero_division=0.0 ensures that if there are no positive predictions, precision is 0 instead of erroring.
    precision = precision_score(y_slice, preds_slice, zero_division=0.0)
    recall = recall_score(y_slice, preds_slice, zero_division=0.0)
    f1 = f1_score(y_slice, preds_slice, zero_division=0.0)

    return precision, recall, f1