import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin

# Add necessary imports from your project
from ml.data import process_data
from ml.model import train_model, inference

# Define categorical features once for use in multiple tests
CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

@pytest.fixture(scope="session")
def data():
    """
    A pytest fixture to load the census data once and make it available to all tests.
    This is more efficient than loading the data file in each test function.
    """
    try:
        df = pd.read_csv("data/census.csv")
        # Clean column names as done in the training script
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        pytest.fail(
            "The dataset 'data/census.csv' was not found. "
            "Please ensure the file is present in the 'data' directory."
        )


def test_process_data_output_shape(data):
    
    
    sample_data = data.sample(n=100, random_state=42)

    X, y, _, _ = process_data(
        sample_data,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True
    )

    # Check that the number of rows in the output matches the input
    assert X.shape[0] == len(sample_data), "The number of rows in X does not match the input."
    assert len(y) == len(sample_data), "The length of y does not match the input."


def test_train_model_returns_classifier(data):
    
    # Create a small training set to speed up the test
    train_df, _ = train_test_split(data, test_size=0.9, random_state=42)
    X_train, y_train, _, _ = process_data(
        train_df,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True
    )

    model = train_model(X_train, y_train)

    # Check if the returned object is a scikit-learn classifier
    assert isinstance(model, ClassifierMixin), "The train_model function should return a scikit-learn classifier."


def test_inference_output_length(data):
    
    # Create train and test sets
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

    # Process both sets
    X_train, y_train, encoder, lb = process_data(
        train_df,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True
    )
    X_test, _, _, _ = process_data(
        test_df,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Train a model and make predictions
    model = train_model(X_train, y_train)
    predictions = inference(model, X_test)

    # Check that the number of predictions matches the number of test examples
    assert len(predictions) == len(test_df), "The number of predictions should match the number of input rows."