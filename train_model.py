import os
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

#  1. Load Data 
print("Loading data...")
try:
    data = pd.read_csv("data/census.csv")
    data.columns = data.columns.str.strip()
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'data/census.csv' not found. Make sure the file exists in the 'data' folder.")
    exit()


#  2. Split Data 
print("Splitting data...")

train, test = train_test_split(data, test_size=0.20, random_state=42)
print(f"Train set size: {len(train)}, Test set size: {len(test)}")


# DO NOT MODIFY (Categorical features list)
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

#  3. Process Data 
print("Processing training data...")
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

print("Processing testing data...")
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

#  4. Train Model 
print("Training model...")
# Use the train_model function to train the model on the training dataset
model = train_model(X_train, y_train)
print("Model training complete.")


#  5. Save Artifacts 
print("Saving model and data processors...")
# Save the model, encoder, and label binarizer
save_model(model, "model/model.pkl")
save_model(encoder, "model/encoder.pkl")
save_model(lb, "model/lb.pkl")
print("Artifacts saved to 'model/' directory.")


#  6. Load Model and Run Inference 
# Load the model (this is redundant but follows the template's structure)
model = load_model("model/model.pkl")

# Use the inference function to run the model inferences on the test dataset.
preds = inference(model, X_test)


#  7. Evaluate Overall Performance 
print("\n Overall Model Performance ")
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")


#  8. Evaluate on Slices 
print("\n Sliced Model Performance ")
# Clear the output file if it exists
if os.path.exists("slice_output.txt"):
    os.remove("slice_output.txt")

# Compute the performance on model slices
for col in cat_features:
    with open("slice_output.txt", "a") as f:
        f.write(f"\n Metrics for feature: {col} \n")

    # Iterate through the unique values in one categorical feature
    for slice_value in sorted(test[col].unique()):
        count = test[test[col] == slice_value].shape[0]

        # Call the slice performance function
        p, r, fb = performance_on_categorical_slice(
            model,
            test,
            col,
            slice_value,
            "salary",
            encoder,
            lb,
            process_data, # Pass the function itself
            cat_features
        )
        with open("slice_output.txt", "a") as f:
            line1 = f"{col}: {slice_value}, Count: {count:,}"
            line2 = f"  -> Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}"
            print(line1, file=f)
            print(line2, file=f)

print("Slice performance analysis complete. Results saved to 'slice_output.txt'.")