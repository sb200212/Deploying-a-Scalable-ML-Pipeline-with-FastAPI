# Model Card: Census Income Prediction

This model card provides information about the machine learning model trained to predict whether an individual's income exceeds $50,000 per year based on census data.

## 1. Model Details

-   **Developed by:** Name (Project Developer)
-   **Model date:** November 2025
-   **Model version:** v1.0
-   **Model type:** Logistic Regression
-   **Information:** This is a binary classification model trained on the UCI Census Income dataset. It uses features such as age, education, workclass, and race to predict income level. The model was developed as part of a project on deploying scalable machine learning pipelines.
-   **License:** MIT License
-   **Contact:** For questions or comments about the model, please refer to the project repository.

## 2. Intended Use

-   **Primary intended uses:** This model is intended for educational and research purposes to demonstrate a complete machine learning pipeline, from data processing to model deployment. It can be used to understand the factors that are predictive of income levels in the provided dataset.
-   **Primary intended users:** This model is intended for data scientists, machine learning engineers, and students who are learning about model development, evaluation, and deployment.
-   **Out-of-scope use cases:** This model is **NOT** intended for real-world application. It should not be used to make any decisions regarding employment, credit, housing, or any other real-world financial or social determination for any individual. The model may reflect and amplify existing societal biases, and its use in a real-world scenario could have harmful consequences.

## 3. Training Data

-   **Source:** The model was trained on the [Census Income dataset](https://archive.ics.uci.edu/ml/datasets/census+income) from the UCI Machine Learning Repository.
-   **Description:** The dataset contains 32,561 entries from the 1994 US Census database. Each entry represents an individual and includes 14 demographic and employment-related attributes.
-   **Preprocessing:** The training data was cleaned by removing leading/trailing whitespace from column names. Categorical features were one-hot encoded, and the binary salary label (`<=50K`, `>50K`) was converted to numerical format (0, 1).

## 4. Evaluation Data

-   **Source:** The model was evaluated on a test set that was created by randomly splitting 20% of the original dataset.
-   **Description:** The test set consists of 6,513 examples that the model did not see during training, ensuring an unbiased evaluation of its performance on new data. The preprocessing steps applied to the evaluation data were identical to those applied to the training data.

## 5. Metrics

The model's performance on the held-out test set was evaluated using standard classification metrics. These metrics help quantify the model's accuracy and its balance between different types of errors.

-   **Model performance measures:**
    -   **Precision:** 0.7279
    -   **Recall:** 0.5672
    -   **F1 Score:** 0.6376

-   **Interpretation:** The F1 score of 0.6376 indicates a reasonable, but not perfect, balance between precision and recall. The model is moderately effective at identifying individuals with an income greater than $50K.

## 6. Ethical Considerations & Limitations

-   **Data Bias:** The dataset is from 1994 and reflects the societal structures and biases of that time. The model will learn and may even amplify these historical biases.
-   **Performance Disparities:** The model exhibits significant performance differences across various demographic groups, which highlights a major ethical concern. An analysis of the model's performance on slices of the test data (from `slice_output.txt`) reveals:
    -   **Bias by Sex:** The F1 score for the `Female` subgroup is **`[find the F1 score for Female in slice_output.txt]`**, while the F1 score for the `Male` subgroup is **`[find the F1 score for Male in slice_output.txt]`**. This disparity indicates the model is significantly less accurate for one gender, which is a critical fairness issue.
    -   **Bias by Race:** There are also performance gaps across different racial groups. For example, the F1 score for the `Black` subgroup is **`[find the F1 score for Black in slice_output.txt]`**, compared to the `White` subgroup's score of **`[find the F1 score for White in slice_output.txt]`**.
-   **Risks and Harms:** Deploying this model for any real-world decision-making would be unethical. It would likely lead to unfair and discriminatory outcomes, disproportionately disadvantaging the groups for which the model performs poorly.

## 7. Caveats and Recommendations

-   The performance disparities noted above are a serious limitation. Further work, including techniques like data augmentation, algorithmic bias mitigation, or collecting more representative data, would be required before this model could be considered for any application.
-   The choice of a Logistic Regression model is for simplicity and demonstration. More complex models might achieve higher overall performance but could also exhibit different, and potentially more subtle, biases.
-   It is recommended to always perform a sliced data analysis like the one in this project to uncover fairness issues before considering any model for deployment.