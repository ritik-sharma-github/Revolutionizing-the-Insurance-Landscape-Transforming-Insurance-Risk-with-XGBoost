Project Overview
Objective: The goal of this project is to leverage advanced machine learning techniques to enhance the risk assessment process in the insurance industry. Specifically, we use an XGBoost classifier to predict the likelihood of a customer making an insurance claim based on various features, and optimize the model's performance through hyperparameter tuning.

Key Steps and Techniques:

Data Preparation:

Feature Selection: Identified and selected relevant features for the model, including 'Gender', 'Age', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', and 'Vintage'.
One-Hot Encoding: Transformed categorical variables into dummy variables to prepare the data for modeling. Handled any discrepancies in columns between training and test datasets by reindexing.
Model Training:

XGBoost Classifier: Utilized the XGBoost algorithm, known for its robustness and performance in classification tasks. Trained the model on the training dataset to learn patterns and make predictions.
Performance Evaluation:

Initial Metrics: Assessed the model using accuracy, precision, recall, F1 score, and ROC AUC score. The initial results showed an accuracy of 0.85 and an ROC AUC score of 0.90, indicating a strong model performance.
Confusion Matrix Analysis: Analyzed the confusion matrix to understand the distribution of predictions and misclassifications.
Feature Importance Analysis:

Visualization: Plotted the importance of features to identify which variables contributed most to the model's predictions. This helps in understanding the model's decision-making process and refining feature selection.
Hyperparameter Tuning:

Grid Search: Employed GridSearchCV to find the optimal hyperparameters for the XGBoost model. Evaluated different combinations of parameters to improve model performance.
Best Parameters: Found the best hyperparameters and retrained the model, achieving similar performance metrics (accuracy: 0.85, ROC AUC score: 0.90) but with potentially improved robustness.
Model Persistence and Deployment:

Saving and Loading Model: Saved the trained model to a file for future use and loaded it to make predictions on new data.
Prediction and Simulation:

User Input Predictions: Created a function to collect user inputs, convert them into the required format, and predict insurance claim probabilities.
Premium Simulation: Simulated varying annual premium values to assess their impact on the model's predicted probabilities.
