**Objective:**
The goal of this project is to leverage advanced machine learning techniques to enhance the risk assessment process in the insurance industry. Specifically, we use an XGBoost classifier to predict the likelihood of a customer making an insurance claim based on various features, and optimize the model's performance through hyperparameter tuning.

**Key Steps and Techniques:**

1. **Data Preparation:**
   - **Feature Selection:** Identified and selected relevant features for the model, including 'Gender', 'Age', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', and 'Vintage'.
   - **One-Hot Encoding:** Transformed categorical variables into dummy variables to prepare the data for modeling. Handled any discrepancies in columns between training and test datasets by reindexing.

2. **Model Training:**
   - **XGBoost Classifier:** Utilized the XGBoost algorithm, known for its robustness and performance in classification tasks. Trained the model on the training dataset to learn patterns and make predictions.

3. **Performance Evaluation:**
   - **Initial Metrics:** Assessed the model using accuracy, precision, recall, F1 score, and ROC AUC score. The initial results showed an accuracy of 0.85 and an ROC AUC score of 0.90, indicating a strong model performance.
   - **Confusion Matrix Analysis:** Analyzed the confusion matrix to understand the distribution of predictions and misclassifications.

4. **Feature Importance Analysis:**
   - **Visualization:** Plotted the importance of features to identify which variables contributed most to the model's predictions. This helps in understanding the model's decision-making process and refining feature selection.

5. **Hyperparameter Tuning:**
   - **Grid Search:** Employed GridSearchCV to find the optimal hyperparameters for the XGBoost model. Evaluated different combinations of parameters to improve model performance.
   - **Best Parameters:** Found the best hyperparameters and retrained the model, achieving similar performance metrics (accuracy: 0.85, ROC AUC score: 0.90) but with potentially improved robustness.
![download](https://github.com/user-attachments/assets/6e5622f8-ff5c-4713-82cb-5ede4969c384)
![download](https://github.com/user-attachments/assets/8833a0fb-a9c2-4e29-9d9e-b4ec50db5ef2)

![download](https://github.com/user-attachments/assets/528ec275-5538-417d-9838-dc18ee29d1eb)

![download](https://github.com/user-attachments/assets/063eedd3-e9c8-41ec-bb92-cca44b70efdb)
![download](https://github.com/user-attachments/assets/80ba6c31-c80c-494a-98c7-5826dc13eb95)

6. **Model Persistence and Deployment:**
7. ![download](https://github.com/user-attachments/assets/9d545f17-b765-4ea7-b166-da05e1a08cd8)

   - **Saving and Loading Model:** Saved the trained model to a file for future use and loaded it to make predictions on new data.

8. **Prediction and Simulation:**
   - **User Input Predictions:** Created a function to collect user inputs, convert them into the required format, and predict insurance claim probabilities.
   - **Premium Simulation:** Simulated varying annual premium values to assess their impact on the model's predicted probabilities.
