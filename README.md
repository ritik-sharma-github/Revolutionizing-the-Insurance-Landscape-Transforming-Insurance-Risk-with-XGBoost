**Objective:**
The goal of this project is to enhance the risk assessment process in the insurance industry by predicting the likelihood of a customer making an insurance claim. This is achieved by employing an XGBoost classifier to analyze various features and optimize the model's performance through hyperparameter tuning.

**Problem Statement:**
Insurance companies need to accurately predict the probability of a customer making an insurance claim to manage risks and set appropriate premiums. Given various customer features such as age, driving history, and vehicle condition, the challenge is to build a predictive model that identifies high-risk customers who are more likely to file a claim.

### **Goal:**
The primary goal of this project is to develop a robust predictive model using XGBoost that can accurately forecast the likelihood of insurance claims. This involves:
- Exploring and understanding the data through exploratory data analysis (EDA).
- Preprocessing the data to prepare it for modeling.
- Applying the XGBoost algorithm to train and predict outcomes.
- Evaluating the model's performance and optimizing it through hyperparameter tuning.

### **Modelling:**
1. **Data Preprocessing:**
   - **Handling Outliers:** Address any Outliers values in the dataset.

![download](https://github.com/user-attachments/assets/8521b19f-91c2-4465-81ca-b58be6dcc495)

   - **Feature Encoding:** Convert categorical variables into numerical values.
   - **Feature Scaling:** Normalize or standardize numerical features if needed.
   
1. **Modeling Approach:**
   - **Algorithm:** XGBoost (Extreme Gradient Boosting)
   - **Training:** Fit the XGBoost model on the training data.
   - **Hyperparameter Tuning:** Use techniques such as GridSearchCV or RandomizedSearchCV to find the optimal parameters for the model.

### **Exploratory Data Analysis (EDA):**
- **Feature Analysis:** Examine the distribution and relationship of each feature with the target variable.

- ![download](https://github.com/user-attachments/assets/a7ceb460-ba54-4e4e-b8f9-6d041120f137)

- **Correlation Analysis:** Identify correlations between features to understand their impact on the target variable.

![download](https://github.com/user-attachments/assets/a6f01eb8-748c-4c83-8cb6-576d114f9c4b)


- **Visualization:** Use visualizations (e.g., histograms, scatter plots, heatmaps) to identify patterns and insights in the data.

![download](https://github.com/user-attachments/assets/2629f842-15bc-4486-97f0-86971519dd71)
![download](https://github.com/user-attachments/assets/c9a4ebaa-2f63-4c87-906b-c3783be0aa5d)
![download](https://github.com/user-attachments/assets/7a2145e7-4f45-4d8b-ade3-080da7ddb556)


### **Algorithm:**
- **XGBoost Classifier:** A gradient boosting framework that is known for its efficiency and performance. It works by creating an ensemble of decision trees to improve predictive accuracy.

### **Results:**
- **Model Performance Metrics:** Evaluate the model using metrics such as accuracy, precision, recall, F1-score, and AUC-ROC.

![download](https://github.com/user-attachments/assets/3c9ef791-3e6c-4b00-a1fd-f4939891000a)



- **Confusion Matrix:** Analyze the confusion matrix to understand the classification results and identify any misclassifications.

**Model Persistence and Deployment:**

   - **Saving and Loading Model:** Saved the trained model to a file for future use and loaded it to make predictions on new data.

**Prediction and Simulation:**
   - **User Input Predictions:** Created a function to collect user inputs, convert them into the required format, and predict insurance claim probabilities.

![Screenshot 2024-09-07 122542](https://github.com/user-attachments/assets/1ceb2d6b-ac0d-41db-ad25-637e586e391f)
![Screenshot 2024-09-07 122602](https://github.com/user-attachments/assets/326b3bc6-507f-4992-8ac3-de80201b123d)

   - 
   - **Premium Simulation:** Simulated varying annual premium values to assess their impact on the model's predicted probabilities.

   - The model has predicted that there is a 96.02% probability that the customer will make a claim based on the provided features. Conversely, there is only a 3.98% probability that the customer will not make a claim.

### **Platform:**
- **Programming Language:** Python
- **Libraries:** XGBoost, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
- **Development Environment:** Jupyter Notebook or Google Colab

### **Example Record Analysis:**
- **Record:** id: 167647, Gender: Male, Age: 22, Driving_License: 1, Region_Code: 7, Previously_Insured: 1, Vehicle_Age: < 1 Year, Vehicle_Damage: No, Annual_Premium: 2630, Policy_Sales_Channel: 152, Vintage: 16, Response: 0

### **Summary**
The project involves developing a predictive model using the XGBoost algorithm to classify insurance policy responses. By effectively preparing the data, training the model, and fine-tuning its hyperparameters, the project aims to achieve high prediction accuracy. The results include detailed performance metrics, insights into feature importance, and practical simulations to support business decisions.



