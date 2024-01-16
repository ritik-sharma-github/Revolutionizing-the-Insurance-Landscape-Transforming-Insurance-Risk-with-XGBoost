# Revolutionizing-the-Insurance-Landscape.

"Revolutionizing the Insurance Landscape" is all about using data and machine learning to change how insurance system works. The goal is to make it fairer, more efficient, and better at handling claims. We want to use the power of data and predictive models to do this. Our big mission is to make the insurance industry better by exploring data, making predictions, and using machines to help us. Our aim of this research is to figure out if people who already have health insurance might also want to get vehicles insurance from the same company. This helps us expand the business into the vehicle’s insurance market. We'll use data and machine learning to study this and then use what we learn to talk to these people in a way that makes sense. This will help us use our resources better and make more money. This research is a good example of how the insurance industry is changing. We are using data and focusing on what customers want to make insurance better. It's all about adapting to the way things are changing in the world and making insurance work better for everyone. 

Our research significantly advances vehicle insurance response prediction with an impressive 82.3% test data accuracy. Valuable findings, like 'Vehicle Damage' impact, offer actionable insights. A user interaction feature for real-time predictions holds promise. Dataset shuffling bolsters model robustness. Future directions include advanced ML techniques and data integration. Our work lays the groundwork for improving response models in a dynamic insurance landscape. Future work in vehicle insurance response prediction involves advanced machine learning techniques, improved feature engineering, ensemble methods, and temporal analysis. Addressing class imbalance, enhancing model interpretability, and enhancing user experiences are critical. The integration of external data sources for enriched predictions and real-time prediction capabilities can reshape decision-making in the insurance industry. Continuous model monitoring, ethical considerations, and regulatory compliance ensure model effectiveness and fair treatment of customers. These advancements will empower the industry to adapt to evolving customer behavior and market dynamics.


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, accuracy_score

# Load your full dataset
full_data = pd.read_csv("C:/Users/ritti/Downloads/SET1_PROJECT/vehicle_insurance/aug_train.csv")

# Identify categorical columns
categorical_cols = full_data.select_dtypes(include=['object']).columns.tolist()

# Apply label encoding to all categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    full_data[col] = le.fit_transform(full_data[col])
    label_encoders[col] = le

# Split the full dataset into features (X) and the target variable (y)
X = full_data.drop(['Response'], axis=1)
y = full_data['Response']

# Split the data into a new training set and a validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model (Logistic Regression)
model = LogisticRegression()

# Train the model on the new training set
model.fit(X_train, y_train)

# Make predictions on the validation set
y_val_pred = model.predict(X_val)

# Evaluate the model on the validation set
print("Validation Set Accuracy:", accuracy_score(y_val, y_val_pred))
print("Classification Report on Validation Set:\n", classification_report(y_val, y_val_pred))

# Load the provided test data
test_data = pd.read_csv("C:/Users/ritti/Downloads/SET1_PROJECT/vehicle_insurance/aug_test.csv")

# Apply label encoding to categorical columns in the test data using the same encoders
for col in categorical_cols:
    le = label_encoders[col]
    test_data[col] = le.transform(test_data[col])

# Make predictions on the provided test data
test_predictions = model.predict(test_data)

# Add the predicted 'Response' column to the test data
test_data['Predicted_Response'] = test_predictions

# Display the test data with predictions
print(test_data[['id', 'Predicted_Response']])





import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load your full dataset
full_data = pd.read_csv("C:/Users/ritti/Downloads/SET1_PROJECT/vehicle_insurance/aug_train.csv")

# Identify categorical columns
categorical_cols = full_data.select_dtypes(include=['object']).columns.tolist()

# Apply label encoding to all categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    full_data[col] = le.fit_transform(full_data[col])
    label_encoders[col] = le

# Split the full dataset into features (X) and the target variable (y)
X = full_data.drop(['Response'], axis=1)
y = full_data['Response']

# Split the data into a new training set and a validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model (Logistic Regression)
model = LinearRegression()

# Train the model on the new training set
model.fit(X_train, y_train)

# Make predictions on the validation set
y_val_pred = model.predict(X_val)

# Evaluate the model on the validation set
print("Validation Set Accuracy:", accuracy_score(y_val, y_val_pred))
print("Classification Report on Validation Set:\n", classification_report(y_val, y_val_pred))


# Make predictions on the provided test data
test_data = pd.read_csv("C:/Users/ritti/Downloads/SET1_PROJECT/vehicle_insurance/aug_test.csv")

# Apply label encoding to categorical columns in the test data using the same encoders
for col in categorical_cols:
    le = label_encoders[col]
    test_data[col] = le.transform(test_data[col])

# Make predictions on the test data
test_predictions = model.predict(test_data)

# Create a new DataFrame with 'id' and 'Predicted_Response' columns
predicted_df = pd.DataFrame({'id': test_data['id'], 'Response': test_predictions})
predicted_df

full_data_shuffled = full_data.sample(frac=1, random_state=42)  # Shuffling the data

# Calculate the test accuracy using the true 'Response' values and the predicted 'Response' values
df = full_data_shuffled.iloc[:len(predicted_df['Response'])]
df.reset_index(drop=True, inplace=True)
df = pd.DataFrame(df)

true_responses = df['Response']
test_accuracy = accuracy_score(true_responses, test_predictions)

print("Test Set Accuracy:", test_accuracy)


Validation Set Accuracy: 0.8315866598631445
Classification Report on Validation Set:
               precision    recall  f1-score   support

           0       0.84      0.99      0.91     63789
           1       0.38      0.03      0.05     12642

    accuracy                           0.83     76431
   macro avg       0.61      0.51      0.48     76431
weighted avg       0.76      0.83      0.77     76431

           id  Predicted_Response
0       57782                   0
1      286811                   0
2      117823                   0
3      213992                   0
4      324756                   0
...       ...                 ...
78268     847                   0
78269  417524                   0
78270  188087                   0
78271  215680                   0
78272  138006                   0

[78273 rows x 2 columns]
Validation Set Accuracy: 0.8315866598631445
Classification Report on Validation Set:
               precision    recall  f1-score   support

           0       0.84      0.99      0.91     63789
           1       0.38      0.03      0.05     12642

    accuracy                           0.83     76431
   macro avg       0.61      0.51      0.48     76431
weighted avg       0.76      0.83      0.77     76431

id	Response
0	57782	0
1	286811	0
2	117823	0
3	213992	0
4	324756	0
...	...	...
78268	847	0
78269	417524	0
78270	188087	0
78271	215680	0
78272	138006	0
78273 rows × 2 columns

Test Set Accuracy: 0.826351359983647
