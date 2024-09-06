#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("aug_train.csv")
df

Understanding the data: Check the structure of the data, the number of rows/columns, and the type of variables.
# In[2]:


df.info()
df.describe()

Data Cleaning: Check for missing or null values and handle them appropriately.
# In[3]:


df.isnull().sum()


# In[4]:


# Load your data
train_data = pd.read_csv('aug_train.csv')
test_data = pd.read_csv('aug_test.csv')


# In[5]:


# Z-score Method to detect outliers in `Annual_Premium`
from scipy import stats
import numpy as np

# Train Data
z_scores_train = np.abs(stats.zscore(train_data['Annual_Premium']))
outliers_z_train = train_data[z_scores_train > 3]

# Test Data
z_scores_test = np.abs(stats.zscore(test_data['Annual_Premium']))
outliers_z_test = test_data[z_scores_test > 3]

print("Outliers in Train Data (Z-score method):", outliers_z_train.shape[0])
print("Outliers in Test Data (Z-score method):", outliers_z_test.shape[0])


# In[6]:


# IQR Method
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]
    return outliers

# Example for `Annual_Premium`
outliers_iqr_train = detect_outliers_iqr(train_data, 'Annual_Premium')
outliers_iqr_test = detect_outliers_iqr(test_data, 'Annual_Premium')

print("Outliers in Train Data (IQR method):", outliers_iqr_train.shape[0])
print("Outliers in Test Data (IQR method):", outliers_iqr_test.shape[0])

# 3. Boxplot Visualization
plt.figure(figsize=(12, 6))
sns.boxplot(data=train_data, x='Annual_Premium')
plt.title('Boxplot for Annual Premium (Train Data)')
plt.show()


# In[7]:


# Remove outliers from train and test based on IQR
clean_train_data = train_data[~train_data.index.isin(outliers_iqr_train.index)]
clean_test_data = test_data[~test_data.index.isin(outliers_iqr_test.index)]

print("Train data after removing outliers:", clean_train_data.shape)
print("Test data after removing outliers:", clean_test_data.shape)


# In[8]:


# Capping based on the 1st and 99th percentiles
cap_train_data = train_data.copy()
cap_test_data = test_data.copy()

# Define percentiles
low_cap = cap_train_data['Annual_Premium'].quantile(0.01)
high_cap = cap_train_data['Annual_Premium'].quantile(0.99)

# Apply capping
cap_train_data['Annual_Premium'] = np.clip(cap_train_data['Annual_Premium'], low_cap, high_cap)
cap_test_data['Annual_Premium'] = np.clip(cap_test_data['Annual_Premium'], low_cap, high_cap)

print("Capped Train Data shape:", cap_train_data.shape)
print("Capped Test Data shape:", cap_test_data.shape)

Z-Score Method for All Features
# In[9]:


# Check if 'Response' exists in the dataset
print(test_data.columns)


# In[10]:


# Update the list of numeric columns to exclude any non-numeric columns
numeric_columns = train_data.select_dtypes(include=[np.number]).columns

# Ensure the same columns are in test_data
numeric_columns = [col for col in numeric_columns if col in test_data.columns]

# Function to get Z-score outliers for each numeric column
def detect_outliers_z(df, columns):
    outliers = pd.DataFrame()
    for column in columns:
        z_scores = np.abs(stats.zscore(df[column].dropna()))  # Drop NaN values
        outliers_in_column = df[z_scores > 3]
        print(f"Outliers in {column} (Z-score method): {outliers_in_column.shape[0]}")
        outliers = pd.concat([outliers, outliers_in_column])  # Use concat instead of append
    return outliers

outliers_z_train_all = detect_outliers_z(train_data, numeric_columns)
outliers_z_test_all = detect_outliers_z(test_data, numeric_columns)

Visualize Outliers Before Handling
# In[11]:


# Function to plot boxplots for each column
def plot_boxplots(df, columns, title_prefix=""):
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(columns):
        plt.subplot(3, 4, i + 1)  # Adjust the grid size as needed
        sns.boxplot(x=df[column])
        plt.title(f'{title_prefix} Boxplot for {column}')
    plt.tight_layout()
    plt.show()

# List of columns with outliers
columns_with_outliers = ['Driving_License', 'Annual_Premium']

# Plot boxplots before handling
plot_boxplots(train_data, columns_with_outliers, title_prefix="Before Handling")

# You can include test data if needed:
plot_boxplots(test_data, columns_with_outliers, title_prefix="Before Handling Test Data")

Handle Outliers
# In[12]:


# Function to handle outliers using IQR and remove them
def handle_outliers_iqr(df, columns):
    df_cleaned = df.copy()
    for column in columns:
        Q1 = df_cleaned[column].quantile(0.25)
        Q3 = df_cleaned[column].quantile(0.75)
        IQR = Q3 - Q1
        df_cleaned = df_cleaned[~((df_cleaned[column] < (Q1 - 1.5 * IQR)) | (df_cleaned[column] > (Q3 + 1.5 * IQR)))]
    return df_cleaned

# Handle outliers in training and test data
clean_train_data = handle_outliers_iqr(train_data, columns_with_outliers)
clean_test_data = handle_outliers_iqr(test_data, columns_with_outliers)

Visualize Outliers After Handling
# In[13]:


# Plot boxplots after handling
plot_boxplots(clean_train_data, columns_with_outliers, title_prefix="After Handling")

# You can include test data if needed:
plot_boxplots(clean_test_data, columns_with_outliers, title_prefix="After Handling Test Data")

Checking Outliers After Handling
# In[14]:


from scipy import stats

# Function to count outliers using Z-score method
def count_outliers_z(df, columns):
    outlier_counts = {}
    for column in columns:
        z_scores = np.abs(stats.zscore(df[column].dropna()))  # Drop NaN values
        outlier_counts[column] = np.sum(z_scores > 3)
    return outlier_counts

# Function to count outliers using IQR method
def count_outliers_iqr(df, columns):
    outlier_counts = {}
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]
        outlier_counts[column] = outliers.shape[0]
    return outlier_counts

# Check for outliers in cleaned training and test data
outliers_z_train_cleaned = count_outliers_z(clean_train_data, columns_with_outliers)
outliers_iqr_train_cleaned = count_outliers_iqr(clean_train_data, columns_with_outliers)

outliers_z_test_cleaned = count_outliers_z(clean_test_data, columns_with_outliers)
outliers_iqr_test_cleaned = count_outliers_iqr(clean_test_data, columns_with_outliers)

print("Outliers in Cleaned Train Data (Z-score method):", outliers_z_train_cleaned)
print("Outliers in Cleaned Train Data (IQR method):", outliers_iqr_train_cleaned)

print("Outliers in Cleaned Test Data (Z-score method):", outliers_z_test_cleaned)
print("Outliers in Cleaned Test Data (IQR method):", outliers_iqr_test_cleaned)


# In[15]:


# Plot boxplot for 'Annual_Premium' in cleaned train and test data
plt.figure(figsize=(12, 6))

# Boxplot for Train Data
plt.subplot(1, 2, 1)
sns.boxplot(data=clean_train_data, x='Annual_Premium')
plt.title('Boxplot for Annual Premium (Cleaned Train Data)')

# Boxplot for Test Data
plt.subplot(1, 2, 2)
sns.boxplot(data=clean_test_data, x='Annual_Premium')
plt.title('Boxplot for Annual Premium (Cleaned Test Data)')

plt.tight_layout()
plt.show()

Convert Categorical to Numeric
If Driving_License is binary (e.g., 0 and 1), ensure it's encoded as such. Otherwise, encode categorical variables:
# In[16]:


# Check if Driving_License is categorical and encode if necessary
if clean_train_data['Driving_License'].dtype == 'object':
    clean_train_data['Driving_License'] = clean_train_data['Driving_License'].astype('category').cat.codes


# In[17]:


# Drop the 'Driving_License' column from the train and test datasets
clean_train_data = clean_train_data.drop(columns=['Driving_License'])
clean_test_data = clean_test_data.drop(columns=['Driving_License'])

# Verify the column has been removed
print(clean_train_data.head())
print(clean_test_data.head())

 Visualize Each Feature
# In[18]:


print(train_data.columns)


# In[19]:


# Plot histograms for each feature in the train dataset
features = ['Age', 'id', 'Gender', 'Region_Code', 'Previously_Insured', 'Annual_Premium',
            'Policy_Sales_Channel', 'Vintage', 'Vehicle_Age', 'Vehicle_Damage', 'Response']

# Determine the number of rows and columns needed for the subplots
num_features = len(features)
num_cols = 3  # Number of columns in the subplot grid
num_rows = (num_features + num_cols - 1) // num_cols  # Calculate rows needed

plt.figure(figsize=(15, 5 * num_rows))

for i, feature in enumerate(features):
    plt.subplot(num_rows, num_cols, i + 1)
    plt.hist(clean_train_data[feature], bins=30, edgecolor='k', alpha=0.7)
    plt.title(f'Histogram of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[20]:


# List of features to plot
features = ['Age','Region_Code', 'Previously_Insured', 'Annual_Premium',
            'Policy_Sales_Channel', 'Vintage', 'Response']

# Determine the number of rows and columns needed for the subplots
num_features = len(features)
num_cols = 3  # Number of columns in the subplot grid
num_rows = (num_features + num_cols - 1) // num_cols  # Calculate rows needed

plt.figure(figsize=(15, 5 * num_rows))

for i, feature in enumerate(features):
    plt.subplot(num_rows, num_cols, i + 1)
    sns.boxplot(data=clean_train_data[feature])
    plt.title(f'Boxplot of {feature}')
    plt.xlabel(feature)

plt.tight_layout()
plt.show()


# In[21]:


# Calculate the correlation matrix
corr_matrix = clean_train_data.corr()

# Plot the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

Ensure Proper Encoding
# In[22]:


# One-hot encode categorical features
clean_train_data_encoded = pd.get_dummies(clean_train_data, columns=['Gender', 'Vehicle_Age', 'Vehicle_Damage'], drop_first=True)

# Verify the new dataset
print(clean_train_data_encoded.head())


# In[23]:


from sklearn.model_selection import train_test_split

# Define feature columns and target variable
features = clean_train_data_encoded.drop(columns=['Response'])
target = clean_train_data_encoded['Response']

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Verify the shape of the split data
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")


# In[ ]:




Logistic Regression
# In[24]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Initialize and train the model
model = LogisticRegression(max_iter=2000, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[25]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=2000, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_scaled)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[26]:


# Ensure that feature names are strings and do not contain special characters
X_train.columns = [str(col).replace('[', '').replace(']', '').replace('<', '').replace(' ', '_') for col in X_train.columns]
X_test.columns = [str(col).replace('[', '').replace(']', '').replace('<', '').replace(' ', '_') for col in X_test.columns]


# In[ ]:




XGBoost with RandomizedSearchCV
# In[27]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Sampling the data for faster hyperparameter tuning
X_train_sampled, _, y_train_sampled, _ = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define the parameter distribution for XGBoost
param_dist_xgb = {
    'classifier__n_estimators': randint(50, 100),
    'classifier__learning_rate': uniform(0.01, 0.2),
    'classifier__max_depth': randint(3, 7),
    'classifier__subsample': uniform(0.7, 0.3),
    'classifier__colsample_bytree': uniform(0.7, 0.3)
}

# Create and fit RandomizedSearchCV for XGBoost with early stopping
random_search_xgb = RandomizedSearchCV(estimator=Pipeline([
    ('resampling', SMOTE()),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
]), param_distributions=param_dist_xgb, n_iter=10, cv=3, scoring='roc_auc', n_jobs=-1, verbose=2)
random_search_xgb.fit(X_train_sampled, y_train_sampled)

# Best parameters and model
print("Best parameters for XGBoost:", random_search_xgb.best_params_)
best_model_xgb = random_search_xgb.best_estimator_

# Evaluate the best model on full test set
y_pred_xgb = best_model_xgb.predict(X_test)

# Classification report
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

print("Classification Report:")
print(classification_report(y_test, y_pred_xgb))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))

print("ROC-AUC Score:", roc_auc_score(y_test, best_model_xgb.predict_proba(X_test)[:, 1]))


# In[ ]:




Further Encoding of Categorical Variables:

Encoding: Convert categorical variables into numeric codes.
Scaling: Standardize numerical features using StandardScaler.
# In[28]:


# Example of encoding categorical variables
clean_train_data['Gender'] = clean_train_data['Gender'].astype('category').cat.codes
clean_train_data['Vehicle_Age'] = clean_train_data['Vehicle_Age'].astype('category').cat.codes
clean_train_data['Vehicle_Damage'] = clean_train_data['Vehicle_Damage'].astype('category').cat.codes

clean_test_data['Gender'] = clean_test_data['Gender'].astype('category').cat.codes
clean_test_data['Vehicle_Age'] = clean_test_data['Vehicle_Age'].astype('category').cat.codes
clean_test_data['Vehicle_Damage'] = clean_test_data['Vehicle_Damage'].astype('category').cat.codes

Feature Scaling:
# In[29]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numerical_columns = ['Age', 'Region_Code', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']

clean_train_data[numerical_columns] = scaler.fit_transform(clean_train_data[numerical_columns])
clean_test_data[numerical_columns] = scaler.transform(clean_test_data[numerical_columns])


# In[ ]:




Feature SelectionUnivariate Feature Selection
# In[30]:


from sklearn.feature_selection import SelectKBest, f_classif

# Define features and target
X = clean_train_data.drop(columns=['Response'])  # Replace 'Response' with your actual target column name
y = clean_train_data['Response']

# Apply ANOVA F-statistic
selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(X, y)

# Get feature scores
scores = pd.DataFrame(selector.scores_, index=X.columns, columns=['Score'])
scores.sort_values(by='Score', ascending=False, inplace=True)
print(scores)

Recursive Feature Elimination (RFE):

Works well with regression and classification models.
# In[31]:


from sklearn.feature_selection import RFE

# Initialize the model
model = LogisticRegression()

# Initialize RFE with the model
rfe = RFE(model, n_features_to_select=5)
rfe = rfe.fit(X, y)

# Get selected features
selected_features = X.columns[rfe.support_]
print("Selected features:", selected_features)


# In[ ]:




Model Training with Selected Features
Model: RandomForestClassifier
# In[32]:


from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Create a new dataset with selected features
selected_features = ['Age', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Policy_Sales_Channel']
X_selected = X[selected_features]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

# Train a model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[33]:


from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train the model on the resampled data
model = RandomForestClassifier(random_state=42)
model.fit(X_resampled, y_resampled)


# In[34]:


# Analyzing Feature Importance with Random Forest

# Assuming you've trained a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Get feature importances
importances = model.feature_importances_

# Create a DataFrame for visualization
feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Print feature importances
print(feature_importances)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.show()


# In[ ]:




PCA (Principal Component Analysis)
Explained Variance Ratio: Mostly dominated by the first component, suggesting most variance is captured by it.
# In[35]:


from sklearn.decomposition import PCA

# Scale the data before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

pca = PCA(n_components=5)
principal_components = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_
print("Explained variance ratio:", explained_variance)


# In[36]:


# Initialize PCA and model
pca = PCA(n_components=5)
model = RandomForestClassifier(random_state=42)

# Create a pipeline
pipeline = Pipeline(steps=[('pca', pca), ('model', model)])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the pipeline
pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipeline.predict(X_test)
print("Accuracy with PCA:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[39]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model with X_train_scaled
model.fit(X_train_scaled, y_train)

# Make predictions with X_test_scaled
y_pred = model.predict(X_test_scaled)


# In[40]:


print("Training features:", X_train.columns)
print("Test features:", X_test.columns)

Model Performance Analysis
# In[44]:


from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Example feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the model
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_scaled)

# Performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()


# In[ ]:




Handling class Imbalance
# In[45]:


from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

# Create pipeline with SMOTE and logistic regression
smote = SMOTE(random_state=42)
model = LogisticRegression(max_iter=1000, random_state=42)

pipeline = ImbPipeline([
    ('smote', smote),
    ('classifier', model)
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_train)
print(y_pred)


# In[46]:


from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Predict on the test set
y_test_pred = pipeline.predict(X_test)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_test_pred))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# Compute ROC AUC score
roc_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
print(f"ROC AUC Score: {roc_auc:.2f}")


# In[ ]:




Model Tuning :Hyperparameter Tuning with GridSearchCV:
# In[47]:


import xgboost as xgb
# Define the model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Fit the model
xgb_model.fit(X, y)

# Predict and evaluate
y_pred = xgb_model.predict(X)
print("Accuracy:", accuracy_score(y, y_pred))
print("Classification Report:")
print(classification_report(y, y_pred))


# In[49]:


from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1]
}

xgb_model = xgb.XGBClassifier(eval_metric='mlogloss')
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy')

# Fit the model
grid_search.fit(X_train, y_train)

# Best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)


# In[50]:


from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='accuracy')
print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Score:", cv_scores.mean())


# In[ ]:





# In[51]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming y_train is your target variable
plt.figure(figsize=(8, 6))
sns.countplot(x=y_train)
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


# In[53]:


from sklearn.metrics import roc_curve, auc

# Get the predicted probabilities for the positive class
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)

# Calculate AUC
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Dashed diagonal line for random guessing
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc='lower right')
plt.show()


# In[55]:


from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Assuming you have your features (X) and target variable (y)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Create XGBoost model
xgb_model = XGBClassifier(eval_metric='mlogloss')

# Train the model on the balanced dataset
xgb_model.fit(X_resampled, y_resampled)

# Make predictions
y_pred = xgb_model.predict(X_test)


# In[56]:


# Evaluate performance
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("ROC AUC Score:", roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1]))

# Visualize ROC curve
fpr, tpr, thresholds = roc_curve(y_test, xgb_model.predict_proba(X_test)[:, 1])
roc_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# In[57]:


# Assuming y_resampled is the balanced target variable after oversampling
y_resampled_counts = pd.Series(y_resampled).value_counts()

plt.figure(figsize=(8, 6))
sns.barplot(x=y_resampled_counts.index, y=y_resampled_counts.values)
plt.title('Class Distribution (Balanced)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


# In[59]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score

# Create XGBoost model
xgb_model = XGBClassifier(eval_metric='mlogloss')

# Train the model on the balanced dataset
xgb_model.fit(X_resampled, y_resampled)

# Make predictions on the test set
y_pred = xgb_model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)

# Visualize ROC curve
fpr, tpr, thresholds = roc_curve(y_test, xgb_model.predict_proba(X_test)[:, 1])
roc_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# In[60]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# Train the model on the balanced dataset
xgb_model.fit(X_resampled, y_resampled)

# Make predictions on the test set
y_pred = xgb_model.predict(X_test)
y_prob = xgb_model.predict_proba(X_test)[:, 1]  # Probabilities for ROC AUC calculation

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"ROC AUC Score: {roc_auc:.2f}")


# In[ ]:





# In[80]:


import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# Define features for training and test datasets
features = ['Gender', 'Age', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']

# Define features and target for training
X_train = clean_train_data[features]
y_train = clean_train_data['Response']

# Define features for testing
X_test = clean_test_data[features]

# Convert categorical variables into dummy variables
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Handle any mismatch in columns between training and test data
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Initialize and train the XGBoost model
xgb_model = xgb.XGBClassifier(eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = xgb_model.predict(X_test)
y_prob = xgb_model.predict_proba(X_test)[:, 1]  # Probabilities for ROC AUC calculation

# Print predictions and probabilities
print("Predictions:")
print(y_pred)
print("Prediction Probabilities:")
print(y_prob)


# In[81]:


# Estimate Response Likelihood Based on Premium
# Create a DataFrame to simulate varying premium values
premium_values = [15000, 25000, 35000, 45000, 55000]  # Example premium values
simulation_data = pd.DataFrame({
    'Gender': ['Male'] * len(premium_values),
    'Age': [30] * len(premium_values),
    'Region_Code': [1] * len(premium_values),
    'Previously_Insured': [0] * len(premium_values),
    'Vehicle_Age': ['1-2 Year'] * len(premium_values),
    'Vehicle_Damage': ['No'] * len(premium_values),
    'Annual_Premium': premium_values,
    'Policy_Sales_Channel': [26] * len(premium_values),
    'Vintage': [100] * len(premium_values)
})

# Convert to dummy variables
simulation_data = pd.get_dummies(simulation_data)
simulation_data = simulation_data.reindex(columns=X_train.columns, fill_value=0)

# Predict response probabilities for simulated data
premium_probabilities = xgb_model.predict_proba(simulation_data)[:, 1]

# Display the results
simulation_results = pd.DataFrame({
    'Annual_Premium': premium_values,
    'Predicted_Probability': premium_probabilities
})

print("Estimated Response Likelihood Based on Premium:")
print(simulation_results)


# In[87]:


import matplotlib.pyplot as plt

# Plot feature importance
importance = xgb_model.feature_importances_
feature_names = X_train.columns
plt.figure(figsize=(10, 8))
plt.barh(feature_names, importance)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance for XGBoost Model')
plt.show()


# In[91]:


# Define features and target
features = ['Gender', 'Age', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']
X = clean_train_data[features]
y = clean_train_data['Response']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert categorical variables into dummy variables
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Handle any mismatch in columns between training and test data
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Initialize and train the XGBoost model
xgb_model = xgb.XGBClassifier(eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = xgb_model.predict(X_test)
y_prob = xgb_model.predict_proba(X_test)[:, 1]

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("Performance on the Test Set:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"ROC AUC Score: {roc_auc:.2f}")


# In[92]:


# Define important features
important_features = ['Vehicle_Damage', 'Previously_Insured']
X_train_important = X_train[important_features]
X_test_important = X_test[important_features]

# Initialize and train the XGBoost model with important features
xgb_model_important = xgb.XGBClassifier(eval_metric='mlogloss')
xgb_model_important.fit(X_train_important, y_train)

# Make predictions on the test set with important features
y_pred_important = xgb_model_important.predict(X_test_important)
y_prob_important = xgb_model_important.predict_proba(X_test_important)[:, 1]

# Evaluate performance
accuracy_important = accuracy_score(y_test, y_pred_important)
precision_important = precision_score(y_test, y_pred_important)
recall_important = recall_score(y_test, y_pred_important)
f1_important = f1_score(y_test, y_pred_important)
roc_auc_important = roc_auc_score(y_test, y_prob_important)

print("Performance with Important Features:")
print(f"Accuracy: {accuracy_important:.2f}")
print(f"Precision: {precision_important:.2f}")
print(f"Recall: {recall_important:.2f}")
print(f"F1 Score: {f1_important:.2f}")
print(f"ROC AUC Score: {roc_auc_important:.2f}")


# In[93]:


from sklearn.model_selection import GridSearchCV

# Define the parameter grid for XGBoost
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=xgb.XGBClassifier(eval_metric='mlogloss'),
                           param_grid=param_grid,
                           scoring='roc_auc',
                           cv=5,
                           verbose=1,
                           n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Best parameters and model
print("Best parameters found: ", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Evaluate performance with the best model
y_pred_best = best_model.predict(X_test)
y_prob_best = best_model.predict_proba(X_test)[:, 1]

accuracy_best = accuracy_score(y_test, y_pred_best)
precision_best = precision_score(y_test, y_pred_best)
recall_best = recall_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best)
roc_auc_best = roc_auc_score(y_test, y_prob_best)

print("Performance with Best Model:")
print(f"Accuracy: {accuracy_best:.2f}")
print(f"Precision: {precision_best:.2f}")
print(f"Recall: {recall_best:.2f}")
print(f"F1 Score: {f1_best:.2f}")
print(f"ROC AUC Score: {roc_auc_best:.2f}")


# In[94]:


print("Best parameters found: ", grid_search.best_params_)


# In[95]:


print("Performance with Best Model:")
print(f"Accuracy: {accuracy_best:.2f}")
print(f"Precision: {precision_best:.2f}")
print(f"Recall: {recall_best:.2f}")
print(f"F1 Score: {f1_best:.2f}")
print(f"ROC AUC Score: {roc_auc_best:.2f}")


# In[96]:


import joblib

# Save the best model to a file
joblib.dump(best_model, 'xgb_best_model.pkl')


# In[97]:


# Load the model from file
loaded_model = joblib.load('xgb_best_model.pkl')


# In[ ]:





# In[104]:


import pandas as pd

def get_user_input():
    # Prompt user for input
    gender = input("Enter Gender (Male/Female): ")
    age = int(input("Enter Age: "))
    region_code = int(input("Enter Region Code: "))
    previously_insured = int(input("Enter Previously Insured (0/1): "))
    vehicle_age = input("Enter Vehicle Age (e.g., '1-2 Year', '3-4 Year', '> 4 Years'): ")
    vehicle_damage = input("Enter Vehicle Damage (Yes/No): ")
    annual_premium = float(input("Enter Annual Premium: "))
    policy_sales_channel = int(input("Enter Policy Sales Channel: "))
    vintage = int(input("Enter Vintage: "))
    
    return pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Region_Code': [region_code],
        'Previously_Insured': [previously_insured],
        'Vehicle_Age': [vehicle_age],
        'Vehicle_Damage': [vehicle_damage],
        'Annual_Premium': [annual_premium],
        'Policy_Sales_Channel': [policy_sales_channel],
        'Vintage': [vintage]
    })

def predict_insurance_probability(user_input_df, model, feature_columns):
    # Convert to dummy variables
    user_input_df = pd.get_dummies(user_input_df)
    
    # Reindex to match the feature columns of the training data
    user_input_df = user_input_df.reindex(columns=feature_columns, fill_value=0)
    
    # Predict response probabilities
    user_input_prob = model.predict_proba(user_input_df)[:, 1]
    return user_input_prob

# Example usage
if __name__ == "__main__":
    # Assume `loaded_model` is your trained model and `X_train.columns` contains the feature columns
    user_input_df = get_user_input()
    predicted_probability = predict_insurance_probability(
        user_input_df=user_input_df,
        model=loaded_model,
        feature_columns=X_train.columns
    )
    
    print("Predicted Probability for New Data:", predicted_probability)


# In[ ]:





# In[ ]:





# In[ ]:




