#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries for data manipulation, numerical operations, data splitting, feature scaling, SVM model training, and accuracy evaluation.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# In[2]:


# Loading the diabetes dataset into a Pandas DataFrame and displaying the first five rows.
data = pd.read_csv('diabetes.csv')
data.head()


# In[3]:


# Displaying the number of rows and columns in the dataset.
data.shape


# In[4]:


# Generating descriptive statistical summaries of the numerical columns in the dataset.
data.describe()


# In[5]:


# Counting the number of occurrences of each outcome class.
data['Outcome'].value_counts()


# In[6]:


# Calculating the average values of all numerical features for each outcome group.
data.groupby('Outcome').mean()


# In[7]:


# Check target distribution
sns.countplot(data=data, x='Outcome')
plt.title('Diabetes Outcome Distribution')
plt.show()


# In[8]:


# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()


# In[9]:


# Boxplot
sns.boxplot(data=data, x='Outcome', y='Glucose')
plt.title('Glucose Distribution by Diabetes Outcome')
plt.show()


# In[10]:


# Feature distributions
data.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()


# In[11]:


# Identify zero values in relevant medical measurements.
columns_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
(data[columns_to_check] == 0).sum()


# In[12]:


# Replace invalid zero measurements with missing values.
data[columns_to_check] = data[columns_to_check].replace(0, np.nan)


# In[13]:


# Separating the independent features from the target variable.
X = X = data.drop(columns='Outcome')
print(X)


# In[14]:


# Storing the target variable (Outcome) separately.
Y = data['Outcome']
print(Y)


# In[15]:


# Splitting the dataset into training and testing sets while maintaining the original class distribution.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1, stratify=Y)
X.shape, X_train.shape, X_test.shape


# In[17]:


# Build a pipeline for missing value imputation, feature scaling, and SVM classification.
model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', svm.SVC(kernel='linear'))
])

model.fit(X_train, Y_train)


# In[18]:


# Evaluate model performance on training data.
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

print("Training Accuracy:", training_data_accuracy)


# In[19]:


# Evaluate model performance on testing data.
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print("Testing Accuracy:", test_data_accuracy)


# In[20]:


# Display the classification report and confusion matrix.
print(classification_report(Y_test, X_test_prediction))
print(confusion_matrix(Y_test, X_test_prediction))


# In[22]:


# Visualize the confusion matrix to understand correct and incorrect predictions.
sns.heatmap(confusion_matrix(Y_test, X_test_prediction), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[21]:


# Predict diabetes outcome using new patient data.
input_data = (1, 103, 30, 38, 83, 43.3, 0.183, 33)

input_df = pd.DataFrame([input_data], columns=X.columns)

prediction = model.predict(input_df)

if prediction[0] == 0:
    print('The person is not diabetic.')
else:
    print('The person is diabetic.')


# In[23]:


# Save the trained machine learning pipeline for future use.
import joblib

joblib.dump(model, 'diabetes_model.pkl')





