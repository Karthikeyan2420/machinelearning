""" 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder

# Load the CSV data
data = pd.read_csv("Tennisdataset1.2.csv")

# Preprocess the features data
feature_cols = ['Outlook', 'Temperature', 'Humidity', 'Wind']
X = data[feature_cols]
X_encoded = X.apply(LabelEncoder().fit_transform)

# Preprocess the target variable
y = data['enjoysport']
y_encoded = LabelEncoder().fit_transform(y)

# Initialize the Naive Bayes classifier
clf = CategoricalNB()

# Train the classifier
clf.fit(X_encoded, y_encoded)

# Example data for prediction
new_data = pd.DataFrame({
    'Outlook': ['Sunny'],
    'Temperature': ['Hot'],
    'Humidity': ['High'],
    'Wind': ['Weak']
})

# Preprocess new data
new_data_encoded = new_data.apply(LabelEncoder().fit_transform)

# Make predictions
predictions = clf.predict(new_data_encoded)
print(predictions)
 """

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

# Load your dataset (replace 'your_dataset.csv' with the path to your dataset)
data = pd.read_csv('tennisdataset1.1.csv')

# Preprocess your data if needed

# Split your dataset into features (X) and target variable (y)
X = data.drop('enjoysport', axis=1)  # Features
y = data['enjoysport']  # Target variable
X = pd.get_dummies(X)
print(X)
# Split your dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Choose a Naive Bayes classifier (e.g., Gaussian Naive Bayes for continuous features)
model = GaussianNB()

# Train your classifier
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate your classifier
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred, average='weighted'))
print("Recall:", metrics.recall_score(y_test, y_pred, average='weighted'))
print("F1 Score:", metrics.f1_score(y_test, y_pred, average='weighted'))




