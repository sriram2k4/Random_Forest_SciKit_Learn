import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Data Loading and Preprocessing

dataset = pd.read_csv("Social_Network_Ads.csv")

# print(dataset.head())

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# print(X)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling

standard_scaler = StandardScaler()
X_train = standard_scaler.fit_transform(X_train) # Calculating the mean and variance and transforming the features
X_test = standard_scaler.transform(X_test) # Transforming the test datset with the same mean variance
print(X_test)

# Training on Random Forest Classification Model

classifier = RandomForestClassifier(n_estimators=75, criterion="entropy", random_state=0)
classifier.fit(X_train, y_train)

# Predicting the test set

print(classifier.predict(standard_scaler.transform([[20,40000]])))

y_pred = classifier.predict(X_test)
y_pred = y_pred.reshape(len(y_pred),1)
y_test = y_test.reshape(len(y_test),1)


print(np.concatenate((y_pred,y_test),1))

# Confusion Matrix

confusion_matrix = confusion_matrix(y_test,y_pred)

print(confusion_matrix)

print(accuracy_score(y_test,y_pred))