import pandas as pd
import numpy as np
from sklearn import neighbors, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

df = pd.read_csv('data/wine.data')  # Load up wine data

X = np.array(df.drop(['class'], 1))  # Create X set
y = np.array(df['class'])  # Create Y set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # Split data into training and testing

# Normalize X_train and X_test values (greatly increases accuracy)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = neighbors.KNeighborsClassifier(n_jobs=5)  # Define classifier with 7 neighbors
model.fit(X_train, y_train)  # Train the model

y_pred = model.predict(X_test)  # Predict
score = model.score(X_test, y_test)  # Score

print(classification_report(y_test, y_pred))