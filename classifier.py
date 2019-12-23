import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, preprocessing
from sklearn.model_selection import train_test_split

df = pd.read_csv('wine.data')  # Load up wine data

X = np.array(df.drop(['class'], 1))  # Create X set
y = np.array(df['class'])  # Create Y set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)  # Split data into training and testing

model = neighbors.KNeighborsClassifier(n_jobs=7)  # Define classifier with 7 neighbors
model.fit(X_train, y_train)  # Train the model

y_pred = model.predict(X_test)  # Predict
score = model.score(X_test, y_test)  # Score