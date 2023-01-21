# import numpy as np
import pandas as pd
# import seaborn as sb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso ,Ridge
import pickle
from sklearn import metrics

# read the input file
df = pd.read_csv('C:\\Users\\navan\\datascience\\ML models\\Machine_learning_1\\Admission_prediction.csv')

# drop the serial no
df.drop("Serial No.", axis='columns', inplace=True)

# fill the null values
df['GRE Score'] = df['GRE Score'].fillna(df['GRE Score'].mean())
df['TOEFL Score'] = df['TOEFL Score'].fillna(df['TOEFL Score'].mean())
df['University Rating'] = df['University Rating'].fillna(df['University Rating'].mean())

# seperate the dependent and independent variable
X = df.iloc[:, :7]
y = df.iloc[:, 7:]

# Split the data for train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

# scale the variables
scalar = StandardScaler()

# For train fit and transform
X_train_scaled = scalar.fit_transform(X_train)
# For test transform
X_test_scaled = scalar.transform(X_test)



# create the model

lasso = Lasso()

# lasso fit
admission_model = lasso.fit(X_train_scaled, y_train)

# stored as serialized object

admission_model_test = lasso.predict(X_test_scaled)
model_accuracy = metrics.r2_score(y_test, admission_model_test)

print(model_accuracy)
file_name = 'finalized_model.sav'

pickle.dump(admission_model, open(file_name, 'wb'))










