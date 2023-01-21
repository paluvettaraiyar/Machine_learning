# import numpy as np
import pandas as pd
# import seaborn as sb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import pickle
from sklearn import metrics

# read the input file
df = pd.read_csv('CAdmission_prediction.csv')

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

ridge = Ridge()
lasso = Lasso()
linear = LinearRegression()

# lasso fit
admission_model = lasso.fit(X_train_scaled, y_train)
admission_model_r = ridge.fit(X_train_scaled, y_train)
admission_model_l = linear.fit(X_train_scaled, y_train)

# stored as serialized object

admission_model_test_l = lasso.predict(X_test_scaled)
admission_model_test_r = ridge.predict(X_test_scaled)
admission_model_test = linear.predict(X_test_scaled)

model_accuracy_li = metrics.r2_score(y_test, admission_model_test_l)


model_accuracy_la = metrics.r2_score(y_test, admission_model_test_r)

model_accuracy_r = metrics.r2_score(y_test, admission_model_test)

print(model_accuracy_li) # -0.00031541530306666843
print(model_accuracy_la) # 0.7902504059740092
print(model_accuracy_r) # 0.7902023531711815




file_name = 'finalized_model.sav'

pickle.dump(model_accuracy_r, open(file_name, 'wb'))










