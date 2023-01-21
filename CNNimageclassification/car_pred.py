import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


df1 = pd.read_csv('CarPrice_Assignment.csv')
df1.drop("car_ID",axis='columns',inplace=True)
plt.rcParams['figure.figsize'] = (12, 12)

for i in range(0, len(df1.columns)):
    plt.subplot(len(df1.columns), 1, i + 1)
    plt.plot(df1.columns[i], df1.columns[24], data=df1)


