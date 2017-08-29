import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('test.csv')
x=dataset.iloc[:,0].values
y=dataset.iloc[:,1].values
x=x.reshape(1,-1)
y=y.reshape(1,-1)
x=np.asarray([5,3,0,4])
y=np.asarray([4,4,1,3])
m=((np.mean(x)*np.mean(y))-np.mean(x*y))/((np.mean(x)*np.mean(x))-np.mean(x*x))
print(m)
print(np.mean(y)-np.mean(x)*m)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x,y)
regressor.predict()



