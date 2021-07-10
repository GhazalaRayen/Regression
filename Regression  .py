#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

kc = pd.read_csv (r'C:\Users\ghaza\OneDrive\Bureau\GoMyCode\kc_house_data.csv')


# In[2]:


kc.head()


# In[3]:


kc.head().isnull().sum()


# In[4]:


import seaborn as sns

Var_Corr = kc.corr()
plt.figure(figsize=(20,20))
sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)

On remarque que les meilleures corrélatoins sont entre :
"sqft_living15" et  "sqft_above" avec un taux de corrélation 0.72, 
"sqft_living" et "bathrooms" avec un taux de corrélation 0.75 ,
"sqft_living et sqft_lot15" avec un taux de corrélation de 0.76 ,
"sqft_living et sqft_living15" avec un taux de corrélation de 0.76 ,
"grade" et "sqft_above" avec un taux de corrélation de 0.76.

# In[83]:


df = pd.read_csv (r'C:\Users\ghaza\OneDrive\Bureau\GoMyCode\kc_house_data.csv')
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import metrics
X= kc[["sqft_living15","sqft_living","sqft_above","sqft_lot"]]
y= kc.bathrooms

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
print ("X_train: ", X_train)
print ("y_train: ", y_train)
print("X_test: ", X_test)
print ("y_test: ", y_test) 


# In[84]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print('Train set:', X_train.shape)
print('Test set:', X_test.shape)


# In[85]:


#Importing libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import metrics


# In[86]:



x=kc["sqft_living15"].values[:,np.newaxis]
y=kc["sqft_above"].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=40)
model=LinearRegression()   
model.fit(x_train,y_train)  
predicted=model.predict(x_test) 

print("MSE", mean_squared_error(y_test,predicted))
print("R squared", metrics.r2_score(y_test,predicted))


# In[87]:


plt.scatter(x,y,color="r")
plt.title("Linear Regression")
plt.ylabel("sqft_living15")
plt.xlabel("sqft_above")
plt.plot(x,model.predict(x),color="k")
plt.show()


# In[96]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures 

from sklearn.metrics import mean_squared_error
from sklearn import metrics
x= kc[["sqft_living15", "sqft_living"]]
y= kc["sqft_above"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=40)  #splitting data
lg=LinearRegression()
poly=PolynomialFeatures(degree=3)

x_train_fit = poly.fit_transform(x_train) #transforming our input data
lg.fit(x_train_fit, y_train)
x_test_ = poly.fit_transform(x_test)
predicted = lg.predict(x_test_)

print("MSE: ", metrics.mean_squared_error(y_test, predicted))
print("R squared: ", metrics.r2_score(y_test,predicted))


# In[102]:


x= kc["sqft_living15"].values.reshape(-1,1)
y= kc["sqft_above"].values
poly = PolynomialFeatures(degree = 2) 
x_poly = poly.fit_transform(x) 
poly.fit(x_poly, y) 
lg=LinearRegression()
lg.fit(x_poly, y) 

plt.scatter(x, y, color="r")
plt.title("Linear regression")
plt.ylabel("Salary")
plt.xlabel("Age")
plt.plot(x, lg.predict(poly.fit_transform(x)), color="k") 


# In[ ]:




