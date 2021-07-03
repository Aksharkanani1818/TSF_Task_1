#!/usr/bin/env python
# coding: utf-8

# # Grip : The Spark Foundation
# 
# Data Science and Business Analytics Intern
# 
# Author : Akshar Kanani
# 
# Task 1 : Prediction using Supervised ML
# abcd

# # Step - 1 : Importing the dataset

# In[9]:


# Importing required libraries

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# To ignore the warnings 

import warnings as wg
wg.filterwarnings("ignore")


# In[10]:


url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
df = pd.read_csv(url)
df.head()


# In[11]:


df.tail()


# In[12]:


# check how many rows & columns are present
df.shape


# In[13]:


# For more info about dataset
df.info()


# In[15]:


# We can also give df.describe through describe
df.describe()


# In[17]:


# Now we will check if our dataset contains null or missing values
df.isnull().sum()


# # Step - 2 : Visulizing the dataset

# In[20]:


# Ploting

plt.rcParams["figure.figsize"] = [16,9]
df.plot(x='Hours', y='Scores', style = '*', color = 'blue', markersize=10)
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.grid()
plt.show()

#From the graph above we can observe that there is a linear relationship between "hours studied" and "percentage score". So we can use the linear regression supervised machine model on it to predct further values.
# In[21]:


# We can aslo use .corr to determine the corelation between the variables.
df.corr()


# # Step - 3 : Data preparation
'''In this step we will divide the data into "features" (inputs) and "labels" (outputs). 
After that we will split whole dataset into 2 parts : 1. Testing data 
                                                      2. Training data'''
# In[23]:


df.head()


# In[24]:


# using iloc function we will divide the data
x = df.iloc[ :, :1].values
y = df.iloc[ :, 1:].values


# In[25]:


x


# In[32]:


# Spliting data into training and testing data

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                            test_size=0.2, random_state=0 )


# # Step - 4 : Training the Algorithm
#We have splited our data into training and testing sets, and now we will train our model.
# In[34]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)


# # Step - 5 : Visulizing the model

# In[53]:


line = model.coef_*x + model.intercept_

# Plotting for the training data

plt.rcParams["figure.figsize"] = [16,9]
plt.scatter(x_train, y_train, color='red')
plt.plot(x, line, color='m')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.grid()
plt.show()


# In[52]:


# plotting for the testing data

plt.rcParams["figure.figsize"] = [16,9]
plt.scatter(x_test, y_test, color='g')
plt.plot(x, line, color='m')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.grid()
plt.show()


# # Step - 6 : Marking Predictions

#Now that we have trained our algorithm, its time to make some pridictions.
# In[42]:


print(x_test)   # Testing data - In hours
y_pred = model.predict(x_test)  # Predicting the scores


# In[43]:


# Comparing Actual vs Predicted
y_test


# In[44]:


y_pred


# In[45]:


# Comparing Actual vs Predicted
comp = pd.DataFrame({'Actual' : [y_test], 'Predicted' : [y_pred] })
comp


# In[46]:


# Testing with own data

hours = 9.25
own_pred = model.predict([[hours]])
print("The predicted score if a person studies for",hours,"hours is",own_pred[0])

#Hence it can be concluded that the predicted score if a person studies for 9.25 hours is 93.69173248737538
# # Step - 7 : Evaluating the model
#In the last step, we are going to evaluate our trained modelby calculating mean absolute error

# In[50]:


from sklearn import metrics 
print("Mean Absolute Error : ", metrics.mean_absolute_error(y_test, y_pred))


# # Thank You :)

# In[ ]:




