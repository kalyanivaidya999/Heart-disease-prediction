#!/usr/bin/env python
# coding: utf-8

# ## Import the Requied Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## load the data

# In[2]:


df=pd.read_csv('heart.csv')


# In[3]:


df.head(10)


# 1) cp: chest pain type
#     — Value 0: asymptomatic
#     — Value 1: atypical angina
#     — Value 2: non-anginal pain
#     — Value 3: typical angina
# 2) restbps: The person’s resting blood pressure 
# 
# 3) chol: The person’s cholesterol measurement in mg/dl
# 
# 4) fbs: The person’s fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
# 
# 5) restecg: resting electrocardiographic results
#     — Value 0: showing probable or definite left ventricular hypertrophy by Estes’ criteria
#     — Value 1: normal
#     — Value 2: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
# 
# 6) thalach: The person’s maximum heart rate achieved
# 
# 7) exang: Exercise induced angina (1 = yes; 0 = no)
# 
# 8) oldpeak: ST depression induced by exercise relative to rest (‘ST’ relates to positions on the ECG plot)
# 
# 9) slope: the slope of the peak exercise ST segment — 0: downsloping; 1: flat; 2: upsloping
#     0: downsloping; 1: flat; 2: upsloping
# 
# 10) ca: The number of major vessels (0–3)
# 
# 11) thal: A blood disorder called thalassemia Value 0: NULL (dropped from the dataset previously
#     Value 1: fixed defect (no blood flow in some part of the heart)
#     Value 2: normal blood flow
#     Value 3: reversible defect (a blood flow is observed but it is not normal)
# 
# 13) target: Heart disease (1 = no, 0= yes)

# In[4]:


df.describe(include='all')


# In[5]:


df.info()


# ## Visualize the data

# In[7]:


c=df.corr()  # Use corr() function to find the correlation among the columns in the dataframe


# In[8]:


plt.figure(figsize=(8,8))  #figsize is a tuple of the width and height of the figure in inches

sns.heatmap(c, annot=True) #annot: If True, write the data value in each cell.


# ## Correlation 

# In[9]:


correlation=c["target"].sort_values(ascending=False) #there is no descending in function
correlation=pd.DataFrame(correlation)


# In[10]:


correlation


# In[11]:


#sns.pairplot(df,hue='target') #To plot multiple pairwise bivariate distributions in a dataset
#plt.show()


# In[106]:


df2 = df.copy()
def fun1(sex):
    if sex == 0:
        return 'female'
    else:
        return 'male'
df2['sex'] = df2['sex'].apply(fun1)
def fun2(prob):
    if prob == 0:
        return 'Heart Disease'
    else:
        return 'No Heart Disease'
df2['target'] = df2['target'].apply(fun2)


# In[110]:


plt.figure(figsize=(20,5))

plt.subplot(1,4,1)
sns.countplot(data= df2, x='sex',hue='target')
plt.title('Gender v/s target\n')

plt.subplot(1,4,2)
sns.histplot(data= df2, x='age', bins=6)
plt.title('age v/s target\n')

plt.subplot(1,4,3)
sns.countplot(data= df2,x='cp',hue='target')
plt.title('chest pain  v/s target\n')

plt.subplot(1,4,4)
sns.lineplot(data=df2,x='target',y='chol')
plt.title('cholestrol v/s target\n')

plt.show()


# ## dummy variables

# In[14]:


dataset=pd.get_dummies(df,columns=['cp']) #It converts categorical data into dummy or indicator variables.


# In[15]:


dataset.head(10)


# ## model 

# In[16]:


from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# In[17]:


model = LogisticRegression()


# In[18]:


x_df = dataset.drop('target',axis=1)
y_df=dataset['target']


# In[19]:


y_df.value_counts() #balanced it means that we can go with accuracy


# In[20]:


splits=ShuffleSplit(n_splits=10,test_size=0.20)
result=cross_val_score(model,x_df,y_df,cv=splits)


# In[25]:


result
np.mean(result)


# In[50]:


X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.20, random_state=90)


# In[51]:


model.fit(X_train,y_train)


# ## scale the data and model fitting

# In[52]:


scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_df)


# In[53]:


y_pred=model.predict(X_test)


# In[31]:


y_pred


# In[114]:


y_pred.shape


# In[111]:


y_test.shape


# In[113]:


accuracy_score(y_test,y_pred) #86 


# In[55]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


# In[58]:


from sklearn.metrics import roc_auc_score, roc_curve, f1_score,  classification_report
auc = roc_auc_score(y_test,y_pred)
auc


# In[59]:


fpr, tpr, thresholds = roc_curve(y_test,y_pred)

plt.plot(fpr , tpr , color='orange',label='ROC')
plt.plot([0,1],[0,1],color = 'darkblue',linestyle='--',label='ROC curve(area = %0.2f)'% auc)
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('receiver operating characterstics (ROC) curve')
plt.legend()
plt.show()


# In[60]:


f1_score(y_test,y_pred)


# In[61]:


print(classification_report(y_test,y_pred))


# In[62]:


from sklearn.metrics import mean_squared_error
error=mean_squared_error(y_test,y_pred)


# In[63]:


error


# square mean error for this model is 0.198 which is used to measure how close a fitted line is to actual data points. The lesser the Mean Squared Error, the closer the fit is to the data set.

# In[64]:


RMSE=mean_squared_error(y_test,y_pred,squared=False)


# In[65]:


RMSE


# RMSE is a used to measure the accuracy of model, but only to compare prediction errors of models 
# according rule of thumb, RMSE values between 0.2 and 0.5 shows that the model can relatively predict the data accurately and 
# this model has RMSE 0.44 which is good and show accuracy.
