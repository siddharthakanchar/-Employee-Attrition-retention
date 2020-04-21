#!/usr/bin/env python
# coding: utf-8

# # Human Resource Data Case Study 

# In[1]:


import os 
os.getcwd()


# In[3]:


os.chdir("C:/Users/siddh/OneDrive/Desktop/robokalam")


# ## Importing required packages and Loading dataset

# In[4]:


import numpy as np
import pandas as pd 
from sklearn import preprocessing
df1=pd.read_csv("HR_comma_sep.csv")


# In[5]:


df1


# ## Checking for Null values

# In[8]:


df1.apply(lambda x:sum(x.isnull()),axis=0)


# ## Using lable encoder for role coloumn 

# In[23]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dfle=df1
dfle.role=le.fit_transform(df1.role)
df1


# ## Using Label encoder for salary coloumn

# In[35]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dfle=df1
dfle.salary=le.fit_transform(df1.salary)
df1


# ## Left cloumn unique vlaues count 

# In[36]:


df1.left.value_counts()


# ## role column unique values count 

# In[37]:


df1.role.value_counts()


# ## Salary coloumn unique values count 

# In[38]:


df1.salary.value_counts()


# ## promotionlast5years coloumn unique values clount 

# In[39]:


df1.promotion_last_5years.value_counts()


# In[111]:


df1.corr()


# ## Heatmap

# In[112]:


import seaborn as sns
corr=df1.corr()
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns)


# ## Features(inputs) Indexing

# In[40]:


x=df1.loc[:,['last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','satisfaction_level','promotion_last_5years','role','salary']]


# In[41]:


x


# In[42]:


x.describe()


# ## Label (output) indexing 

# In[43]:


y=df1.loc[:,['left']]


# In[44]:


y


# In[45]:


y.describe()


# ## Normalize the numeric vaiables from x (inputs) dateframe df1

# In[46]:


minmax=preprocessing.MinMaxScaler(feature_range=(0,1))
minmax.fit(x).transform(x)


# # KNN CLASSIFICATION ALGORITHM

# In[47]:


from sklearn import model_selection,neighbors 


# ## Train test Data split 

# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)


# In[48]:


x_train


# In[49]:


y_train


# ## Converting Datatype

# In[50]:


y_train=y_train.astype(int)


# In[51]:


y_train


# ## Converting data type 

# In[52]:


y_test=y_test.astype(int)


# In[53]:


y_test


# In[54]:


print("Actual test data:")
print(y_test.values)


# In[55]:


clf=neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)


# In[56]:


y_pred=clf.predict(x_test)


# In[57]:


print("\nPredicted test data:")
print(y_pred)


# In[61]:


from sklearn.metrics import accuracy_score, recall_score, roc_auc_score,confusion_matrix
print("\nAccuracy score:%f"%(accuracy_score(y_test,y_pred)*100))
print("Recall score:%f"%(recall_score(y_test,y_pred)*100))
print("Roc score:%f\n"%(roc_auc_score(y_test,y_pred)*100))
print(confusion_matrix(y_test,y_pred))


# # KNN CLASSIFICATION ALGORITHM Accuracy Prediction score 92.53333 %

# # CONFUSION MATRIX ,3183 Observations,False positives-233(partial classification) and False Negatives- 103(serious misclassification) 
# 
# 

# In[62]:


probas=clf.predict_proba(x_test)


# In[63]:


probas


# In[64]:


import matplotlib.pyplot as plt
plt.figure(dpi=150)


# In[65]:


plt.hist(probas,bins=20)
plt.title ('Classification probabilities')
plt.xlabel('probability')
plt.ylabel('# of instances')
plt.xlim([0.5,1.0])
plt.legend(y_test)
plt.show()


# In[66]:


x_train_std=minmax.fit_transform(x_train)
x_test_std=minmax.transform(x_test)


# In[67]:


x_train_std


# In[68]:


x_test_std


# In[69]:


from sklearn.model_selection import cross_val_score,cross_val_predict
clf_acc=cross_val_score(clf,x_train_std,y_train,cv=3,scoring="accuracy",n_jobs=-1)
clf_proba=cross_val_predict(clf,x_train_std,y_train,cv=3,method='predict_proba')
clf_scores=clf_proba[:,1]


# In[70]:


clf_acc


# # Knn classification Algorithm accuracy Increases  from 92% to 94 % using cross validation score¶

# # Logistic regression

# In[126]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
lr_acc=cross_val_score(lr,x_train_std,y_train,cv=3,scoring='accuracy',n_jobs=-1)
lr_proba=cross_val_predict(lr,x_train_std,y_train,cv=3,method='predict_proba')
lr_scores=lr_proba[:,1]


# In[127]:


lr_acc


# In[128]:


y_pred=lr.predict(x_test)


# In[129]:


y_pred


# In[130]:


print("Actual test data:")
print(y_test.values)


# In[131]:


print("\nPredicted test data:")
print(y_pred)


# In[132]:


from sklearn.metrics import accuracy_score, recall_score, roc_auc_score,confusion_matrix
print("\nAccuracy score:%f"%(accuracy_score(y_test,y_pred)*100))
print("Recall score:%f"%(recall_score(y_test,y_pred)*100))
print("Roc score:%f\n"%(roc_auc_score(y_test,y_pred)*100))
print(confusion_matrix(y_test,y_pred))


# # Logistic regression Accuracy Prediction score 76%

# # Confusion Matrix 3131 observations , 285-False Positives(Partial Classification) and 785 -False Neagatives (Serious Misclassifiacation)

# # support vector classification

# In[81]:


from sklearn.svm import SVC
svc=SVC(kernel='rbf',probability=True)
svc_classifier=svc.fit(x_train,y_train)


# In[82]:


from sklearn.svm import SVC
svc_acc=LogisticRegression()
svc_acc.fit(x_train,y_train)
svc_acc=cross_val_score(svc_classifier,x_train_std,y_train,cv=3,scoring='accuracy',n_jobs=-1)
svc_proba=cross_val_predict(svc_classifier,x_train_std,y_train,cv=3,method='predict_proba')
svc_scores=svc_proba[:,1]


# In[83]:


svc_acc


# In[84]:


y_pred=svc.predict(x_test)


# In[85]:


y_pred


# In[86]:


print("Actual test data:")
print(y_test.values)


# In[87]:


print("\nPredicted test data:")
print(y_pred)


# In[88]:


from sklearn.metrics import accuracy_score, recall_score, roc_auc_score,confusion_matrix
print("\nAccuracy score:%f"%(accuracy_score(y_test,y_pred)*100))
print("Recall score:%f"%(recall_score(y_test,y_pred)*100))
print("Roc score:%f\n"%(roc_auc_score(y_test,y_pred)*100))
print(confusion_matrix(y_test,y_pred))


# # support vector classification Accuracy prediction score - 94.75%

# # Confusion Matrix -3274 total observations , 142-False positive (partial classification) and 94-False Negative Clasification(Serious Misclassification)

# # Decision tree Classification

# In[90]:


from sklearn.tree import DecisionTreeClassifier
dtc_clf=DecisionTreeClassifier()
dtc_clf.fit(x_train,y_train)


# In[91]:


dtc_clf_acc=cross_val_score(dtc_clf,x_train_std,y_train,cv=3,scoring="accuracy",n_jobs=-1)


# In[92]:


dtc_proba=cross_val_predict(dtc_clf,x_train_std,y_train,cv=3,method="predict_proba")


# In[93]:


dtc_clf_scores=dtc_proba[:,1]


# In[94]:


dtc_clf_scores


# In[95]:


dtc_clf_acc


# In[96]:


y_pred=dtc_clf.predict(x_test)


# In[97]:


from sklearn.metrics import accuracy_score, recall_score, roc_auc_score,confusion_matrix
print("\nAccuracy score:%f"%(accuracy_score(y_test,y_pred)*100))
print("Recall score:%f"%(recall_score(y_test,y_pred)*100))
print("Roc score:%f\n"%(roc_auc_score(y_test,y_pred)*100))
print(confusion_matrix(y_test,y_pred))


# # Decision tree Classification Algorithm Prediction score - 97% 

# # Confusion Matrix 3341 total observations, 75-false Positives(partial classification) and 46 -true positives (serious misclassification)

# # random forest classifier¶

# In[98]:


from sklearn.ensemble import RandomForestClassifier
rmf=RandomForestClassifier(max_depth=3,random_state=0)
rmf_clf=rmf.fit(x_train,y_train)


# In[102]:


rmf_clf_acc=cross_val_score(rmf_clf,x_train_std,y_train,cv=3,scoring="accuracy",n_jobs=-1)


# In[103]:


rmf_proba=cross_val_predict(rmf_clf,x_train_std,y_train,cv=3,method='predict_proba')


# In[104]:


rmf_clf_scores=rmf_proba[:,1]


# In[105]:


rmf_clf_scores


# In[106]:


y_pred=rmf_clf.predict(x_test)


# In[107]:


y_pred


# In[108]:


from sklearn.metrics import accuracy_score, recall_score, roc_auc_score,confusion_matrix
print("\nAccuracy score:%f"%(accuracy_score(y_test,y_pred)*100))
print("Recall score:%f"%(recall_score(y_test,y_pred)*100))
print("Roc score:%f\n"%(roc_auc_score(y_test,y_pred)*100))
print(confusion_matrix(y_test,y_pred))


# # random forest classifier Accuracy prediction score-91%

# # Confusion Matrix - 3384 total obersvations , 32-False positives(Partial classification) and 363 True positives( serious misclassification) 

# In[109]:


from sklearn.metrics import roc_auc_score,roc_curve 
def roc_cur(title,y_train,scores,label=None):
    #caluculate the roc score
    fpr,tpr,thresholds=roc_curve(y_train,scores)
    print('AUC score({}):{:.2f}'.format(title,roc_auc_score(y_train,scores)))
    plt.figure(figsize=(8,6))
    plt.plot(fpr,tpr,linewidth=2,label=label)
    plt.xlabel('False positive rate',fontsize=16)
    plt.ylabel('true positive rate',fontsize=16)
    plt.title('roc curve:{}'.format(title),fontsize=16)
    plt.show()


# In[110]:


roc_cur('knn',y_train,clf_scores)
roc_cur('Logistic Regression',y_train,lr_scores)
roc_cur('support vector classification',y_train,svc_scores)
roc_cur('Decision tree Classifier',y_train,dtc_clf_scores)
roc_cur('Random forest classifier',y_train,rmf_clf_scores)


# # ALGORITHMS AND METRIC SCORES
# 
# ## KNNC-ACCURACY SCORE-94.7
# ####          RECALL SCORE-95.2
# ####          ROC SCORE-94.6
# 
# ## LR -ACCURACY SCORE-76.2%
# ####        RECALL SCORE-27.58%
# ####        ROC SCORE-59%
# 
# ## SVC-ACCURACY SCORE-94%
# ####         RECALL SCORE-91%
# ####         ROC SCORE-93%
# 
# 
# ## DTC-ACCURACY SCORE-97.3%
# ####         RECALL SCORE-95%
# ####         ROC SCORE-96%
# 
# ## RFC-ACCURACY SCORE-91%
# ####          RECALL SCORE-66%
# ####          ROC-82%

# # THANK YOU FOR THE OPPUTUNITY 

# ## NAME:K.V.S.S SIDDHARTHA
# 
# ## PHONE NUMBER:9054175175
# 
# ## EMAIL ID:siddharthakancharla@gmail.com
