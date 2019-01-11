# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 14:13:02 2018

@author: Nancy Wu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import linear_model
get_ipython().magic('matplotlib inline')

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

def classification_error(y, yhat):
    return np.mean(y!=yhat)

credit_data = pd.read_excel('C:\\Users\\Nancy Wu\\CPSC 340 Assignments\\t4l0b_a6\\data\\default_of_credit_card_clients.xls', header=1)


X = credit_data.iloc[:,1:24]
X = X.as_matrix()

y = credit_data.iloc[:,[-1]]
y = y.as_matrix()

subset = credit_data.iloc[:,[1,2,3,4,5]]
subset.describe()

plt.hist(np.log(X[:,0]))
sns.lmplot(x='BILL_AMT1',y="PAY_AMT1",data=credit_data, fit_reg=False, hue='default payment next month', legend='default payment next month')
sns.lmplot(x='BILL_AMT2',y="PAY_AMT2",data=credit_data, fit_reg=False, hue='default payment next month', legend='default payment next month')
sns.lmplot(x='BILL_AMT3',y="PAY_AMT3",data=credit_data, fit_reg=False, hue='default payment next month', legend='default payment next month')
sns.lmplot(x='BILL_AMT4',y="PAY_AMT4",data=credit_data, fit_reg=False, hue='default payment next month', legend='default payment next month')
sns.lmplot(x='BILL_AMT5',y="PAY_AMT5",data=credit_data, fit_reg=False, hue='default payment next month', legend='default payment next month')
sns.lmplot(x='BILL_AMT6',y="PAY_AMT6",data=credit_data, fit_reg=False, hue='default payment next month', legend='default payment next month')

fig, axs = plt.subplots(2,3)
sns.countplot(x="PAY_0",data=credit_data, hue="default payment next month", ax=axs[0,0])
sns.countplot(x="PAY_2",data=credit_data, hue="default payment next month", ax=axs[0,1])
sns.countplot(x="PAY_3",data=credit_data, hue="default payment next month", ax=axs[0,2])
sns.countplot(x="PAY_4",data=credit_data, hue="default payment next month", ax=axs[1,0])
sns.countplot(x="PAY_5",data=credit_data, hue="default payment next month", ax=axs[1,1])
sns.countplot(x="PAY_6",data=credit_data, hue="default payment next month", ax=axs[1,2])

sns.lmplot(x="AGE", y="LIMIT_BAL", data=credit_data, fit_reg=False, hue='default payment next month', legend='default payment next month')
sns.countplot(x="SEX", hue="default payment next month", data=credit_data)
sns.countplot(x="MARRIAGE", hue="default payment next month", data=credit_data)
sns.countplot(x="EDUCATION", hue="default payment next month", data=credit_data)

### shuffle the data ###

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, random_state=1)

y_test = np.squeeze(y_test)
y_train = np.squeeze(y_train)
y_val = np.squeeze(y_val)

### get baseline model ###

baseline = DummyClassifier(strategy = 'most_frequent',random_state=1)

baseline.fit(X_train,y_train)
#baseline.score(X_val,y_val) # = 1-np.mean(y_val!=yhat) (1 - classification error)
yhat = baseline.predict(X_val)
classification_error(y_val, yhat)

# error: 0.2202

### try logisitic regression ###

# with L2 regularization
log_l2 = LogisticRegression()

log_l2.fit(X_train,y_train)
log_l2_yhat = log_l2.predict(X_val)
classification_error(y_val, log_l2_yhat)

# error: 0.2202

# with L1 regularization
log_l1 = LogisticRegression(penalty='l1')
log_l1.fit(X_train,y_train)
log_l1_yhat = log_l1.predict(X_val)
classification_error(y_val, log_l1_yhat)

# error: 0.191

## plot validation error vs regularization strength L2

regularization = np.array([0.01,0.1,1,10])

valid_score = []

for i in regularization:
    log_l1 = LogisticRegression(penalty='l1', C=i)
    log_l1.fit(X_train,y_train)
    log_l1_yhat = log_l1.predict(X_val)
    valid_score.append(classification_error(y_val, log_l1_yhat))
    
plt.plot(regularization,valid_score)

## choose C=1


####### change some features #####

## regroup pays ##

for i in range(5,11):
    X_train[:,i][X_train[:,i] < 0]=0
    X_val[:,i][X_val[:,i] < 0]=0


log_l1 = LogisticRegression(penalty='l1')
log_l1.fit(X_train,y_train)
log_l1_yhat = log_l1.predict(X_val)
classification_error(y_val, log_l1_yhat)

#error 0.1815

## regroup education ##

#X_train[:,2][X_train[:,2] == 0]=4
#X_train[:,2][X_train[:,2] > 4]=4
#X_val[:,2][X_val[:,2] == 0]=4
#X_val[:,2][X_val[:,2] > 4]=4
#
#
#log_l1 = LogisticRegression(penalty='l1')
#log_l1.fit(X_train,y_train)
#log_l1_yhat = log_l1.predict(X_val)
#classification_error(y_val, log_l1_yhat)
#
### regroup marriage ##
#
#X_train[:,3][X_train[:,3] == 0]=3
#X_val[:,3][X_val[:,3] == 0]=3
#
#log_l1 = LogisticRegression(penalty='l1')
#log_l1.fit(X_train,y_train)
#log_l1_yhat = log_l1.predict(X_val)
#classification_error(y_val, log_l1_yhat)



### encode marriage categories as dummies
lb = preprocessing.LabelBinarizer()
lb.fit(X_train[:,3])

dummies = lb.transform(X_train[:,3])
X_dummies = np.delete(X_train,3,1)
X_dummies = np.concatenate((X_dummies,dummies), axis=1)

val_dummies = lb.transform(X_val[:,3])
X_val_dummies = np.delete(X_val,3,1)
X_val_dummies = np.concatenate((X_val_dummies,val_dummies), axis=1)

log_l1_dummies = LogisticRegression(penalty='l1')
log_l1_dummies.fit(X_dummies,y_train)
log_l1_yhat = log_l1_dummies.predict(X_val_dummies)
classification_error(y_val, log_l1_yhat)

(log_l1_dummies.coef_ != 0).sum()

#error 0.1813

##try to add variable 

#sept_payment = (X_dummies[:,16] > 0).astype(int)
#sept_payment = sept_payment.reshape((len(sept_payment),1))
#
#
#X_made_payment = np.concatenate((X_dummies, sept_payment), axis=1)
#
#
#sept_val_payment = (X_val_dummies[:,16] > 0).astype(int)
#sept_val_payment = sept_val_payment.reshape((len(sept_val_payment),1))
#X_val_made_payment = np.concatenate((X_val_dummies, sept_val_payment), axis=1)
#
#log_l2_made_payment = LogisticRegression(penalty='l2')
#log_l2_made_payment.fit(X_made_payment,y_train)
#log_l2_yhat = log_l2_made_payment.predict(X_val_made_payment)
#classification_error(y_val, log_l2_yhat)

####### SVM #######

#try svc with just rbf kernal, C = 1.0

svm_model = svm.SVC()
svm_model.fit(X_train,y_train)
svm_yhat=svm_model.predict(X_val)
classification_error(y_val, svm_yhat)

#error: 0.2197 ... this is too slow

####### Random Forest ######

# try random forest , n_estimators (number of trees) = 10, max_depth = 10
rf_model = RandomForestClassifier(n_estimators = 10, max_depth = 10)
rf_model.fit(X_dummies,y_train)
rf_yhat = rf_model.predict(X_val_dummies)
classification_error(y_val,rf_yhat)

# error = 0.1823

# optimize the hyperparameters for random forest, this doesn't make much sense
# since the errors will change because of random bootstrap samples.
parameters = {'n_estimators':[10,20,30,40], 'max_depth':[5,7,10,13,15]}
rf = RandomForestClassifier()
search_param = GridSearchCV(rf,parameters)
search_param.fit(X_dummies,y_train)

search_param.best_estimator_
# max_depth = 7, n_estimators=40

parameters = {'n_estimators':[20], 'max_depth':[5,6,7,8,9,10,11,12,13,14,15]}
rf = RandomForestClassifier()
search_param = GridSearchCV(rf,parameters)
search_param.fit(X_dummies,y_train)

search_param.best_estimator_

# max_depth = 7

## refit with best params
rf_model = RandomForestClassifier(n_estimators = 20, max_depth = 7)
rf_model.fit(X_dummies,y_train)
rf_yhat = rf_model.predict(X_val_dummies)
classification_error(y_val,rf_yhat)

#error = 0.1767


####### Neural Net #######

# try a neural net with hidden_layer_sizes = 100
nn_model = MLPClassifier(hidden_layer_sizes=(100,),activation='logistic')
nn_model.fit(X_dummies,y_train)
1-nn_model.score(X_val_dummies,y_val)

# error: 0.2202

# optimize the hidden_layer_sizes

parameters = {'hidden_layer_sizes':[(100,),(100,10),(100,50),(100,100)]}
nn = MLPClassifier()
search = GridSearchCV(nn,parameters)
search.fit(X_dummies,y_train)

search.best_estimator_
#(100,)

nn_opt_yhat = search.predict(X_val_dummies)
classification_error(y_val,nn_opt_yhat)
# error: 0.2202

###### Feature Selection ######

# L0 logistic regression from linear_model with foward selection

model_l0 = linear_model.logRegL0(L0_lambda=1.0, maxEvals=400)
model_l0.fit(X_dummies,y_train)
print("# nonZeros: %d" % (model_l0.w != 0).sum())
classification_error(y_val,model_l0.predict(X_val_dummies))

# error: 0.7798


### FINAL TEST ERROR ###

for i in range(5,11):
    X_test[:,i][X_test[:,i] < 0]=0

dummies = lb.transform(X_test[:,3])
X_test_dummies = np.delete(X_test,3,1)
X_test_dummies = np.concatenate((X_test_dummies,dummies), axis=1)

rf_yhat = rf_model.predict(X_test_dummies)
print("classification error: %.4f" %classification_error(y_test,rf_yhat))






