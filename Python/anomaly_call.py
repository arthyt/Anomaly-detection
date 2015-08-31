

import os
os.getcwd()
os.chdir('C:\\Users\\Administrator\\Desktop\\Python')

import math
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.metrics import confusion_matrix
import anomalydensity as an #user defined module


#reading csv with pandas as np.loadtxt throws error if column names are present
xtrain = pd.read_csv('xtrain.csv',delimiter=',') #1000*11 training set
xtest = pd.read_csv('xtest.csv',delimiter=',') #1000*11 testing set
ytest = pd.read_csv('ytest.csv',delimiter=',') #1000*1 original y values

#converting from pandas dataframe to numpy array
xtrain1 =np.array(xtrain) 
xtest1 = np.array(xtest)
ytest1 = np.array(ytest)

#creating object1
b = an.anomaly(xtrain1) #anomaly is a class inside anomalydensity module
par = b.parameter()
p_density=b.multivariateGaussian(par[0],par[2])

#creating object2
c=an.anomaly(xtest1)
pval_density=c.multivariateGaussian(par[0],par[2])

print p_density.shape,pval_density.shape
type(p_density),type(pval_density)

epsilon = np.mean(pval_density)
bestepsilon = 0.
bestf1=0.
f1=0.
pred=np.empty([1000,])
yhat=np.empty([1000,])
stepsize = (np.amax(pval_density)-np.amin(pval_density))/1000


for epsilon in np.arange(np.amin(pval_density),np.amax(pval_density),stepsize):
    yhat = c.predict(pval_density,epsilon,pred)
    cm = confusion_matrix(ytest1, yhat) #misclassification matrix from sklearn
    fp = cm[0,1] #false positive
    tp = cm[1,1] #true positive
    fn = cm[1,0] #false negative
    f1 = c.validate(fp,tp,fn) #f1 score
    if(f1>bestf1):
        bestf1=f1
        bestepsilon=epsilon
    else:
        bestf1=bestf1
        bestepsilon=bestepsilon

print bestf1,bestepsilon
pfin = np.empty([1000,])
pfinal = b.predict(p_density,bestepsilon,pfin)
print "Number of anomalies found:",np.sum(pfinal,0)
xfinal = np.insert(xtrain1,11,p_density,axis=1)
xfinal1 = np.insert(xfinal,12,pfinal,axis=1)
#np.sum(xfinal1[:,12],0)
outliers= pd.DataFrame(xfinal1[xfinal1[:,12] == 1])
#outliers.shape # should indicate 117 outliers in data
outliers.to_csv('C:\\Users\\Administrator\\Desktop\\Python\\output.csv',header = ['id','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12'])


