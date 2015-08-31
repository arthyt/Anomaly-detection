import os
os.getcwd()
os.chdir('C:\\Users\\Administrator\\Desktop\\Python')

import math
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.metrics import confusion_matrix
import anomalydensity as an #user defined module
import matplotlib
import matplotlib.pyplot as plt

#reading csv with pandas as np.loadtxt throws error if column names are present
xtrain_s = pd.read_csv('xtrain_sam.csv',delimiter=',') #1000*11 training set
xtest_s = pd.read_csv('xtest_sam.csv',delimiter=',') #1000*11 testing set
ytest_s = pd.read_csv('ytest_sam.csv',delimiter=',') #1000*1 original y values

#converting from pandas dataframe to numpy array
xtrain2 =np.array(xtrain_s) 
xtest2 = np.array(xtest_s)
ytest2 = np.array(ytest_s)

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

#Initializing the plot
xplt = np.arange(0, 36, 0.5)
yplt = np.arange(0,36,0.5)
X, Y = np.meshgrid(xplt, yplt)
H = np.column_stack((X.flatten(1),Y.flatten(1)))

#creating object1
b1 = an.anomaly(xtrain2) #anomaly is a class inside anomalydensity module
par1 = b1.parameter()
p_density1=b1.multivariateGaussian(par1[0],par1[2])

#creating object for plot
plt1 = an.anomaly(H) #anomaly is a class inside anomalydensity module
plt_density=plt1.multivariateGaussian(par1[0],par1[2])

plt_density.resize((72,72))
plt_density.shape
levels = [1.0000e-020,1.0000e-017,1.0000e-014,1.0000e-011,1.0000e-008,1.0000e-005,1.0000e-002]
fig=plt.figure(figsize=(30,20))
ax1 = fig.add_subplot(121)
ax1.scatter((xtrain2[:,0]),(xtrain2[:,1]),color='blue',s=5,edgecolor='none')
ax1.set_xlabel('Latency')
ax1.set_ylabel('Throughput')
ax1.set_aspect(1./ax1.get_data_ratio()) # make axes square
plt.hold(True)
plt.title('Contour plot')
plt.contour(X, Y, plt_density,levels)


#creating object2
c1=an.anomaly(xtest2)
pval_density1=c1.multivariateGaussian(par1[0],par1[2])

print p_density1.shape,pval_density1.shape
type(p_density1),type(pval_density1)

epsilon1 = np.mean(pval_density1)
bestepsilon1 = 0.
bestf11=0.
f11=0.
pred1=np.empty([307,])
yhat1=np.empty([307,])
stepsize1 = (np.amax(pval_density1)-np.amin(pval_density1))/307


for epsilon1 in np.arange(np.amin(pval_density1),np.amax(pval_density1),stepsize1):
    yhat1 = c1.predict(pval_density1,epsilon1,pred1)
    cm1 = confusion_matrix(ytest2, yhat1) #misclassification matrix from sklearn
    fp1 = cm1[0,1] #false positive
    tp1 = cm1[1,1] #true positive
    fn1 = cm1[1,0] #false negative
    f11 = c1.validate(fp1,tp1,fn1) #f1 score
    if(f11>bestf11):
        bestf11=f11
        bestepsilon1=epsilon1
    else:
        bestf11=bestf11
        bestepsilon1=bestepsilon1

print bestf11,bestepsilon1
pfin1 = np.empty([307,])
pfinal1 = b1.predict(p_density1,bestepsilon1,pfin1)
print "number of anomalies found:",np.sum(pfinal1,0)
xfinal_s = np.insert(xtrain2,2,p_density1,axis=1)
xfinal1_s = np.insert(xfinal_s,3,pfinal1,axis=1)
#np.sum(xfinal1[:,12],0)


v = xfinal1_s[:,3]
out =np.array(np.nonzero(v))
plt.hold(True)
plt.scatter(xfinal1_s[out,0],xfinal1_s[out,1],marker='o',s=8,color='red')
plt.show()
outliers1= pd.DataFrame(xfinal1_s[xfinal1_s[:,3] == 1])

#outliers.shape # should indicate 117 outliers in data
outliers1.to_csv('C:\\Users\\Administrator\\Desktop\\Python\\output_sam.csv',header = ['x1','x2','x3','x4'])


