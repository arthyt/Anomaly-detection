import math
import numpy as np

class anomaly(object):
    
    def __init__(self,X): #X is np array - 1000*11
        self.X = X
    
    def parameter(self):
        mean = np.mean(self.X,axis=0) #column mean - 11
        variance = np.var(self.X,axis=0) #column variance - 11
        #vcv = np.cov((X).T)    #with covariance
        vcv =np.diag(variance) #covariance is 0
        #vcv = (vcv_temp/m)*(m-1) #use in case if sample std is returned
        return([mean,variance,vcv])
    
    def multivariateGaussian(self,mean,vcv): #can change mean and vcv
        n = self.X.shape[1] #size of column - 11
        if (len(mean) == n and vcv.shape == (n,n)): #vcv should be square matrix
            determinant = np.linalg.det(vcv)
            if (determinant == 0):
                raise ValueError("Variance-Covariance-Matrix is singular and cannot be inverted")
            else:
                constant = (math.pow((2*math.pi),(-0.5*n)))*(math.pow(determinant,-0.5))
                # ** gives incorrect results and so use pow
                inverse = np.linalg.inv(vcv) #11*11 matrix
                meandev = (self.X-mean) #1000 * 11 matrix
                #matrix mult of meandev-inverse and array mult of the result
                exponential = np.exp(-0.5*np.sum(((np.dot(meandev,inverse))*meandev),axis=1))
                density = constant*exponential
                return density
        else:
            raise ValueError("The input dimensions do not match")
    
    def predict(self,density,epsilon,pred):
        for i in range(0,density.shape[0]):
            if(density[i]>epsilon):
                pred[i] = 0
            else:
                pred[i] =1
        return pred
    
    def validate(self,fp,tp,fn):
        try:
            precision = float(tp)/(tp+fp)
            recall = float(tp)/(tp+fn)
            f1 = float(2*precision*recall)/(precision+recall)
        except ZeroDivisionError:
        #print "Division by zero in the calculation of F1 score" 
        #interested in high f1 score and so exception intimation is not required
            f1=0
        return f1

        