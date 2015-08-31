rm(list=ls())
#install.packages('R.matlab')
#install.packages('mvtnorm')
library('R.matlab')
library('mvtnorm')
getwd()
intru <- as.data.frame(readMat("C:/Users/Administrator/Desktop/R/ex8data2.mat"))
#sample <- as.data.frame(readMat("C:/Users/Administrator/Desktop/R/ex8data1.mat"))
v <- ggplot(sample, aes(sample[,1], sample[,2], z = z))
par.sample <- estimateGuassian(sample[,1:2])
mu.sample <- par.sample[[1]]
sigma2.sample <- par.sample[[2]]
sigma2_tmp.sample <- diag(as.vector(sigma2.sample))
p.sample <- matrix(multivariateGuassian(sample[,1:2],mu.sample,sigma2_tmp.sample))

#graphs
hist(intru[,1],main = "histogram of X1")
hist(intru[,3],main = "histogram of X2")
qqnorm(X[,1])
plot(sample$X.1,sample$X.2,col='blue',xlab='throughput',ylab='latency')

#head(intru)
#hist(intru[,1])
#intru[,1:11]
#intru[,12:22]
#write.table(intru[,1:11],'train.csv',sep= ',')
#write.table(intru[,12:22],'xtest.csv',sep= ',')
#write.table(intru[,23],'ytest.csv',sep= ',')

#names(intru[12:22]) <- names(intru[1:11]) 
intru1 <- unname(intru)
X_train <-rbind(as.matrix(intru1[,1:11]),as.matrix(intru1[,12:22]))
X_train <- X_train[1:1400,]

identical(names(intru[,12:22]),names(intru[1:11]) )
#training set
X <- as.matrix(intru[,1:11],nrow=nrow(intru),ncol=11)

#test set
Xval <- as.matrix(intru[,12:22],nrow=nrow(intru),ncol=11)
yval <- as.vector(intru[,23])
plot(X[,1],X[,2],'p')
X_test <- Xval[401:1000,]
y_test <- yval[401:1000]

#function to estimate gaussian paramters

estimateGuassian <- function(X){
    size <- dim(X)
    m <- size[[1]]
    n <- size[[2]]
    mu <- matrix(0,n,1)
    sigma2 <- matrix(0,n,1)
    mu <- matrix(colMeans(X))
    sigma2_temp <- matrix(apply(X,2,var))
    sigma2 <- sigma2_temp*((m-1)/m)
    return(list(mu,sigma2))
}

parameter <- estimateGuassian(X)
mu <- parameter[[1]]
sigma2 <- parameter[[2]]

#Multivariate gaussian density function

multivariateGuassian <- function(X,mu,sigma2){
    size <- dim(X)
    m <- size[[1]]
    n <- size[[2]]
    p <- matrix(0,m,1)
    p <- dmvnorm(X,mu,sigma2)
    return(p)
    }


SelectThreshold <- function(yval,pval){
    bestepsilon = 0
    epsilon = mean(pval)
    bestf1 = 0
    f1 = 0
    stepsize = (max(pval)-min(pval))/1000
    
    is.integer0 <- function(x)
    {
        is.integer(x) && length(x) == 0L
    }
    
    for (epsilon in seq(min(pval),max(pval),stepsize)){
        prediction = ifelse(pval<epsilon,1,0)
        b <- data.frame(cbind(prediction,yval))
        colnames(b) <- c("prediction","yval")
        confusion <- data.frame(table(b))#$prediction,b$yval)
    
        tp=confusion[confusion$yval==1 & confusion$prediction==1,3]
        fp=confusion[confusion$yval==0 & confusion$prediction==1,3]
        fn=confusion[confusion$yval==1 & confusion$prediction==0,3]     
        
        tp = ifelse(is.integer0(tp),0,tp)
        fp = ifelse(is.integer0(fp),0,fp)
        fn = ifelse(is.integer0(fn),0,fn)
        
        
        if((tp+fp==0)||(tp+fn==0)){
            f1=0
        }else{
        prec <- tp/(tp+fp)
        rec <- tp/(tp+fn)
        f1 <- (2*prec*rec)/(prec+rec)
        }
    
        if(f1>bestf1){
            bestf1=f1
            bestepsilon=epsilon
        }else{
            bestf1=bestf1
            bestepsilon=bestepsilon
        }
     }
    return (list(bestf1,bestepsilon,prec,rec,prediction))
}


parameter <- estimateGuassian(X)
mu <- parameter[[1]]
sigma2 <- parameter[[2]]
sigma2_tmp <- diag(as.vector(sigma2))
p <- matrix(multivariateGuassian(X,mu,sigma2_tmp))
pval <- (multivariateGuassian(Xval,mu,sigma2_tmp))
e <- SelectThreshold(yval,pval)
epsilon <- e[[2]]
f1 <- e[[1]]
prec <- e[[3]]
rec <- e[[5]]
Prediction <- e[[5]]
combo <- as.data.frame(cbind(X,p))
colnames(combo[12]) <- "p"
outliers <- which(combo[12] < epsilon)
Anomaly <- combo[outliers,]


parameter.m1 <- estimateGuassian(X_train)
mu.m1 <- parameter.m1[[1]]
sigma2.m1 <- parameter.m1[[2]]
sigma2_tmp.m1 <- diag(as.vector(sigma2.m1))
p.m1 <- matrix(multivariateGuassian(X_train,mu.m1,sigma2_tmp.m1))
pval.m1 <- (multivariateGuassian(X_test,mu.m1,sigma2_tmp.m1))
e.m1 <- SelectThreshold(y_test,pval.m1)
epsilon.m1 <- e.m1[[2]]
f1.m1 <- e.m1[[1]]
prec.m1 <- e.m1[[3]]
rec.m1 <- e.m1[[5]]
Prediction.m1 <- e.m1[[5]]
combo.m1 <- as.data.frame(cbind(X_train,p.m1))
colnames(combo.m1[12]) <- "p"
outliers.m1 <- which(combo.m1[12] < epsilon.m1)
Anomaly.m1 <- combo.m1[outliers.m1,]



















