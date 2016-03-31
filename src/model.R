library(e1071)
library(kernlab)
library(caret)
library(ada)
library(pROC)
set.seed(123)

######################
##   final output   ##
######################
# using pca outputs
fin.test <- dfEvaluatePCA[is.na(dfEvaluatePCA$target),]
fin.mdl <- dfEvaluatePCA[!is.na(dfEvaluatePCA$target),]
# using original features
fin.test <- data.mdl[is.na(data.mdl$target),]
fin.mdl <- data.mdl[!is.na(data.mdl$target),]

#------------------------------------------------------
# MODEL: SVM from e1071 package
# submit000, 001, 002
ptm <- proc.time()
# svmFit.fin <- svm(target ~., data=fin.mdl, kernel="linear", probability=TRUE)
svmFit.fin <- svm(target ~., data=fin.mdl, probability=TRUE)
svmPTM.fin <- proc.time()-ptm
pred.svm.fin <- predict(svmFit.fin, fin.test[-1], probability=TRUE)
prob.svm.fin <- round(attr(pred.svm.fin, "probabilities"),4)
write.csv(prob.svm.fin, "submit.csv")

#------------------------------------------------------
# MODEL: SVM from kernlab package
# submit003
rbf <- rbfdot(sigma=0.1)
ptm <- proc.time()
ksvmFit.fin <- ksvm(target~.,data=fin.mdl,type="C-bsvc",
                    kernel=rbf,C=10,prob.model=TRUE)
ksvmPTM.fin <- proc.time()-ptm
prob.ksvm.fin <- predict(object=ksvmFit.fin, fin.test[-1], type="prob" )
prob.ksvm.fin.adj <- prob.ksvm.fin
prob.ksvm.fin.adj[prob.ksvm.fin.adj<0.0001] <- 0.0001  # some probs were negative
prob.ksvm.fin.adj <- round(prob.ksvm.fin.adj/rowSums(prob.ksvm.fin.adj),4)
  
write.csv(prob.ksvm.fin.adj, "submit.csv")

#------------------------------------------------------
# MODEL: knn from caret package
# submit004
ctrl <- trainControl(method="cv", classProbs=TRUE)
ptm <- proc.time()
knnFit.fin <- train(target ~., data=fin.mdl, method="knn", trControl=ctrl)
knnPTM.fin <- proc.time()-ptm
prob.knn.fin <- predict(object=knnFit.fin, fin.test[-1], type="prob" )
write.csv(prob.knn.fin, "submit.csv")


######################
##    test model    ##
######################
ind <- sample(2, nrow(fin.mdl), replace=TRUE, prob=c(0.67, 0.33))  # train on % of data
data.train <- fin.mdl[ind==1, ]
data.test <- fin.mdl[ind==2, ]

#------------------------------------------------------
# using caret multiclass methods
ctrl <- trainControl(method="cv", classProbs=TRUE)
ctrl <- trainControl(method="cv", classProbs=TRUE, summaryFunction=multiClassSummary)
ptm <- proc.time()
glmFit <- train(target ~., data=data.train, method="glmnet", trControl=ctrl)
glmPTM <- proc.time()-ptm  # measures runtime
knnFit <- train(target ~., data=data.train, method="knn", trControl=ctrl)
knnPTM <- proc.time()-glmPTM  # measures runtime
# rdaFit takes a really long time to run
# rdaFit <- train(target ~., data=data.train, method="rda", trControl=ctrl)

prob.glm <- predict(object=glmFit, data.test[-1], type="prob" )
prob.knn <- predict(object=knnFit, data.test[-1], type="prob" )
pred.glm <- predict(object=glmFit, data.test[-1] )
pred.knn <- predict(object=knnFit, data.test[-1] )
# probAUC.glmnet <- multiclass.roc(data.test$target, pred.glm)
eval.glm <- postResample(pred=pred.glm, obs=data.test$target)
eval.knn <- postResample(pred=pred.knn, obs=data.test$target)

dev.off()
pdf('plots.pdf')
for(stat in c('Accuracy', 'Kappa', 'AccuracyLower', 'AccuracyUpper', 'AccuracyPValue', 
              'Sensitivity', 'Specificity', 'Pos_Pred_Value', 
              'Neg_Pred_Value', 'Detection_Rate', 'ROC', 'logLoss')) {
  
  print(plot(knnFit, metric=stat))
}
dev.off()

allModels <- list(glm = glmFit,
                  knn = knnFit )
probs <- extractProb(allModels, testX=data.test[-1], testY=data.test$target)

#------------------------------------------------------
# using SVM from kernlab package
# http://www.inside-r.org/node/63499

## Create a kernel function using the build in rbfdot function
rbf <- rbfdot(sigma=0.1)

## train a bound constraint support vector machine
ptm <- proc.time()
ksvmFit <- ksvm(target~.,data=data.train,type="C-bsvc",
                kernel=rbf,C=10,prob.model=TRUE)
ksvmPTM <- proc.time()-ptm

prob.ksvm <- predict(object=ksvmFit, data.test[-1], type="prob" )
pred.ksvm <- predict(object=ksvmFit, data.test[-1] )
eval.ksvm <- postResample(pred=pred.ksvm, obs=data.test$target)

#------------------------------------------------------
# using gbm
# https://github.com/harrysouthworth/gbm
library(devtools)
install_github("harrysouthworth/gbm")

ptm <- proc.time()
gbmFit <- gbm ... # return here
gbmPTM <- proc.time()-ptm

prob.gbm <- predict(object=gbmFit, data.test[-1], type="prob" )
pred.gbm <- predict(object=gbmFit, data.test[-1] )
eval.gbm <- postResample(pred=pred.gbm, obs=data.test$target)

#------------------------------------------------------
# using adaboost
myModels <- oneVsAll(data.train[-1], data.train$target, ada)
preds <- predict.oneVsAll(myModels, data.test[-1], type='probs')
preds <- data.frame(lapply(preds, function(x) x[,2])) #Make a data.frame of probs
#------------------------------------------------------
# helper functions to turn multiclass model into 2-class models
# gbm is a 2-class model
oneVsAll <- function(X, Y, FUN, ...) {
  models <- lapply(unique(Y), function(x) {
    name <- as.character(x)
    .Target <- factor(ifelse(Y==name,name,'other'), levels=c(name,'other'))
    dat <- data.frame(.Target, X)
    model <- FUN(.Target~., data=dat, ...)
    return(model)
  })
  names(models) <- unique(Y)
  info <- list(X=X, Y=Y, classes=unique(Y))
  out <- list(models=models, info=info)
  class(out) <- 'oneVsAll'
  return(out)
}
predict.oneVsAll <- function(object, newX=object$info$X, ...) {
  stopifnot(class(object)=='oneVsAll')
  lapply(object$models, function(x) {
    predict(x, newX, ...)
  })
}
classify <- function(dat) {
  out <- dat/rowSums(dat)
  out$Class <- apply(dat, 1, function(x) names(dat)[which.max(x)])
  out
}
#------------------------------------------------------
# Multi-Class Summary Function, based on caret:::twoClassSummary
# http://www.r-bloggers.com/error-metrics-for-multi-class-problems-in-r-beyond-accuracy-and-kappa/
require(compiler)
multiClassSummary <- cmpfun(function (data, lev = NULL, model = NULL) {

  #Load Libraries
  require(Metrics)
  require(caret)
  
  #Check data
  if (!all(levels(data[, "pred"]) == levels(data[, "obs"]))) 
    stop("levels of observed and predicted data do not match")
  
  #Calculate custom one-vs-all stats for each class
  prob_stats <- lapply(levels(data[, "pred"]), function(class){
    
    #Grab one-vs-all data for the class
    pred <- ifelse(data[, "pred"] == class, 1, 0)
    obs  <- ifelse(data[,  "obs"] == class, 1, 0)
    prob <- data[,class]
    
    #Calculate one-vs-all AUC and logLoss and return
    cap_prob <- pmin(pmax(prob, .000001), .999999)
    prob_stats <- c(auc(obs, prob), logLoss(obs, cap_prob))
    names(prob_stats) <- c('ROC', 'logLoss')
    return(prob_stats) 
  })
  prob_stats <- do.call(rbind, prob_stats)
  rownames(prob_stats) <- paste('Class:', levels(data[, "pred"]))
  
  #Calculate confusion matrix-based statistics
  CM <- confusionMatrix(data[, "pred"], data[, "obs"])
  
  #Aggregate and average class-wise stats
  #Todo: add weights
  class_stats <- cbind(CM$byClass, prob_stats)
  class_stats <- colMeans(class_stats)
  
  #Aggregate overall stats
  overall_stats <- c(CM$overall)
  
  #Combine overall with class-wise stats and remove some stats we don't want 
  stats <- c(overall_stats, class_stats)
  stats <- stats[! names(stats) %in% c('AccuracyNull', 
                                       'Prevalence', 'Detection Prevalence')]
  
  #Clean names and return
  names(stats) <- gsub('[[:blank:]]+', '_', names(stats))
  return(stats)
  
})

#------------------------------------------------------
# neural network model
library(nnet)

nnFit <- nnet(target ~., data=data.train, size=20, maxit=100, decay=0.001, MaxNWts=8000, probability=TRUE)
pred.nn <- predict(nnFit, newdata=data.test[-1], type="class")
postResample(pred.nn, data.test$target)  # Acc:0.7766 K:0.7268 e:23194
# size=50,maxit=1000,decay=0.001:        # Acc:0.7694 K:0.7204 e:14269
# size=20,maxit=100, decay=0.001:        # Acc:0.7827 K:0.7347 e:23155 <-- best
# size=25,maxit=100, decay=0.001:        # Acc:0.7792 K:0.7298 e:22638
# size=15,maxit=100, decay=0.001:        # Acc:0.7746 K:0.7250 e:22956
prob.nn <- round(predict(nnFit, newdata=data.test[-1], probability=TRUE),4)

multinomFit <- multinom(target ~., data=data.train, maxit=1000, trace=TRUE)
pred.multinom <- predict(multinomFit, newdata=data.test[-1], type="class")
prob.multinom <- predict(multinomFit, newdata=data.test[-1], type="prob")
postResample(pred.multinom, data.test$target)  # Acc:0.7673, K:0.7138, e:25539

# test final model
# nnFit <- nnet(target ~., data=fin.mdl, size=9, maxit=1000, probability=TRUE)

ptm <- proc.time()
nnFit <- nnet(target ~., data=fin.mdl, size=20, maxit=10000, decay=0.001, MaxNWts=2500, probability=TRUE)  # converged at 2180
nnPTM.fin <- proc.time()-ptm
pred.nn <- predict(nnFit, newdata=fin.test[-1], probability=TRUE)
prob.nn <- pred.nn

prob.nn[prob.nn<0.0001] <- 0.0001  # some probs were negative
prob.nn <- round(prob.nn/rowSums(prob.nn),4)

write.csv(prob.nn, "submit.csv")

#------------------------------------------------------
# boosting tree model
load("otto.RData")
library(gbm)
set.seed(123)
gbmFit.numbrcrunch <- gbm(target ~ ., data=data.train, distribution="multinomial", n.trees=1000,
                          shrinkage=0.05, interaction.depth=12, cv.folds=2)