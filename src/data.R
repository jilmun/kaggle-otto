library(ggvis)
library(ggplot2)
library(grid)
library(reshape)
library(class)
library(gmodels)
library(corrplot)
library(caret)
library(FactoMineR)

library(ROCR)
library(Metrics)
library(pROC)
library(e1071)

setwd("U:/Desktop Files/My Stuff/otto")
setwd("D:/kaggle/otto")
setwd("C:/Users/Janet/Desktop/Data Projects/Kaggle/OttoGroup/Working_JW")
save.image("otto.RData")
load("otto.RData")

set.seed(123)

orig.train <- read.csv("train.csv")
orig.test <- read.csv("test.csv")
orig.test$target <- factor(rep(NA, nrow(orig.test)), levels = row.names(table(orig.train$target)))

data.norm <- as.data.frame(lapply(rbind(orig.train[2:94],orig.test[2:94]), normalize))  # normalize features
targets <- data.frame(
  dataType = c(rep("Training",nrow(orig.train)), rep("Test",nrow(orig.test))),
  target = factor(c(orig.train$target, orig.test$target), 
                  levels=1:nlevels(orig.train$target), 
                  labels=levels(orig.train$target)))
data.norm <- cbind(target=targets[,2], data.norm)  # 206246 x 94

######################
## data exploration ##
######################

# correlation plots
i.beg <- 1     # set beginning feature to include
i.end <- 93    # set ending feature to include 
M <- cor(log1p(data.norm[,(i.beg+1):(i.end+1)]))
dimnames(M)[[1]] <- paste("f.",i.beg:i.end,sep="")
dimnames(M)[[2]] <- dimnames(M)[[1]]

#------------------------------------------------------
# outputs correlation .png plots
res1 <- cor.mtest(log1p(data.norm[,(i.beg+1):(i.end+1)]), 0.95)
res2 <- cor.mtest(log1p(data.norm[,(i.beg+1):(i.end+1)]), 0.99)
png(paste("corrplot_f",i.beg,"_f",i.end,".png",sep=""), width=3500, height=3500)
corrplot.mixed(abs(M), p.mat = res1[[1]], sig.level = 0.1, cl.lim=c(0,1))  # used this for each 25 features
corrplot(M, p.mat = res1[[1]], order="hclust", addrect= 25, insig = "pch") # used this for 1-93 plot
dev.off()
#------------------------------------------------------

# filter out vars w > 0.7 correlation
data.mdl <- log1p(data.norm[,2:94])  # drops target col for now: 61878 93
highlyCor <- findCorrelation(M, 0.7)  # part of the caret package
data.mdl <- data.mdl[,-highlyCor]  # remove cols 54  3 15  9 39
data.mdl <- cbind(target=data.norm$target, data.mdl)  # add back target col as first col
# M.filtered <- cor(data.mdl)
# corrplot(M.filtered, order="hclust")
# (order.hc <- corrMatOrder(M.filtered, order="hclust"))  # shows order of hclusts
# M.hc <- M.filtered[order.hc,order.hc]

# # pca test with FactoMineR package
# m.pca <- PCA(data.mdl, scale.unit=TRUE, ncp=5, graph=TRUE)
# summary(m.pca)
# plot(m.pca, choix="var", shadow=TRUE, select="contrib 10", cex=0.7)  # top 10 vars, cex sets font size
# dimdesc(m.pca)

# pca test with built-in stats data (https://www.youtube.com/watch?t=411&v=qhvkVxuwvLk)
nzv <- nearZeroVar(data.mdl[-1], saveMetrics=TRUE)
print(paste("Range:",range(nzv$percentUnique)))  # all feats have var > 0.01
# dim(nzv[nzv$percentUnique>0.05,])  # 57 feats have var > 0.05
data.mdl_nzv <- data.mdl[c(rownames(nzv[nzv$percentUnique>0.05,])) ]

# evaluate full model with some classifier algorithm
# Evaluate_GBM_AUC(data.mdl, CV=5, trees=10, depth=2, shrink=1) 

princ <- prcomp(data.mdl[-1])

nComp <- 10
dfComponents <- predict(princ, newdata=data.mdl[-1])[,1:nComp]
dfEvaluatePCA <- cbind(target=data.mdl$target, as.data.frame(dfComponents))

# evaluate with classifier again
# Evaluate_GBM_AUC(dfEvaluatePCA, CV=5, trees=10, depth=2, shrink=1) 



#------------------------------------------------------
dfmelt <- melt(data.norm, id="target", variable_name="feature")
dfmelt$valuelog <- log(dfmelt$value + 1)  # transforming bc many 0s

# prints boxplots by class for each feature
plots <- list()
pdf(file="feat_boxplotsLOG.pdf")
for (i in 1:93) {
  plots[[i]] <- ggplot(
    dfmelt[dfmelt$feature %in% paste("feat_",i,sep=""),], 
    aes(x=target, y=valuelog, fill=target)) +
    ylab(paste("Feature #",i,"  (scaled)",sep="")) + xlab("") + 
    geom_boxplot()
}
bquiet = lapply(plots, print)
dev.off()  # exports to PDF, 1 plot/page


ggplot(data.norm, aes(x=feat_2, colour=target, fill=target)) +
  geom_density(alpha=0.01) +
  scale_x_continuous(limits=c(0, .03)) +
  scale_y_sqrt()

# ggvis takes a while
# data.norm %>% ggvis(~feat_1, ~feat_2, fill = ~target) %>% layer_points()



######################
## helper functions ##
######################

normalize <- function(x) {
  num <- x - min(x)
  denom <- max(x) - min(x)
  return (num/denom)
}

cor.mtest <- function(mat, conf.level = 0.95) {
  mat <- as.matrix(mat)
  n <- ncol(mat)
  p.mat <- lowCI.mat <- uppCI.mat <- matrix(NA, n, n)
  diag(p.mat) <- 0
  diag(lowCI.mat) <- diag(uppCI.mat) <- 1
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      tmp <- cor.test(mat[, i], mat[, j], conf.level = conf.level)
      p.mat[i, j] <- p.mat[j, i] <- tmp$p.value
      lowCI.mat[i, j] <- lowCI.mat[j, i] <- tmp$conf.int[1]
      uppCI.mat[i, j] <- uppCI.mat[j, i] <- tmp$conf.int[2]
    }
  }
  return(list(p.mat, lowCI.mat, uppCI.mat))
}

Evaluate_GBM_AUC <- function(dfEvaluate, CV=5, trees=3, depth=2, shrink=0.1) {  # NEED TO FIX THIS!!! -- return here
  CVs <- CV
  cvDivider <- floor(nrow(dfEvaluate) / (CVs+1))
  indexCount <- 1
  outcomeName <- c('target')
  predictors <- names(dfEvaluate)[!names(dfEvaluate) %in% outcomeName]
  lsErr <- c()
  lsAUC <- c()
  for (cv in seq(1:CVs)) {
    print(paste('cv',cv))
    
    dataTestIndex <- c((cv * cvDivider):(cv * cvDivider + cvDivider))
    dataTest <- dfEvaluate[dataTestIndex,]
    dataTrain <- dfEvaluate[-dataTestIndex,]
        
    dataTrain[,outcomeName] <- ifelse(dataTrain[,outcomeName]==1,'yes','nope')
    
    # create caret trainControl object to control the number of cross-validations performed
    objControl <- trainControl(method='cv', number=2, returnResamp='none', classProbs = TRUE)
    
    # run model
    bst <- train(dataTrain[,predictors], dataTrain[,outcomeName], 
                 method='gbm', 
                 trControl=objControl,
                 metric = "Accuracy",
                 #tuneGrid = expand.grid(n.trees = trees, interaction.depth = depth, shrinkage = shrink)
    )
    
    predictions <- predict(object=bst, dataTest[,predictors], type='prob')
    auc <- auc(ifelse(dataTest[,outcomeName]==1,1,0),predictions[[2]])
    err <- rmse(ifelse(dataTest[,outcomeName]==1,1,0),predictions[[2]])
    
    lsErr <- c(lsErr, err)
    lsAUC <- c(lsAUC, auc)
    gc()
  }
  print(paste('Mean Error:',mean(lsErr)))
  print(paste('Mean AUC:',mean(lsAUC)))
}


#------------------------------------------------------
# knn model -- takes a while to run
pred.knn <- knn(train=data.train[-1], test=data.test[-1], cl=data.train$target, k=3, prob=TRUE)
CrossTable(x = data.test$target, y = pred.knn, prop.chisq=FALSE)
