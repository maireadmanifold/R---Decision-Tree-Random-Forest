##Decision Tree & Random Forest Kaggle Banking Marketing Dataset  


install.packages("dplyr")
install.packages("ggplot2")
install.packages("ggthemes")
install.packages("mice")
install.packages("randomForest")
install.packages("scales")
install.packages("tidyverse")
install.packages("tidyr")

library(tidyverse)
library(dplyr)
library(ggplot2)
library(ggthemes)
library(mice)
library(randomForest)
library(scales)

## dataset chosen from Kaggle is a marketing banking dataset. 
##https://www.kaggle.com/volodymyrgavrysh/bank-marketing-campaigns-dataset

banking = read.table("bank-additional-full.csv", header=TRUE, sep=";", na.strings=c("NA", "unknown"))



##*2 - job : type of job (categorical: "admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown")
##*3 - marital : marital status (categorical: "divorced","married","single","unknown"; note: "divorced" means divorced or widowed)
##*4 - education (categorical: "basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown")
##*5 - default: has credit in default? (categorical: "no","yes","unknown")
##*6 - housing: has housing loan? (categorical: "no","yes","unknown"
##*7 - loan: has personal loan? (categorical: "no","yes","unknown")
##*8 - contact: contact communication type (categorical: "cellular","telephone")
##*9 - month: last contact month of year (categorical: "jan", "feb", "mar", ., "nov", "dec")
##*10 - dayofweek: last contact day of the week (categorical: "mon","tue","wed","thu","fri")
##*11 - duration: last contact duration, in seconds (numeric). Important note: this attribute 
##*##   highly affects the output target (e.g., if duration=0 then y="no"). 
##*##   Yet, the duration is not known before a call is performed. Also, after the end of the call 
##*##   y is obviously known. 
##*##   Thus, this input should only be included for benchmark purposes and should be discarded 
##*##   if the intention is to have a realistic predictive model.
##*12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)

##*13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)

##*14 - previous: number of contacts performed before this campaign and for this client (numeric)

##15 - outcome: outcome of the previous marketing campaign (categorical: "failure","nonexistent","success")

##social and economic context attributes
##*16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)

##*17 - cons.price.idx: consumer price index - monthly indicator (numeric)

##*18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)

##*19 - euribor3m: euribor 3 month rate - daily indicator (numeric)

##20 - nr.employed: number of employees - quarterly indicator (numeric)

## Output variable (desired target):
  
## 21 - y - has the client subscribed a term deposit? (binary: "yes","no")

## check if any NAs
sapply(banking, function(x) sum(is.na(x)))

## checking output in Excel: write.csv(banking,"bankingCSV.csv", quote = FALSE, row.names = FALSE)

## remove rows with housing NAs
banking <- banking %>% filter(!is.na(housing)) 
## remove rows with default NAs
banking <- banking %>% filter(!is.na(default)) 

## see missing data in heat map and in table
md.pattern(banking) 

## see missing data pairs
md.pairs(banking)

## check if any NAs
sapply(banking, function(x) sum(is.na(x)))

## remove rows with job NAs
banking <- banking %>% filter(!is.na(job)) 
## remove rows with marital NAs
banking <- banking %>% filter(!is.na(marital)) 
## remove rows with education NAs
banking <- banking %>% filter(!is.na(education)) 

## convert euribor3m to 1 decimal place as too many levels for Random Forest
banking$euribor3m <- round(banking$euribor3m, digits = 1)

## check if any NAs
sapply(banking, function(x) sum(is.na(x)))



banking$job <- as.factor(banking$job)
banking$marital <- as.factor(banking$marital)
banking$education <- as.factor(banking$education)
banking$default <- as.factor(banking$default)
banking$housing <- as.factor(banking$housing)
banking$loan <- as.factor(banking$loan)
banking$contact <- as.factor(banking$contact)
banking$month <- as.factor(banking$month)
banking$day_of_week <- as.factor(banking$day_of_week)
banking$campaign <- as.factor(banking$campaign)
banking$pdays <- as.factor(banking$pdays)
banking$previous <- as.factor(banking$previous)
banking$poutcome <- as.factor(banking$poutcome)
banking$emp.var.rate <- as.factor(banking$emp.var.rate)
banking$cons.conf.idx <- as.factor(banking$cons.conf.idx)
banking$cons.price.idx <- as.factor(banking$cons.price.idx)
banking$euribor3m <- as.factor(banking$euribor3m)
banking$nr.employed <- as.factor(banking$pdays)
banking$y <- as.factor(banking$y)


attach(banking)


hist(as.numeric(y))
banking[which(banking$y == "no"),] ##only 47 rows shown rest omitted
countNRowsNo <- length(which(banking$y == "no")) ## 26,629
countNRowsYes <- length(which(banking$y == "yes")) ##3,859

###### create test and train sets

## randomly assign rows between train and test
seed = 123
s <- sample(nrow(banking), (nrow(banking)*.8))

train_banking <- banking[s, ]
test_banking <- banking[-s, ]

dim(test_banking) #6,098
dim(train_banking) #24,390


write.csv(banking, "bankingBeforeSplit.csv")

# decision tree
#install.packages("rpart")
library(rpart)
#install.packages("rpart.plot")
library(rpart.plot)

fit <- rpart(y~., data = train_banking, method = 'class', control = rpart.control(minsplit = 20, minbucket = 7, maxdepth = 10, usesurrogate = 2, xval =10 ))
rpart.plot(fit, type = 2, tweak = 1.1, extra = 101, box.palette=c("red", "green") )

predicted = predict(fit, test_banking, type = 'class')

### Confusion test Decision Tree *********************

table <- table(test_banking$y, predicted)

decision_tree_accuracy <- sum(diag(table)) / sum(table)

paste("The accuracy is ", decision_tree_accuracy)

#install.packages("caret")

library(caret)

confusionMatrix(table <- table(test_banking$y, predicted), positive = "yes")

###########################################

install.packages("rattle")
install.packages("RcolorBrewer")
#Beautify tree
library(rattle)
library(RColorBrewer)

#view1
prp(fit, faclen = 0, cex = 0.8, extra = 1)

## ------------------------------------------------------------------------
#view2 - total count at each node
tot_count <- function(x, labs, digits, varlen)
{paste(labs, "\n\nn =", x$frame$n)}

prp(fit, faclen = 0, cex = 0.8, node.fun=tot_count)

## ------------------------------------------------------------------------
#view3- fancy Plot
library(rattle)
#library(gKt)
#rattle()
fancyRpartPlot(fit)

## ------------------------------------------------------------------------
printcp(fit)
bestcp <- fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"]

# Prune the tree using the best cp.
pruned <- prune(fit, cp = bestcp)


## ------------------------------------------------------------------------
# Plot pruned tree
prp(pruned, faclen = 0, cex = 0.8, extra = 1)

## ------------------------------------------------------------------------
# confusion matrix (training data) pruned
conf.matrix_pruned <- table(train_banking$y, predict(pruned,type="class"))
rownames(conf.matrix_pruned) <- paste("Actual", rownames(conf.matrix_pruned), sep = ":")
colnames(conf.matrix_pruned) <- paste("Pred", colnames(conf.matrix_pruned), sep = ":")
print(conf.matrix_pruned)



decision_tree_accuracy_pruned <- sum(diag(conf.matrix_pruned)) / sum(conf.matrix_pruned)

paste("The accuracy of pruned is ", decision_tree_accuracy_pruned)



## ------------------------------------------------------------------------
#Scoring
library(ROCR)
val1 = predict(pruned, test_banking, type = "prob")
#Storing Model Performance Scores
pred_val <-prediction(val1[,2],test_banking$y)

# Calculating Area under Curve
perf_val <- performance(pred_val,"auc")
perf_val

par(mfrow=c(2,2))
# Plotting Lift curve
plot(performance(pred_val, measure="lift", x.measure="rpp"), colorize=TRUE)

# Calculating True Positive and False Positive Rate
perf_val.No <- performance(pred_val, "tpr", "fpr")
# Calculating True Negative and False Negative Rate
perf_val.Yes <- performance(pred_val, "tnr", "fnr")
par(mfrow=c(2,2))
# Plot the ROC curve
plot(perf_val.Yes, col = "green", lwd = 1.5)

# Plot the ROC curve
plot(perf_val.No, col = "red", lwd = 1.5)

#Calculating KS statistics
ks1.tree <- max(attr(perf_val.Yes, "y.values")[[1]] - (attr(perf_val.Yes, "x.values")[[1]]))
ks1.tree
ks1.treeNo <- max(attr(perf_val.No, "y.values")[[1]] - (attr(perf_val.No, "x.values")[[1]]))
ks1.treeNo

## ------------------------------------------------------------------------
# Advanced Plot
prp(pruned, main="Beautiful Tree",
    extra=106, 
    nn=TRUE, 
    fallen.leaves=TRUE, 
    branch=.5, 
    faclen=0, 
    trace=1, 
    shadow.col="gray", 
    branch.lty=3, 
    split.cex=1.2, 
    split.prefix="is ", 
    split.suffix="?", 
    split.box.col="lightgray", 
    split.border.col="darkgray", 
    split.round=.5)


#####################################################
### --
###-------------------Random Forest--------------------
##
#####################################################

library(randomForest)
set.seed(1)
fit.rf <- randomForest(y~., data=banking)
## examine the results
fit.rf
## variable importance
varImpPlot(fit.rf)

### PAUSE HERE AS random forest takes a few mins to complete ~~~~

prediction.rf <- predict(fit.rf, newdata = test_banking)

confusionMatrix(table <- table(test_banking$y, prediction.rf))

prediction.rf.train <- predict(fit.rf, newdata = train_banking)

confusionMatrix(table <- table(train_banking$y, prediction.rf.train), positive = "yes")


