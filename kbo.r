
# Listing 17.7 은 가급적 돌리지 말것. 많은 시간이 소요되며 잘 반영되지 않음.
# 본 코드를 수행하기 위해서는 
# 원본 데이터를 아래와 같이 변형해야 함 head(kbo)참조. 

# 원 데이터는 두팀이 한row에 있으며 모든 자료가 같은 row를 차지하고 있음.
# 팀명으로 홈 어웨이를 구분(hOrA)하고 같은 row의 자료를 아래로 Rbind하여 위 데이터셋을 만들어야 함. 
# 마지막에 있는 코드는 각 분류방법에 대한 정확도와 민감도를 나타냄. 가장 좋은 것은 역시 랜덤포레스트.


# install.packages(c("rpart", "party", "randomForest", "e1071", "rpart.plot", "readr"))
# install.packages("partykit")
library(rpart)
library(party)
library(randomForest)
library(e1071)
library(rpart.plot)
library(readr)
library(readxl)

kbo <- read_csv("kbolist_re.csv")
head(kbo)

df <- kbo[-1]
df$winOrloss <- factor(df$winOrloss, levels=c(1,0), 
                   labels=c("win", "loss"))

set.seed(1234)
train <- sample(nrow(df), 0.7*nrow(df))
df.train <- df[train,]
df.validate <- df[-train,]
table(df.train$winOrloss)
# win loss 
# 2443 2422 
table(df.validate$winOrloss)
# win loss 
# 1032 1053 

# Listing 17.2 - Logistic regression with glm()
fit.logit <- glm(winOrloss~., data=df.train, family=binomial())
summary(fit.logit)
##############################################################################################
# Call:
#   glm(formula = winOrloss ~ ., family = binomial(), data = df.train)
# 
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -2.0658  -1.0863  -0.5907   1.0797   2.0736  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept)  0.307633   0.160909   1.912 0.055897 .  
# hOrA        -0.414391   0.060146  -6.890 5.59e-12 ***
#   inning      -0.232867   0.098284  -2.369 0.017821 *  
#   tajasu       0.214432   0.054491   3.935 8.31e-05 ***
#   tugusu      -0.000244   0.003561  -0.069 0.945372    
# tasu        -0.160913   0.047018  -3.422 0.000621 ***
#   anta         0.013630   0.033197   0.411 0.681376    
# fourball    -0.181790   0.057509  -3.161 0.001572 ** 
#   homerun      0.083115   0.044310   1.876 0.060690 .  
# samjin      -0.009144   0.016582  -0.551 0.581337    
# siljum       0.130653   0.049832   2.622 0.008744 ** 
#   jacheck     -0.041126   0.048831  -0.842 0.399676    
# score       -0.096739   0.008490 -11.395  < 2e-16 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 6744.2  on 4864  degrees of freedom
# Residual deviance: 6326.0  on 4852  degrees of freedom
# AIC: 6352
# 
# Number of Fisher Scoring iterations: 4
##############################################################################################
prob <- predict(fit.logit, df.validate, type="response")
logit.pred <- factor(prob > .5, levels=c(FALSE, TRUE), 
                     labels=c("win", "loss"))
logit.perf <- table(df.validate$winOrloss, logit.pred,
                    dnn=c("Actual", "Predicted"))
logit.perf
#prob
#     Predicted
# Actual win loss
#  win  686  346
#  loss 359  694
##############################################################################################

# Listing 17.3 - Creating a classical decision tree with rpart()
library(rpart)
set.seed(1234)
dtree <- rpart(winOrloss ~ ., data=df.train, method="class",      
               parms=list(split="information"))
dtree$cptable
plotcp(dtree)

dtree.pruned <- prune(dtree, cp=.0125) 

library(rpart.plot)
#?prp()
prp(dtree.pruned, type = 2, extra = 104,  
    fallen.leaves = TRUE, main="Decision Tree")

dtree.pred <- predict(dtree.pruned, df.validate, type="class")
dtree.perf <- table(df.validate$winOrloss, dtree.pred, 
                    dnn=c("Actual", "Predicted"))
dtree.perf


# Listing 17.4 - Creating a conditional inference tree with ctree()
library(party)
fit.ctree <- ctree(winOrloss~., data=df.train)
plot(fit.ctree, main="Conditional Inference Tree")

ctree.pred <- predict(fit.ctree, df.validate, type="response")
ctree.perf <- table(df.validate$winOrloss, ctree.pred, 
                    dnn=c("Actual", "Predicted"))
ctree.perf


library(partykit)
plot(as.party(dtree.pruned))

# Listing 17.5 - Random forest
library(randomForest)
set.seed(1234)
fit.forest <- randomForest(winOrloss~., data=df.train,        
                           na.action=na.roughfix,
                           importance=TRUE)             
fit.forest
importance(fit.forest, type=2)                          
forest.pred <- predict(fit.forest, df.validate)         
forest.perf <- table(df.validate$winOrloss, forest.pred, 
                     dnn=c("Actual", "Predicted"))
forest.perf

varImpPlot(fit.forest)



# Listing 17.6 - A support vector machine
library(e1071)
set.seed(1234)
fit.svm <- svm(winOrloss~., data=df.train)
fit.svm
svm.pred <- predict(fit.svm, na.omit(df.validate))
svm.perf <- table(na.omit(df.validate)$winOrloss, 
                  svm.pred, dnn=c("Actual", "Predicted"))
svm.perf


# Listing 17.7 Tuning an RBF support vector machine (this can take a while)
# many time spend..... if you have a dull, please cancel this process.
#set.seed(1234)
#tuned <- tune.svm(winOrloss~., data=df.train,
#                  gamma=10^(-6:1),
#                  cost=10^(-10:10))
#tuned
#fit.svm <- svm(winOrloss~., data=df.train, gamma=.01, cost=1)
#svm.pred <- predict(fit.svm, na.omit(df.validate))
#svm.perf <- table(na.omit(df.validate)$winOrloss,
#                  svm.pred, dnn=c("Actual", "Predicted"))
#svm.perf


# Listing 17.8 Function for assessing binary classification accuracy
performance <- function(table, n=2){
  if(!all(dim(table) == c(2,2)))
    stop("Must be a 2 x 2 table")
  tn = table[1,1]
  fp = table[1,2]
  fn = table[2,1]
  tp = table[2,2]
  sensitivity = tp/(tp+fn)
  specificity = tn/(tn+fp)
  ppp = tp/(tp+fp)
  npp = tn/(tn+fn)
  hitrate = (tp+tn)/(tp+tn+fp+fn)
  result <- paste("Sensitivity = ", round(sensitivity, n) ,
                  "\nSpecificity = ", round(specificity, n),
                  "\nPositive Predictive Value = ", round(ppp, n),
                  "\nNegative Predictive Value = ", round(npp, n),
                  "\nAccuracy = ", round(hitrate, n), "\n", sep="")
  cat(result)
}


# Listing 17.9 - Performance of breast cancer data classifiers
performance(dtree.perf)
performance(ctree.perf)
performance(forest.perf)
performance(svm.perf)


