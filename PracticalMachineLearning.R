# 1. Your submission should consist of a link to a Github repo with your R markdown
# and compiled HTML file describing your analysis. Please constrain the text of the
# writeup to < 2000 words and the number of figures to be less than 5. It will make 
# it easier for the graders if you submit a repo with a gh-pages branch so the HTML
# page can be viewed online (and you always want to make it easy on graders :-).

# 2. You should also apply your machine learning algorithm to the 20 test cases 
# available in the test data above. Please submit your predictions in appropriate
# format to the programming assignment for automated grading. See the programming 
# assignment for additional details. 

# We propose a dataset with 5 classes (sitting-down, standing-up, standing, 
# walking, and sitting) collected on 8 hours of activities of 4 healthy subjects.
# walking, and sitting) collected on 8 hours of activities of 4 healthy subjects.
# walking, and sitting) collected on 8 hours of activities of 4 healthy subjects.

install.packages("caret")
install.packages('e1071', dependencies=TRUE)
install.packages('doMC')
#--------------------------------------------------------
download.file(destfile="training.csv", url='https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv')
download.file(destfile = 'test.csv', url='https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv')
#--------------------------------------------------------

library(caret)
library(e1071)

#Enable parallel computation for 4 cores
library(doMC)
registerDoMC(cores = 6)

#This might make the results reproducible
set.seed(1)

#Load data into workspace
train = read.csv('training.csv', stringsAsFactors=FALSE)
test = read.csv('test.csv', stringsAsFactors=FALSE)

#Remove observation ID, subject name, and timestamp/window information
#With only 20 items in the test set, it doesn't make much sense to treat them as time series data.
train = train[,-(1:8)]
#For test set, also remove problem ID
test = test[,-c(1:8, 160)]

#Data frame still has a lot of garbage in the form:
#$ skewness_yaw_belt       : chr  "" "" "" "" ...
#$ max_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
#Remove any atribute with more than 50% NA or empty strings
getGetFeatureAttributes = function(df) { 
  cutoff = 0.5 * nrow(df)
  good_cols1 = colSums(is.na(df)) < cutoff 
  #Define helper function. Return true if input is an empty string
  equalsEmptyString = function(x) x==""
  good_cols2 = colSums(equalsEmptyString(df)) < cutoff
  good_cols_all = good_cols1 & good_cols2
  return(good_cols_all)
}

test = test[,getGetFeatureAttributes(test)]
train = train[, getGetFeatureAttributes(train)]

#The only non-numeric attribute left is the classification or "classe"
#Caret gets pissy if the classes are characters and not factors.
train$classe = as.factor(train$classe)

#Use cross fold validation with 7 folds.
trainCtrl = trainControl(method='cv', number=7)

#Define helper function to test different machine learning algorithms
validation = sample(nrow(train), 0.2 * nrow(train))
holdout = train[validation,]
train=train[-validation,]

learn = function(x) train(classe ~ ., data=train, method=x, trControl=trainCtrl, preProcess=c("scale", "center"))

#Create some models
gbm_model = learn('gbm')
rf_model = learn('rf')

#Look at accuracy and kappa
gbm_model$results
rf_model$results

predictions = predict(rf_model, holdout[,-52])
confusionMatrix(predictions, holdout[,52])

#Look at confusion matrices
confusionMatrix(gbm_model)
confusionMatrix(rf_model)

#Make predictions
answers <- predict(rf_model, test)

# Class A corresponds to the specified execution of the exercise, while the other
# 4 classes correspond to common mistakes.

# In this project, your goal will be to use data from accelerometers on the belt, 
# forearm, arm, and dumbell of 6 participants. They were asked to perform barbell 
# lifts correctly and incorrectly in 5 different ways. 

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(as.character(answers))