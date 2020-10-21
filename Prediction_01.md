## Introduction  
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.  

In this project, we will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did the exercise.  

## Data Preprocessing  
```{r, cache = T}
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(corrplot)
```
### Download the Data
First we download the data
```{r, cache = T}
trainUrl <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
trainPml <- "./database/pml-training.csv"
testPml  <- "./database/pml-testing.csv"
```  
### Read the Data
Now we can read the two csv files into two data frames.  
```{r, cache = T}
train_Raw_Data <- read.csv("./database/pml-training.csv")
test_Raw_Data <- read.csv("./database/pml-testing.csv")
dim(train_Raw_Data)
dim(test_Raw_Data)
```
The training data set contains 19622 observations and 160 variables, while the testing data set contains 20 observations and 160 variables. 
The "classe" variable in the training set is the outcome to predict. 

### Clean the data
In this step, we clean the data and get rid of meaningless observations.
```{r, cache = T}
sum(complete.cases(train_Raw_Data))
```
First, we remove columns that contain NA missing values.
```{r, cache = T}
train_Raw_Data <- train_Raw_Data[, colSums(is.na(train_Raw_Data)) == 0] 
test_Raw_Data <- test_Raw_Data[, colSums(is.na(test_Raw_Data)) == 0] 
```  
Next, we get rid of some columns that do not contribute much 
```{r, cache = T}
classe <- train_Raw_Data$classe
trainRemove <- grepl("^X|timestamp|window", names(train_Raw_Data))
train_Raw_Data <- train_Raw_Data[, !trainRemove]
trainCleaned <- train_Raw_Data[, sapply(train_Raw_Data, is.numeric)]
trainCleaned$classe <- classe
testRemove <- grepl("^X|timestamp|window", names(test_Raw_Data))
test_Raw_Data <- test_Raw_Data[, !testRemove]
testCleaned <- test_Raw_Data[, sapply(test_Raw_Data, is.numeric)]
```
Now, the cleaned training data set contains 19622 observations and 53 variables, while the testing data set contains 20 observations and 53 variables. 
The "classe" variable is still in the cleaned training set.

### Split the data
Then, we can split the cleaned training set into a pure training a validation data set. We will use the validation data set to conduct cross validation in future steps.  
```{r, cache = T}
set.seed(22519) 
inTrain <- createDataPartition(trainCleaned$classe, p=0.70, list=F)
trainData <- trainCleaned[inTrain, ]
testData <- trainCleaned[-inTrain, ]
```

## Data Modeling
We fit a predictive model for activity recognition using **Random Forest** algorithm because it automatically selects important variables and is robust to correlated covariates & outliers in general. We will use **5-fold cross validation** when applying the algorithm.  
```{r, cache = T}
ctrlRf <- trainControl(method="cv", 5)
modelRf <- train(classe ~ ., data=trainData, method="rf", trControl=ctrlRf, ntree=250)
modelRf
```
Then, we estimate the performance of the model on the validation data set.  
```{r, cache = T}
prdtRf <- predict(modelRf, testData)
confusionMatrix(testData$classe, prdtRf)
```
```{r, cache = T}
accuracy <- postResample(prdtRf, testData$classe)
accuracy
oose <- 1 - as.numeric(confusionMatrix(testData$classe, prdtRf)$overall[1])
oose
```
So, the estimated accuracy of the model is 99.42% and the estimated out-of-sample error is 0.58%.

## Predicting Test Data
Now, we apply the model to the original testing data set downloaded from the data source. We remove the `problem_id` column first.  
```{r, cache = T}
result <- predict(modelRf, testCleaned[, -length(names(testCleaned))])
result
```  

## Figures
1. Correlation Matrix  
```{r, cache = T}
corrPlot <- cor(trainData[, -length(names(trainData))])
corrplot(corrPlot, method="color")
```
2. Decision Tree 
```{r, cache = T}
treeModel <- rpart(classe ~ ., data=trainData, method="class")
prp(treeModel) 
```