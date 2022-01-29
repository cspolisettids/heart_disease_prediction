# Title: "Heart Disease Prediction System"
# Author: "Chandra Sekhar Polisetti"
# Program Outline ####
# In this Program the primary focus would be on predicting the heart disease using various machine learning algorithms
# ,present their results and recommend algorithms based sensitivity and specificity expectations . 
# The R Code snippets used in the .RMD file for analysis is not include here as it is not required for the algorithms
# and as it was only used for gaining the understanding.

# The following is the high level outline of this program, each step would be detailed out in the subsequent sections.

# 1) Package Installation & Loading
# 2) Data Download & Cleanup
# 3) Data Partition for Training & Testing
# 4) Algorithm evaluation criteria
# 5) Model 1 - Novice Heart Disease Model
# 6) Model 2 - Logistic Regression Heart Disease Model
# 7) Model 3 - KNN Heart Disease Model
# 8) Model 4 - Classification Tree Heart Disease Model
# 9) Model 5 - Random Forest Heart Disease Model
# 10) Model 6 - Ensemble Model
# 11) Results Comparison & Model Recommendation
# 12) Conclusion

# 1) Package Installation & Loading ####
# In this section all the required packages will be installed if present in your system and 
# then load the packages.

## Install Required packages ####

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(ggthemes)) install.packages("ggthemes", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")

## Load required package libraries ####

library(tidyverse)
library(ggplot2)
library(ggthemes)
library(caret)
library(knitr)
library(kableExtra)
library(rpart)
library(randomForest)
rafalib::mypar()

# Ingore Warning During the program
oldw <- getOption("warn")
options(warn = -1)
# 
# #[your "silenced" code]
# 
# options(warn = oldw)

# 2) Data Download & Cleanup ####

## Download data ####

# Heart disease dataset is downloaded form the UCI machine learning repository and here is the path to the repository, 
# http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data.

# Download 
url <- "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
heart_disease_ds <- read.csv(url, header=FALSE)

## Data Cleanup ####

#Lets look at the dimensions of the dataset
#As we will see there are 14 dimensions and the total number of rows are 304
dim(heart_disease_ds)

#Let's look at the first 6 rows of the data.

head(heart_disease_ds)

#As we can see that there are no column names so let's fix the column names and re look at the data.
### Add Column Names ####

colnames(heart_disease_ds) <- c(
  "age",
  "sex",# 0 = female, 1 = male
  "cp", # chest pain
  # 1 = typical angina,
  # 2 = atypical angina,
  # 3 = non-anginal pain,
  # 4 = asymptomatic
  "restbps", # resting blood pressure (in mm Hg)
  "chol", # serum cholestoral in mg/dl
  "fbs",  # fasting blood sugar if less than 120 mg/dl, 1 = TRUE, 0 = FALSE
  "restecg", # resting electrocardiographic results
  # 1 = normal
  # 2 = having ST-T wave abnormality
  # 3 = showing probable or definite left ventricular hypertrophy
  "thalach", # maximum heart rate achieved
  "exang",   # exercise induced angina, 1 = yes, 0 = no
  "oldpeak", # ST depression induced by exercise relative to rest
  "slope", # the slope of the peak exercise ST segment
  # 1 = upsloping
  # 2 = flat
  # 3 = downsloping
  "ca", # number of major vessels (0-3) colored by fluoroscopy
  "thal", # this is short of thalium heart scan
  # 3 = normal (no cold spots)
  # 6 = fixed defect (cold spots during rest and exercise)
  # 7 = reversible defect (when cold spots only appear during exercise)
  "hd" # (the predicted attribute) - diagnosis of heart disease
  # 0 if less than or equal to 50% diameter narrowing
  # 1 if greater than 50% diameter narrowing
)

head(heart_disease_ds)

#Lets look at the structure of the dataset

str(heart_disease_ds)

# Based on the above data we need to make the below changes to the data structure
# 
# * Sex column has values 0 and 1 to represent female and male patients, 
#   for better readability lets mark them as $F$ for female and $M$ for male values.
# * There are missing values in ca and thal, we need to convert them into NA's first and 
#   later we will discuss on what to do with the values.
#   * cp, fbs, restecg , exang , slop needs to be converted into factors
#   * hd values has 0 represent healthy heart with no heart disease , and 1 ,2 and 3 values 
#     represent unhealthy heart. As our main goal is to predict the presence of heart disease, 
#     hd 0 values would be converted into as Healthy and hd values $1,2 and 3 as Unhealthy.
#   

#   Here is the code to make all the above changes
### Change Structure of the dataset ####

# Mark male and female as M and F for better readability
heart_disease_ds[heart_disease_ds$sex == 0,]$sex <- "F"
heart_disease_ds[heart_disease_ds$sex == 1,]$sex <- "M"

# replace missing values with NA
heart_disease_ds[heart_disease_ds == "?"] <- NA

# Convert the columns which have factors to factor

heart_disease_ds$sex <- as.factor(heart_disease_ds$sex)
heart_disease_ds$cp <- as.factor(heart_disease_ds$cp)
heart_disease_ds$fbs <- as.factor(heart_disease_ds$fbs)
heart_disease_ds$restecg <- as.factor(heart_disease_ds$restecg)
heart_disease_ds$exang <- as.factor(heart_disease_ds$exang)
heart_disease_ds$slope <- as.factor(heart_disease_ds$slope)

# Since ca and thal have ?, R converts the values as level's of string values, 
# but as they are integer values lets convert them to integers and then convert to factor.

heart_disease_ds$ca <- as.integer(heart_disease_ds$ca) 
heart_disease_ds$ca <- as.factor(heart_disease_ds$ca)
heart_disease_ds$thal <- as.integer(heart_disease_ds$thal) # "thal" also had "?"s in it.
heart_disease_ds$thal <- as.factor(heart_disease_ds$thal)

## This next line replaces 0 and 1 with "Healthy" and "Unhealthy"
heart_disease_ds$hd <- ifelse(test=heart_disease_ds$hd == 0, yes="Healthy", no="Unhealthy")
heart_disease_ds$hd <- as.factor(heart_disease_ds$hd) 

# Now after all the above changes lets look at the data
str(heart_disease_ds)

### Handle Missing Values ####
#Now that the data structure has been cleaned up lets work on the missing values.

# Calculate Missing Value numbers
rows_with_missing_values <- sum(is.na(heart_disease_ds))
total_rows <- nrow(heart_disease_ds)
missing_value_row_percent <- (rows_with_missing_values/total_rows) * 100

# Print Missing Value numbers
rows_with_missing_values
total_rows
missing_value_row_percent

# Remove Missing Values as the percentage of missing values are pretty low

heart_disease_ds <- heart_disease_ds[!(is.na(heart_disease_ds$ca) | is.na(heart_disease_ds$thal)),]

# Here is the dimensions of the data after removing missing values

dim(heart_disease_ds)

# As a last step of cleaning lets re order the levels of hd to "Unhealthy", "Healthy" to facilitate confusion matrix results

heart_disease_ds$hd <- heart_disease_ds$hd %>% factor(levels = c("Unhealthy","Healthy"))

# 3) Data Partition for Training & Testing ####

# Before we start building the algorithms we need to split the data for training and testing the algorithm.
# 
# Training data will only be used for training and optimizing the algorithm ,
# and the testing data will be exclusively used for testing the optimized algorithm.
# 
# Following are the steps that are performed for data partitioning 
# 
# 1) 20% of the data rows from the heart disease dataset (heart_disease_ds) are randomly selected
#    and placed in test_set  dataset, and this will be kept aside for performing the validation of each Heart Disease Prediction Model we build in the subsequent sections. This dataset will not be used for training and optimizing the model. This dataset is not used for training mainly to avoid over fitting the data.
# 2) The remaining 80% of the data rows from the heart disease dataset are brought into the 
#    train_set dataset.This dataset is mainly used for training and optimizing each of the algorithms we build.

# Here is to code to partition the dataset

set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = heart_disease_ds$hd, times = 1, p = 0.2, list = FALSE)
train_set <- heart_disease_ds[-test_index,]
test_set <- heart_disease_ds[test_index,]

# Here are the dimensions of the train and test datasets

dim(train_set)
dim(test_set)

# 4) Algorithm evaluation criteria ####

# As we are dealing with a classification problem we will be using sensitivity , specificity and accuracy to 
# evaluate the performance of the algorithms.
# 
# The high level process we follow for Training & Optimization, and Validation is listed below, and note that the below process
# would be followed in all the Algorithms that we build in the subsequent sections and hence I am giving an overview 
# of the process before starting the individual algorithms.
# 
# * We use cross validation with 10 folds on train_set,to train and optimize the algorithm.
# * We take the parameters that optimized the algorithm from the above step and use them 
#   to re-train the algorithm using the entire train_set. Retraining the algorithm with optimized parameters 
#   will make sure that the algorithm is exposed to the entire train_set. 
# * The retrained model in the above step is used to validate the data in test_test.
# * Algorithm accuracy,sensitivity and specificity values are published in the 
#   respective sections and the complete comparison matrix will be printed in Results section along with the recommendations.

# 5) Model 1 - Novice Heart Disease Model ####

# Before getting into various machine learning models, lets work on a rudimentary model 
# which predicts the presence of heart disease using the prevalence of the heart disease in the dataset.

## Train Algorithm ####

# Calculate the prevelance of heart disease in the train_set and assing it as p
# We have taken the prevalence from train_set to avoid over-fitting.

p <- mean(train_set$hd == "Unhealthy")
p

## Validate Algorithm ####

# Use the above prevalence p and predict the presence of heart disease

set.seed(2008)

# Predict Outcome
y_hat_guessing <- sample( c("Healthy","Unhealthy"),length(test_set$hd),
                          replace = TRUE, prob = c(1-p,p)) %>% 
  factor(levels = levels(test_set$hd))


# Confusion matrix to evaluate the performance
cm <- confusionMatrix(y_hat_guessing, test_set$hd, positive = "Unhealthy")
cm
# Store the results of this algorithm to compare this algorithm against other algorithms
guess_results <- data.frame(method ="Model 1 - Guessing",
                            accuracy = cm$overall[["Accuracy"]],
                            Sensitivity=cm$byClass[["Sensitivity"]],
                            Specificity=cm$byClass[["Specificity"]],
                            F1 = cm$byClass[["F1"]]
) 

# Here is the summary of the results
guess_results

#The above metrics are same as flipping a coin, and we can take this as the baseline and try to beat this algorithm.

# 6) Model 2 - Logistic Regression Heart Disease Model ####

# The next simple model to the Novice Model is linear model, and moreover linear models 
# are easy to interpret and often taken as a baseline model before getting into the complex models. 
# 
# Lets build a linear model to predict the heart disease.
# 
# Logistic Regression is the linear model which is used for Classification problems and 
# lets build this model for Heart Disease Prediction.

## Train Model ####

# Lets fit the logistic regression model using cross validation and look at the fitted model.
# As the results of the logistic regression fit is a random variable , 
# we used cross validation to get the expected values of the accuracy. 
# In logistic regression there are no parameters to the model and hence we used cross validation to 
# get the expected value of the accuracy. 


# Lets do Cross Validation with 10 folds get the expected value of the accuracy

set.seed(2008)
fit_logit = train(
  form = hd ~ .,
  data = train_set,
  trControl = trainControl(method = "cv", number = 10),
  method = "glm",
  family = "binomial"
)

# Here is the summary of the fit
# Maximum likelihood is used to fit the linear model. The coefficients of the fitted model are given below. 
# Note that the coefficients are in log(odds of Unhealthy heart).


summary(fit_logit)


# Here are the accuracy values we got in various folds of the cross validation.

fit_logit$resample

# Here is the Estimated accuracy of the model

fit_logit$results$Accuracy

# This is same as the mean accuracy of all the folds, see here

mean(fit_logit$resample$Accuracy)

## Refit the Model with the entire train_set ####

# As we have seen the estimated performance of the model, we can refit the model with the entire train_set 
# so that the model will get more data to train on, as we have only used partial data to fit the 
# model during the cross validation. 
# 
# Lets refit the model with the train_set and look at McFadden's Pseudo R Squared, see below for formula,
#  this would give us an estimate of the R Squared, by which we can check what percent of the variance 
#  in the output is explained by the Logistic Regression Model.

# McFadden's Pseudo R^2 = [ LL(Null) - LL(Proposed) ] / LL(Null)
# Here LL(Null) is the log likelihood of the Null model and LL(Proposed) is the 
# log likelihood of the logistic regression model.


# Refit the model with all the data before predicting the results
set.seed(2009)
fit_logit_final <- glm(hd ~ ., family = "binomial", data = train_set)

## Now calculate the overall "Pseudo R-squared" and its p-value
ll.null <- fit_logit_final$null.deviance/-2
ll.proposed <- fit_logit_final$deviance/-2

## McFadden's Pseudo R^2 = [ LL(Null) - LL(Proposed) ] / LL(Null)
pseudo_r2 <- (ll.null - ll.proposed) / ll.null

# R Squared
pseudo_r2*100

## The p-value for the R^2
p_value <- 1 - pchisq(2*(ll.proposed - ll.null), df=(length(fit_logit_final$coefficients)-1))
p_value


# The  R Squared value above shows the percent of the variance in the output and the
# p-value 0 tell us that the R Squared value is statistically significant.

## Validate  Model ####

# Lets validate our model by predicting the presence of heart disease in the test_test and look at the results

# Predict the presence of heart disease in the test data using the refitted model from above section

p_hat_logit <- predict(fit_logit_final, newdata = test_set , type = "response")
y_hat_logit <- ifelse(p_hat_logit > 0.5, "Healthy", "Unhealthy") %>% factor ( levels = c("Unhealthy","Healthy"))

# Confusion Matrix
cm <- confusionMatrix(y_hat_logit, test_set$hd)
cm
# Store the results of this algorithm to compare this algorithm against other algorithms
logit_results <- data.frame(method ="Model 2 - Logistic Regression",
                            accuracy = cm$overall[["Accuracy"]],
                            Sensitivity=cm$byClass[["Sensitivity"]],
                            Specificity=cm$byClass[["Specificity"]],
                            F1 = cm$byClass[["F1"]]
) 

results_df <- rbind(guess_results,logit_results)

# Here is the compasion of Logistic Regression model with the Novice model

results_df

# There metrics are far better than the Novice model.
# Lets build few more models and see whether the accuracy improves.

# 7) Model 3 - KNN Heart Disease Mode ####

# Lets build KNN (k nearest neighbors) model and see whether the accuracy improves.

# To implement the algorithm, we can use the `knn3` function from the __caret__ package. 

## Train & Optimize ####

# Lets train the knn model on the train_set using 10 fold cross validation and k values seq(5,200,3) for tuning.

fit_knn = train(
  hd ~ .,
  data = train_set,
  method = "knn",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = expand.grid(k = seq(5, 200, by = 2))
)

max_index <- which.max(fit_knn$results$Accuracy)
best_k <- fit_knn$results$k[max_index]
best_acc <- fit_knn$results$Accuracy[max_index]

# Here is the accuracy of the knn model vs k-values graph from cross validation

plot(fit_knn)

# From the above figure we can see that the k-value that performed better in the cross validation

best_k
best_acc

## Validate Model ####

# Lets refit the knn model on the train_set with the optimal k-value obtained in the above step, 
# as it will give the model an opportunity to train on the entire train_set, 
# and use the model to validate the model on the validation dataset (test_set).

# Refit the model using the entire train_set
set.seed(2008)
final_knn_fit <- knn3(hd ~ ., data = train_set, k = best_k)
# Predict the output using the fitted model
y_hat <- predict(final_knn_fit, newdata = test_set, type = "class")
# Confusion Matrix
cm <- confusionMatrix(y_hat, test_set$hd, positive = "Unhealthy")
cm
# Results
knn_results <- data.frame(method ="Model 3 - KNN",
                          accuracy = cm$overall[["Accuracy"]],
                          Sensitivity=cm$byClass[["Sensitivity"]],
                          Specificity=cm$byClass[["Specificity"]],
                          F1 = cm$byClass[["F1"]]
) 
## Sotre the results for comparison

results_df <- rbind(results_df,knn_results)

#Here is the KNN Model comparison with other models

results_df

# Here is the summary of the KNN Validation Results
# 
# * This accuracy is slightly better than the Novice model 
# * Sensitivity of Novice models better than KNN
# * Logistic regression model is way better than KNN in accuracy, sensitivity and specificity.

# 8) Model 4 - Classification Tree Heart Disease Model ####

# Classification trees, or decision trees, are used in prediction problems where the outcome 
# is categorical so we will use Classification trees to predict the heart disease.
# We are going to use CART package in R to train the algorithm, 
# and CART uses _Gini Index_ to choose the partition, here is the definition of the Gini Index.

## Train & Optimize Model ####

# Lets train and optimize Heart Disease Classification Model using cross validation with 10 folds 
# and use _complexity parameter_ (cp) as tuning parameter with values  seq(0.0, 0.1, len = 25). 
# Lets keep the mtry at default value which is the square root of number of parameters, and minsplit 
# at 0 so that the algorithm gets the flexibility during training.


fit_rpart <- train(hd ~ .,
                   method = "rpart",
                   tuneGrid = data.frame(cp = seq(0.0, 0.1, len = 25)),
                   control = rpart.control(minsplit = 0),
                   data = train_set)

ind <- which.max(fit_rpart$results$Accuracy)
best_cp <- fit_rpart$results$cp[ind]
best_acc <- fit_rpart$results$Accuracy[ind]

# Here is the performance of the Heart Disease Classification Tree for various values of Complexity Parameter

plot(fit_rpart)

best_cp
best_acc

# As we can see from the above graph the best value of Complexity Parameter (cp) is 
# best_cp which has got an accuracy of best_acc during training.

## Validate Model ####

# As we got the parameters that optimized the model, lets retrain the algorithm using the 
# optimal parameters obtained during the above cross training on the entire train_set. 

# Here is the Heart Disease classification tree after the retraining on the entire train_set,

fit_rpart <- rpart(hd ~ ., data = train_set, 
                   control = rpart.control(cp = best_cp, minsplit = 0))
plot(fit_rpart, margin = 0.1)
text(fit_rpart, cex = 0.75)

# Here is the variable importance based on the fit

fit_rpart$variable.importance

# Lets use the trained model and predict the Heart Disease outcome.

y_hat_rpart <- predict(fit_rpart, test_set,type = "class")
cm <- confusionMatrix(y_hat_rpart, test_set$hd)

# Here is the Confusion Matrix based on the predicted outcomes.

cm


# Store Results
rpart_results <- data.frame(method ="Model 4 - Classification Trees",
                            accuracy = cm$overall[["Accuracy"]],
                            Sensitivity=cm$byClass[["Sensitivity"]],
                            Specificity=cm$byClass[["Specificity"]],
                            F1 = cm$byClass[["F1"]]
) 

results_df <- rbind(results_df,rpart_results)

# Here is the model Comparison table

results_df

# As we see from the above Classification Tree Model has the height specificity across all built so far ,  
# but the sensitivity is still less than the Logistic Regression Model.
# 
# As we know that Random Forest improves the performance of the Classification trees by 
# building large number of random trees. Lets build Random Forest Model in the next section.

# 9) Model 5 - Random Forest Heart Disease Model ####

# Random forests are a **very popular** machine learning approach that addresses the shortcomings 
# of decision trees using a clever idea. The goal is to improve prediction performance 
# and reduce instability by _averaging_ multiple decision trees (a forest of trees constructed with randomness). 

# Lets use Random Forest to model Heart Disease predictions.

## Train & Optimize Model ####

# Lets train and optimize Random Forest Heart Disease Model using cross validation with 10 folds 
# and use the following turning parameters,
# 
# 1) mtry:  `r 1:10`
# 
# 2) nodesize: `r seq(1, 51, 10)` 
# 

set.seed(2009)
nodesize <- seq(1, 51, 10)
acc <- map_df(nodesize, function(ns){
  fit <- train(hd ~ ., method = "rf", data = train_set,
               tuneGrid = data.frame(mtry = 1:10),
               nodesize = ns)
  ind <- which.max(fit$results$Accuracy)
  best_mtry <- fit$results$mtry[ind]
  best_acc <- fit$results$Accuracy[ind]
  list(nodesize = ns, mtry =best_mtry, accuracy = best_acc)
  
})

# Here are the results of the cross validation

acc

ind <- which.max(acc$accuracy)
best_mtry <- acc$mtry[ind]
best_nodesize <- acc$nodesize[ind]
best_acc <- acc$accuracy[ind]

best_mtry
best_nodesize
best_acc

# Above table shows the best nodesize and mtry in each fold of the cross validation along with thier accuracy.

# We can see that optimal values for which the max accuracy has achieved, that is 
# `r best_acc` , are mtry = `r best_mtry` and nodesize = `r best_nodesize`.


fit <-       train(hd ~ .,
                   method = "rf", 
                   nodesize = best_nodesize,
                   tuneGrid = data.frame(mtry = 1:10),
                   data = train_set)

ggplot(fit)

# Above plot shows accuracy when we set the optimal nodesize, 
# best_nodesize, and fit the model with various values of mtry, and it clearly shows that the maximum accuracy 
# achived when mtry =  best_mtry.

## Validate Model ####

# We need to retrain the algorithm on the train_set using the parameters that optimized the model 
# in the previous section so that the model is exposed to the entire train_set. 
# 
# Here is the Confusion Matrix from the predictions of the test_set by using the Random Forest Heart Disease Model

# Retrain the model with the optimal parameters
set.seed(2009)
train_rf <- randomForest(hd ~ ., data=train_set, nodesize = best_nodesize , mtry = best_mtry)

# Predict the outcome
y_hat_rf_opt <- predict(train_rf, test_set)

# Confusion Matrix 
cm <- confusionMatrix(y_hat_rf_opt, test_set$hd)

cm

# Store Results
rf_results <- data.frame(method ="Model 5 - Random Forest",
                         accuracy = cm$overall[["Accuracy"]],
                         Sensitivity=cm$byClass[["Sensitivity"]],
                         Specificity=cm$byClass[["Specificity"]],
                         F1 = cm$byClass[["F1"]]
) 

results_df <- rbind(results_df,rf_results)

# Here is the Random Forest Model comparison with other models.

results_df 

# As we see from the above Random Forest Model has performed better than Classification Trees , 
# but the sensitivity is still less than the Logistic Regression Model.

## 10) Model 6 - Ensemble Model ####

# Lets combine the models we have already built in the previous sections to build 
# an Ensemble Model which could improve the overall performance of the predictions.
# 
# The approach that we follow to select the models for Ensemble Model is as follows
# 
# * Calculate the Training Performance of all the models that we built so far except the Novice Model
# * Calculate the combined average of the all the model
# * Select the models that performed greater than or equal to the combined average
# 
# See below for their training accuracy of individual models and their combined average

# Here is the code which does the above calculations

# Logistic Regression Training Accuracy
y_hat_logit_train <- ifelse(fit_logit_final$fitted.values > 0.5, "Healthy", "Unhealthy") %>% factor ( levels = c("Unhealthy","Healthy"))
cm_train_logit <- confusionMatrix(y_hat_logit_train,train_set$hd)
logit_acc <- cm_train_logit$overall["Accuracy"]
logit_acc
# KNN Training Accuracy
y_hat_knn_train <- predict(final_knn_fit, newdata = train_set, type = "class")
cm_train_knn <- confusionMatrix(y_hat_knn_train,train_set$hd)
knn_acc <- cm_train_knn$overall["Accuracy"]
knn_acc
# Classification Trees Training Accuracy
y_hat_rpart_train <- predict(fit_rpart, train_set , type = "class")
cm_rpart_rpart <- confusionMatrix(y_hat_rpart_train, train_set$hd)
rpart_acc <- cm_rpart_rpart$overall["Accuracy"]
rpart_acc
# Random Forest Training Accuracy
y_hat_rf_train <- predict( train_rf, train_set , type = "class")
cm_rf_train <- confusionMatrix(y_hat_rf_train, train_set$hd)
rf_acc <- cm_rf_train$overall["Accuracy"]
rf_acc

# Combined Accuracy
average <- mean(logit_acc,knn_acc,rpart_acc,rf_acc)


# Logistic Regression Heart Disease Model Training Accuracy :  logit_acc
# 
# KNN Heart Disease Model Training Accuracy                 : knn_acc
# 
# Heart Disease Classification Tree Model Training Accuracy : rpart_acc
# 
# Random Forest Heart Disease Model Training Accuracy       : rf_acc
# 
# Combined Accuracy of all the above Models                     : average
# 
# We see that Logistic Regression and Random Forest Models only have their Training Accuracy 
# reater than or equal to the combined accuracy, so we can consider combining these models to 
# improve the accuracy of the predictions. 
# 
# Here is the approach we take for predicting the outcome
# 
# * For every row in test_set, set the Random Forest vote to 1 if Random Forest predicts the outcome 
#   as "Unhealthy" otherwise 0.   
# * For every row in test_set, set the Logistic Regression vote to 1 if Logistic Regression predicts the outcome 
#   as "Unhealthy" otherwise 0.
# * For every row in the test_set, calculate the average vote, by taking the average of Random Forest vote 
#   and the Logistic Regression vote
# * If the average vote > 0.5 then predict the outcome as "Unhealthy" otherwise "Healthy"
# 
# Here are the results of the prediction after combining Logistic Regression Heart Disease Model and 
# Random Forest Heart Disease Models

# Logit Votes
logit_votes_for_unhealthy <- ifelse(y_hat_logit == "Unhealthy",1,0)
# Random Forest Votes
rf_votes_for_unhealthy <- ifelse(y_hat_rf_opt == "Unhealthy",1,0)
my_list <- list(logit = logit_votes_for_unhealthy , rf = rf_votes_for_unhealthy)
model_predictions_df <- as_tibble(my_list)
# Average of Logit and Random Forest Votes
votes <- rowMeans(model_predictions_df)
# Predict Outcome
y_hat <- ifelse(votes > 0.5, "Unhealthy", "Healthy") %>% factor(levels=c("Unhealthy","Healthy"))
# Prediction Accuracy
mean(y_hat == test_set$hd)
# Confusion Matrix
cm <- confusionMatrix(y_hat,test_set$hd)
cm
# Store Results
ensembles_results <- data.frame(method ="Model 6 - Ensemble Model",
                                accuracy = cm$overall[["Accuracy"]],
                                Sensitivity=cm$byClass[["Sensitivity"]],
                                Specificity=cm$byClass[["Specificity"]],
                                F1 = cm$byClass[["F1"]]
) 

results_df <- rbind(results_df,ensembles_results)

# Here is the Ensemble Model comparison with other models.

results_df

# As we see from the above Ensemble Model has the height accuracy and specificity across 
# all the models but the sensitivity is still less than the Logistic Regression Model

# 11) Results Comparison & Model Recommendation####

# Now that we have completed building the Models for Hear Disease Prediction System, 
# lets look at all Model performances and analyze the results.
# 
# Here is the table which summarizes the performance of all the models


results_df

## Results Comparison ####

# We started with Model 1 - Guessing as a novice model which has not considered any of the independent variables, 
# and this model is only use to give us a base line estimate. 
# This models accuracy and specificity are same as flipping a coin, sensitivity is little high, 
# due to prevalence of the Heart Disease.
# 
# Model 2 - Logistic Regression has second height accuracy after Model 6 - Ensemble Model, 
# and height sensitivity among all the models,But this model's specificity, is less than the model with the highest specificity.
# As it is a linear model it is simple and easy to interpret. 
# This model could be chosen to predict the Heart Disease if sensitivity is more important than specificity,
# that is predicting the patient with Heart Disease as Heart Disease is more important than predicting the patient 
# who is Healthy as Healthy by taking a little compromise in specificity. 
# Slight compromise in sensitivity would slightly increase the chance of predicting Unhealthy patients as Healthy, 
# and this incorrect diagnosis could put a patient in danger, where as a slight compromise in specificity 
# would slightly increase the chance of predicting the patient who is Healthy as Unhealthy and in which case the patient's 
# incorrect diagnosis adverse impact is less. 
# Based on the above argument we could recommend Logistic Regression when sensitivity is more important 
# than the specificity while predicting the Heart Disease.
# 
# Model 3 - KNN Models accuracy and specificity are higher than the Novice Model but sensitivity is less 
# than the Novice Model. Sensitivity of this model is  not even close to flipping a coin and 
# hence this model would not be recommended for predicting the Heart Disease. 
# 
# Model 4 - Classification Trees Model accuracy is better than Novice and KNN models, 
# but still less than the logistic regression model , 
# and is better than Logistic Regression Model but not better than the Ensemble Model. 
# Its sensitivity is less than the Logistic Regression Model. The main advantage of this model is its interpretation, 
# very easy to interpret and even so that the logistic regression model. 
# As there is a significant difference in the sensitivity with the best model and moreover this model's 
# variance would be high as it would most frequently get over-fitted with the training data, 
# we would not recommend this for the prediction. 
# However we could use this models variable importance to understand how individual variables contribution to the output.
# 
# Model 5 - Random Forest Model accuracy, is better than Novice and KNN models, and same as the logistic regression model, 
# specificity is  better than Logistic Regression Model but not better than the Ensemble Model. 
# Its specificity is more than the Logistic Regression Model which put this model at advantage 
# if specificity is more than sensitivity. 
# Due to the randomness introduced during the training from random selection of variables 
# during model fitting and the random samples from bootstrapping reduces the variance in the predictions.
# Its main drawback of this model is that we lose the model interpretation.
# 
# Model 6 - Ensemble Model accuracy is higer than Novice, KNN , logistic regression model and Classification Trees models. 
# Its specificity is higher than all other models, and in specific it is more than the Logistic Regression Model 
# which put this model at advantage if specificity is more than sensitivity. 
# The main downside of the model is loss of interpretation due to the average of all the models.
# 
## Model Recommendation ####
# In summary we could choose Model 2 - Logistic Regression if sensitivity is more important than the specificity. 
# If specificity and overall accuracy is more important than sensitivity we could choose the Model 6 - Ensemble Model 
# as it has got higher values of both accuracy and specificity.

# 12) Conclusion ####

# We have started the project with data download and clean up, followed by partitioning the data into train_set and test_set. 
# We built various machine learning models to predict whether the patient has Heart Disease 
# in Methods section and compared all the models in the Results section and in the 
# same section we discussed individual models pro's and con's, and recommended 
# Model 2 - Logistic Regression if sensitivity is more important than the specificity, 
# and if specificity and overall accuracy is more important than sensitivity we recommended 
# Model 6 - Ensemble Model. 

# Reset the warnings to normal
options(warn = oldw)




