#Prepare a classification model using Naive Bayes for salary data 
#Data Description:

#age     -- age of a person work
#class	-- A work class is a grouping of work 
#education -- Education of an individuals	
#maritalstatus -- Marital status of an individulas	
#occupation -- occupation of an individuals
#relationship -- 	
#race --  Race of an Individual
#sex --  Gender of an Individual
#capitalgain -- profit received from the sale of an investment	
#capitalloss -- A decrease in the value of a capital asset
#hoursperweek -- number of hours work per week	
#native -- Native of an individual
#Salary -- salary of an individual


salary_train_data <- read.csv(file.choose())
salary_test_data <- read.csv(file.choose())

str(salary_train_data)
str(salary_test_data)

## will check if any missing values are available in dataset

sum(is.na(salary_train_data))
sum(is.na(salary_test_data))

## both do not have any missing value so we can go ahead

## lets check colinearity problem in int columns

pairs(cor(salary_train_data[c(1,4,10,11,12)]))

library(corpcor)

cor2pcor(cor(salary_train_data[c(1,4,10,11,12)]))

#  [,1]       [,2]        [,3]        [,4]       [,5]
#  [1,] 1.00000000 0.01600693  0.07242278  0.05676179 0.08949733
#  [2,] 0.01600693 1.00000000  0.11540956  0.07631784 0.13813507
#  [3,] 0.07242278 0.11540956  1.00000000 -0.04920819 0.05761555
#  [4,] 0.05676179 0.07631784 -0.04920819  1.00000000 0.03837653
#  [5,] 0.08949733 0.13813507  0.05761555  0.03837653 1.00000000

## there is no strong relationship between any 2 varaible, so no colinearity prob

## lets try to analyse categorical variable against salary 

plot(salary_train_data$education,salary_train_data$Salary)
plot(salary_train_data$workclass,salary_train_data$Salary)
plot(salary_train_data$maritalstatus,salary_train_data$Salary)
plot(salary_train_data$relationship,salary_train_data$Salary)
plot(salary_train_data$race,salary_train_data$Salary)
plot(salary_train_data$sex,salary_train_data$Salary)
plot(salary_train_data$native,salary_train_data$Salary)

## now we will build our first navie bayes model

library(e1071)

salary_model1 <- naiveBayes(salary_train_data$Salary~.,data = salary_train_data)
salary_model1

### model has given the classficiation of data in <=50K,>50K in this two classes

prop.table(table(salary_train_data$Salary))

##  <=50K      >50K 
##  0.7510693 0.2489307 

## lets predict salary class for train data

train_salary_pred <- predict(salary_model1,salary_train_data)
train_salary_pred

## lets plot confusion matrix 

table(train_salary_pred,salary_train_data$Salary)

# train_salary_pred  <=50K  >50K
#             <=50K  21103  3805
#             >50K    1550  3703

## mow lets evaluate the model performance ,check accuracy

mean(train_salary_pred == salary_train_data$Salary) ## accuracy is 0.8224528

## now we will apply model-1 on test data set

test_salary_pred <- predict(salary_model1,salary_test_data)

## confusion matrix

table(test_salary_pred,salary_test_data$Salary)

# test_salary_pred  <=50K  >50K
#            <=50K  10550  1911
#           >50K     810  1789

## Accuracy

mean(test_salary_pred == salary_test_data$Salary) ## accuracy is 0.8193227
