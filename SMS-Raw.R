## Build a naive Bayes model on the data set for classifying the ham and spam

sms_data <- read.csv(file.choose(),stringsAsFactors = F)

## here we have to classify gien text into ham and spam

str(sms_data)

sms_data$type <- factor(sms_data$type)

table(sms_data$type)
# ham  spam 
# 4812  747 

install.packages("tm")
install.packages("NLP")
library(NLP)
library(tm)


# now will prepare corpus for the data

sms_corpus <- Corpus(VectorSource(sms_data$text))
sms_corpus$content[1:10]


## now we will remove unwanted symbol i.e. data cleaning

corpus_clean <- tm_map(sms_corpus,tolower)
corpus_clean <- tm_map(corpus_clean,removeNumbers)
corpus_clean <- tm_map(corpus_clean,removeWords,stopwords())
corpus_clean <- tm_map(corpus_clean,removePunctuation)
remove_punct <- function(x) gsub("[^[:alpha:] [:space:]]*","",x)
corpus_clean <- tm_map(corpus_clean,content_transformer(remove_punct))
corpus_clean <- tm_map(corpus_clean,stripWhitespace)
class(corpus_clean)

corpus_clean$content[1:10]

## create document term sparse matrix

sms_dtm <- DocumentTermMatrix(corpus_clean)
class(sms_dtm)

## now we will split data into train and test using sequential splitting

sms_raw_train <- sms_data[1:4169,]
sms_raw_test <- sms_data[4170:5559,]

sms_dtm_train <- sms_dtm[1:4169,]
sms_dtm_test <- sms_dtm[4170:5559,]

sms_corpus_train <- corpus_clean[1:4169]
sms_corpus_test <- corpus_clean[4170:5559]


## lets check proportion of ham and spam in every model

prop.table(table(sms_data$type))

# ham      spam 
# 0.8656233 0.1343767 

prop.table(table(sms_raw_train$type))

# ham      spam 
# 0.8647158 0.1352842 

## now will check frequency of words :

sms_dict <- findFreqTerms(sms_dtm_train,3)
list(sms_dict[1:100])

sms_train <- DocumentTermMatrix(sms_corpus_train,list(dictionary = sms_dict))
sms_test <- DocumentTermMatrix(sms_corpus_test,list(dictionary = sms_dict))

## now we will convert counts to factor
convert_counts <- function(x) { 
  x <- ifelse(x > 0,1,0 )
  x <- factor (x,levels = c(0,1),labels = c("No","Yes"))}

## Now will apply this convert counts on train/test data column thts why we have given margin =2

sms_train <- apply(sms_train , MARGIN = 2,convert_counts)
sms_test <- apply(sms_test,MARGIN = 2,convert_counts)

View(sms_train)
View(sms_test)

## now we wil build our model on training data

install.packages(e1071)
library(e1071)

sms_classifier <- naiveBayes(sms_train,sms_raw_train$type)
sms_classifier$levels

## now will evaluate performance of model

sms_test_pred <- predict(sms_classifier,sms_test)
sms_test_pred[1:25]

table1 <- table(sms_test_pred,sms_raw_test$type)
table(sms_raw_test$type,sms_test_pred)
## sms_test_pred
##       ham spam
## ham  1203    4
## spam   28  155

table(sms_raw_test$type)

# ham spam 
# 1207  183 

library(gmodels)

CrossTable(sms_test_pred,sms_raw_test$type ,
           prop.chisq = FALSE,prop.t = FALSE , prop.r = FALSE,
           dnn = c('predicted','actual'))

sms_classifier2 <- naiveBayes(sms_train,sms_raw_train$type,laplace = 11)
sms_test_pred2 <- predict(sms_classifier2,sms_test)


table2 <- table(sms_test_pred2,sms_raw_test$type)

CrossTable(sms_test_pred2,sms_raw_test$type,
           prop.chisq = FALSE,prop.t = FALSE,prop.r = FALSE,
           dnn = c('predicted','actual'))

## Accuracy

accuracy1 <- (sum(diag(table1))/sum(table1))
accuracy1 ## accuracy for this model is 0.9769784

accuracy2 <- (sum(diag(table2))/sum(table2))
accuracy2 ## accuracy for this model is 0.8726619

## we can get accuracy by using below code as well :

mean(sms_test_pred==sms_raw_test$type) 

mean(sms_test_pred2==sms_raw_test$type)
