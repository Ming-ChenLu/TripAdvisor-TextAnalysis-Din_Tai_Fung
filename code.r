setwd("/Users/Amy/Desktop/Learning from Big Data/dataset")
library(stringr)
library(tidyr)
library(dplyr)
library(ggplot2)
library(tidyverse)
library(lubridate)
library(tm)
library(wordcloud)
library(tidytext)
library(readr)
library(e1071)
library(Matrix)

#training text cleaning####
train_food = read_file("train_food.txt")
train_service = read_file("train_service.txt")
train_value = read_file("train_value.txt")
t_food = Corpus(VectorSource(train_food))
t_service = Corpus(VectorSource(train_service))
t_value = Corpus(VectorSource(train_value))
t_food = t_food %>%
  tm_map(tolower) %>%
  tm_map(removeNumbers) %>%
  tm_map(removePunctuation) %>%
  tm_map(stripWhitespace) %>%
  tm_map(removeWords, c("the", "and", "pardpardeftabpartightenfactor", "bfsfsmilli","fieldfldinsthyperlink",
                        "httpswwwwillflyforfoodnetnightmarketsandfoodstreetstovisitintaiwanjiufenoldstreetfldrslt",
                        "httpswwwwillflyforfoodnetthingstoeatonjiufenoldstreettaiwanfldrslt",
                        "httpswwwwillflyforfoodnetnightmarketsandfoodstreetstovisitintaiwanliouhetouristnightmarketfldrslt",
                        "pardintblitappardeftabpartightenfactor","theyre","youll","thats","ulc", "many",
                        stopwords("English")))
t_service = t_service %>%
  tm_map(tolower) %>%
  tm_map(removeNumbers) %>%
  tm_map(removePunctuation) %>%
  tm_map(stripWhitespace) %>%
  tm_map(removeWords, c("the", "and", "pardpardeftabpartightenfactor", "expndexpndtwkerning","kerningexpndexpndtw", 
                        "listtext","lsilvlcb","fieldfldinsthyperlink","discleveltextleveltemplateiducu",
                        "evelnumbersfililin","pardtxtxpardeftablifipartightenfactor","aafieldfldinsthyperlink",
                        "levelnumbersfililin", "listlisttemplateidlisthybridlistlevellevelnfclevelnfcnleveljcleveljcnlevelfollowlevelstartatlevelspacelevelindentlevelmarker",
                        "colortblredgreenblueredgreenblueredgreenblueredgreenblue", "expandedcolortblcssrgbccccssrgbccccssrgbccc",
                        "can","theyll", stopwords("English")))
t_value = t_value %>%
  tm_map(tolower) %>%
  tm_map(removeNumbers) %>%
  tm_map(removePunctuation) %>%
  tm_map(stripWhitespace) %>%
  tm_map(removeWords, c("the", "and", "pardpardeftabslsapartightenfactor", "many","fsfsmilli","may","thus","bfs",
                        stopwords("English")))
t_food = t_food %>% DocumentTermMatrix()
t_service = t_service %>% DocumentTermMatrix()
t_value = t_value %>% DocumentTermMatrix()
tt_food = tidy(t_food)
tt_service = tidy(t_service)
tt_value = tidy(t_value)
set.seed(123)
tt_food_new = sample_n(tt_food, 452)
tt_value_new = sample_n(tt_value, 452)

#likelihood for training data####
t_df = data.frame(tt_food_new, tt_service, tt_value_new)
colnames(t_df) = c("1","t_food","c_food","2","t_service","c_service","3","t_value","c_value")
t_df[,c(1,4,7)] = NULL
sum(t_df$c_food,t_df$c_service,t_df$c_value)
t_df = t_df %>%
  mutate(l_food = log((t_df$c_food+1)/(452+3032))) %>%
  mutate(l_service = log(t_df$c_service+1)/(452+3032)) %>%
  mutate(l_value = log((t_df$c_value+1)/(452+3032)))
t_df_food = t_df[,c(1,2,7)]
t_df_service = t_df[,c(3,4,8)]
t_df_value = t_df[,c(5,6,9)]

#reviews cleaning & find likelihood of each review####
rv = read.csv("scraper.csv")
attach(rv) 
rv = rv %>% unite(text, c(review,X,X.1,X.2,X.3,X.4,X.5,X.6), sep = "")
rv$text = as.vector(rv$text)
rv_corpus = Corpus(VectorSource(rv$text))
rv_corpus = rv_corpus %>%
  tm_map(tolower) %>%
  tm_map(removeNumbers) %>%
  tm_map(removePunctuation) %>%
  tm_map(stripWhitespace) %>%
  tm_map(removeWords, c("the", "and",stopwords("en")))
  
dtm_rv = DocumentTermMatrix(rv_corpus)
inspect(dtm_rv[3930:3937,1:10])
freq = data.frame(sort(colSums(as.matrix(dtm_rv)), decreasing=T))
tidy_rv = tidy(dtm_rv)

t_df = merge(t_df_food,t_df_service, by.x = 1, by.y = 1, all.x = T, all.y = T)
t_df = merge(t_df,t_df_value, by.x = 1, by.y = 1, all.x = T, all.y = T)
colnames(t_df) = c("term","c_food","l_food","c_service","l_service","c_value","l_value")

rv_df = data.frame(tidy_rv$document,tidy_rv$term, tidy_rv$count)
rv_df = merge(rv_df,t_df,by.x = 2, by.y = 1, all.y = T)
rv_df = rv_df[!is.na(rv_df$tidy_rv.document),]
rv_df[is.na(rv_df)] = 1e-20

#calculate posterior for each review####

z = matrix(nrow = max(as.numeric(rv_df$tidy_rv.document)),ncol = 3)

#note that doc 430 and 440 are missing
for(l in c(431:max(as.numeric(rv_df$tidy_rv.document)))){
  x = rv_df[rv_df$tidy_rv.document == l,]
  y = matrix(nrow = sum(as.numeric(x$tidy_rv.count))+1,ncol = 3)
  y[1,] = c(1/3,1/3,1/3)
  for(k in c(1:sum(as.numeric(x$tidy_rv.count)))){
    for(i in c(1:nrow(x))){
      for(j in c(1:x[i,3])){
        y[k+1,1] = y[k,1]*x[i,5]/(y[k,1]*x[i,5]+y[k,2]*x[i,7]+y[k,3]*x[i,9])
        y[k+1,2] = y[k,2]*x[i,7]/(y[k,1]*x[i,5]+y[k,2]*x[i,7]+y[k,3]*x[i,9])
        y[k+1,3] = y[k,3]*x[i,9]/(y[k,1]*x[i,5]+y[k,2]*x[i,7]+y[k,3]*x[i,9])
      }
    }
  }
  z[l,] = y[k,]
}

z1 = z %>% as.data.frame() %>%
  mutate( max = apply(z, 1, which.max)) 
z_final = z1 %>% mutate( doc = c(1:3937))

#matching result into rv_df
rv_df = rv_df %>%
  mutate( dim_result = z_final$max[match(rv_df$tidy_rv.document,z_final$doc)])

#split reviews into 3 dimension with max posterior
##other way doing it: select(filter(z, z$max == 2),c("V1","V2","V3")) from "dplyr"
dim1 = subset(rv_df, rv_df$dim_result == 1, select = c("tidy_rv.term","tidy_rv.document","tidy_rv.count"))
dim2 = subset(rv_df, rv_df$dim_result == 2, select = c("tidy_rv.term","tidy_rv.document","tidy_rv.count"))
dim3 = subset(rv_df, rv_df$dim_result == 3, select = c("tidy_rv.term","tidy_rv.document","tidy_rv.count"))

#sentiment text cleaning & likelihood####
s_pos = read_file("positive_sentiment.txt")
s_neg = read_file("negative_sentiment.txt")
s_pos = Corpus(VectorSource(s_pos))
s_neg = Corpus(VectorSource(s_neg))
s_pos = s_pos %>%
  tm_map(tolower) %>%
  tm_map(removeNumbers) %>%
  tm_map(removePunctuation) %>%
  tm_map(stripWhitespace) %>%
  tm_map(removeWords, c("the", "and", "pardpardeftabpartightenfactor","expandedcolortblcssrgbccccssrgbccc",
                        "expndexpndtwkerning", "fonttblffnilfcharset", "paperwpaperhmarglmargrviewwviewhviewkind",
                        "rtfansiansicpgcocoartfcocoasubrtf",stopwords("English")))
s_neg = s_neg %>%
  tm_map(tolower) %>%
  tm_map(removeNumbers) %>%
  tm_map(removePunctuation) %>%
  tm_map(stripWhitespace) %>%
  tm_map(removeWords, c("the", "and", "pardpardeftabslsapartightenfactor", "strokec", "fsfsmilli","ffsfsmilli",
                        "fieldfldinsthyperlink", "aafieldfldinsthyperlink", "wasafieldfldinsthyperlink",
                        "rtfansiansicpgcocoartfcocoasubrtf", "reviewsfieldfldinsthyperlink", "reviewedafieldfldinsthyperlink",
                        "redgreenblueredgreenblue", "ffs", "aacf",
                        stopwords("English")))
s_pos = s_pos %>% DocumentTermMatrix()
s_neg = s_neg %>% DocumentTermMatrix()
ts_pos = tidy(s_pos)
ts_neg = tidy(s_neg)
set.seed(123)
ts_neg_new = sample_n(ts_neg, 587)

s_df = data.frame(ts_pos, ts_neg_new)
colnames(s_df) = c("1","t_pos","c_pos","2","t_neg","c_neg")
s_df[,c(1,4)] = NULL
sum(s_df$c_pos,s_df$c_neg)
s_df = s_df %>%
  mutate(l_pos = ((s_df$c_pos+1)/(587+1944))) %>%
  mutate(l_neg = ((s_df$c_neg+1)/(587+1944)))
s_df_pos = s_df[,c(1,2,5)]
s_df_neg = s_df[,c(3,4,6)]

s_df = merge(s_df_pos,s_df_neg, by.x = 1, by.y = 1, all.x = T, all.y = T)
colnames(s_df) = c("term","c_pos","l_pos","c_neg","l_neg")

#sentiment posterior for each dimension####

dim1_df = merge(dim1,s_df,by.x = 1, by.y = 1, all.y = T)
dim1_df = dim1_df[!is.na(dim1_df$tidy_rv.document),]
dim1_df[is.na(dim1_df)] = 1e-10
dim1_df = dim1_df %>% mutate( seq = c(1:19300))
dim2_df = merge(dim2,s_df,by.x = 1, by.y = 1, all.y = T)
dim2_df = dim2_df[!is.na(dim2_df$tidy_rv.document),]
dim2_df[is.na(dim2_df)] = 1e-10
dim2_df = dim2_df %>% mutate( seq = c(1:19916))
dim3_df = merge(dim3,s_df,by.x = 1, by.y = 1, all.y = T)
dim3_df = dim3_df[!is.na(dim3_df$tidy_rv.document),]
dim3_df[is.na(dim3_df)] = 1e-10
dim3_df = dim3_df %>% mutate( seq = c(1:2498))

#weird
x = dim2_df %>% filter(tidy_rv.document == 3924)
y = matrix(nrow = 2498 ,ncol = 2)
y[1,] = c(1/2,1/2)
for(k in c(1:sum(as.numeric(dim1$tidy_rv.count)))){
for(i in c(1:nrow(x))){
  for(j in c(1:x[i,3])){
    y[k+1,1] = y[k,1]*x[i,5]/(y[k,1]*x[i,5]+y[k,2]*x[i,7])
    y[k+1,2] = y[k,2]*x[i,7]/(y[k,1]*x[i,5]+y[k,2]*x[i,7])
  }
  y[i+1,] = y[k,]
}
}

#Exploratory data analysis####
##With total 3937 reviews on TripAdvisor for Din Tai Fung, the review date range from 2011-08-17 to 2018-09-09
rv = rv[complete.cases(rv),]
rv$date = as.Date(rv$X...date)
str(txt)
#ggplot
rv %>%
  count(Year = round_date(rv$date, "year")) %>%
  ggplot(aes(Year, n)) +
  geom_line() +
  ggtitle('The Number of Reviews Per Year')
table(month(rv_analysis$date))
##From the number of reviews per week, there is flunctuation between each month.
##And we can see number of reviews per week gradually increased until 2016, then slowly decreased afterward.
#wordcloud
dtm_rv1 = as.matrix(dtm_rv)
rownames(dtm_rv1) = 1:nrow(dtm_rv1)
freq = data.frame(sort(colSums(dtm_rv1), decreasing=TRUE))
wordcloud(rownames(freq), freq[,1], scale = c(3,.5), 
          max.words=120, colors=brewer.pal(8, "Dark2"))

#analysis####
#wordcloud
c1 = Corpus(VectorSource(dim1$tidy_rv.term))
d1_dtm = DocumentTermMatrix(c)
d1_dtm = as.matrix(d1_dtm)
rownames(d1_dtm) = 1:nrow(d1_dtm)
freq = data.frame(sort(colSums(d1_dtm), decreasing=TRUE))
wordcloud(rownames(freq), freq[,1], scale = c(3,.5), 
          max.words=120, colors=brewer.pal(8, "Dark2"))

c2 = Corpus(VectorSource(dim2$tidy_rv.term))
d2_dtm = DocumentTermMatrix(c)
d2_dtm = as.matrix(d2_dtm)
rownames(d2_dtm) = 1:nrow(d2_dtm)
freq = data.frame(sort(colSums(d2_dtm), decreasing=TRUE))
wordcloud(rownames(freq), freq[,1], scale = c(3,.5), 
          max.words=120, colors=brewer.pal(8, "Dark2"))

c3 = Corpus(VectorSource(dim3$tidy_rv.term))
d3_dtm = DocumentTermMatrix(c)
d3_dtm = as.matrix(d3_dtm)
rownames(d3_dtm) = 1:nrow(d3_dtm)
freq = data.frame(sort(colSums(d3_dtm), decreasing=TRUE))
wordcloud(rownames(freq), freq[,1], scale = c(3,.5), 
          max.words=120, colors=brewer.pal(8, "Dark2"))

#Average posterior for each dimension
subset(rv_df, rv_df$dim_result == 3, select = c("l_food","l_service","l_value")) %>% summary()

#ggplots
rv = cbind(rv, z_final)
rv_analysis = rv[,c(-1,-4,-5,-6)] 
rv_analysis$cnt = 1
rv_analysis$max =  as.numeric(rv_analysis$max)

d1 = rv_analysis %>% filter(rv_analysis$max == 1) %>%
  group_by(month=floor_date(date, "month")) %>%
  summarize(cnt = sum(cnt))
ggplot(data = d1, aes(x = month, y = cnt)) +
  geom_bar(stat = "identity") +
  labs(title = "Monthly reviews of food",
       x = "Date", y = "Numbers of reviews") +
  scale_y_continuous(limit = c(0,80))

d2 = rv_analysis %>% filter(rv_analysis$max == 2) %>%
  group_by(month=floor_date(date, "month")) %>%
  summarize(cnt = sum(cnt))
ggplot(data = d2, aes(x = month, y = cnt)) +
  geom_bar(stat = "identity") +
  labs(title = "Monthly reviews of service",
       x = "Date", y = "Numbers of reviews") +
  scale_y_continuous(limit = c(0,80))

d3 = rv_analysis %>% filter(rv_analysis$max == 3) %>%
  group_by(month=floor_date(date, "month")) %>%
  summarize(cnt = sum(cnt))
ggplot(data = d3, aes(x = month, y = cnt)) +
  geom_bar(stat = "identity") +
  labs(title = "Monthly reviews of value",
       x = "Date", y = "Numbers of reviews") +
  scale_y_continuous(limit = c(0,80))

