
install.packages("rpart")
install.packages("hmeasure")
install.packages("rpart.plot")
install.packages("caret")
install.packages("gmodels")
install.packages("arules")
library(arules)
library(gmodels)
library(rpart.plot)
library(hmeasure)
library(rpart)
library(caret)
library(boot)

dados <- TELEGV[-c(1,11:15)]
View(dados)

col_list<- paste(colnames(dados),collapse = "+")
col_list

fit=glm(data=dados,PR0M0~PROD_A+PROD_B+PROD_C+REGIAO, family = binomial())
fit

dados$p_sim <- predict(fit,newdata=dados,type="response")
PC<- 0.6
dados$klass= ifelse(dados$p_sim > PC,1,0)
dados
table(dados$PR0M0,dados$klass)

m<- (566+ 273)/ 2100
m

Kprob<- discretize(dados$p_sim,method="frequency", breaks = 8)
CrossTable(Kprob,dados$PR0M0,prop.c = FALSE,prop.t = FALSE,prop.chisq = FALSE)


