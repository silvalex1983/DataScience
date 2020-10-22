
install.packages("rpart")
install.packages("hmeasure")
install.packages("rpart.plot")
library(rpart.plot)
library(hmeasure)
library(rpart)

#arvore regressão
dados <- TELEGV[-c(1,11:15)]
dados$PR0M0=ifelse(dados$PR0M0 == 0,"Nao","Sim")

dados

#imprimir as colunas da tabela
col_list<- paste(colnames(dados),collapse = "+")
col_list


#rodar uma arvore de classificação

ad= rpart(data= dados, PR0M0~PROD_A+PROD_B+PROD_C+REGIAO+SETOR+TMPCLI+SEDE+FUNC) #daria na mesma ad=rpart(data=lrn,cancel~.)

ad

printcp(ad)

#prp(ad,type=3,extra=101,nn=T,fallen.leaves = T,branch.col ="salmon",branch.lty=5,box.col=c("blue","green"))


#probabilidade do unico individuo

PROD_A <- 19
PROD_B<- 7.98
PROD_C <- 8.36
REGIAO <- 'SUL'
SETOR <- 'COM'
TMPCLI <- 'OLD'
SEDE <- 0
FUNC <- 10

tst <- data.frame(PROD_A,PROD_B,PROD_C,REGIAO,SETOR,TMPCLI,SEDE,FUNC)

prob <- predict(ad,newdata = tst, type="prob")

tst$p_nao<- prob[,1]
tst$p_sim<- prob[,2]

head(prob) 

tst


#probabilidade do unico individuo


# Roc da arvore
prob <- predict(ad,newdata = dados, type="prob")
head(prob) 

dados$p_nao<- prob[,1]
dados$p_sim<- prob[,2]

dados


HMeasure(dados$PR0M0, dados$p_sim)$metrics

xx=roc(dados$PR0M0, dados$p_sim)
plot(xx)
xx

#fim roc da arvore


#poda da arvore

printcp(ad)

ad2 = prune(ad,cp=.09)
prp(ad2,type=3,extra=104,nn=T,fallen.leaves = T,branch.col ="salmon",branch.lty=5,box.col=c("blue","green"))
printcp(ad2)

#fim poda da arvore



#laredo

dados <- TELEGV[-c(1,11:15)]

dados$PR0M0=ifelse(dados$PR0M0 == 0,"0","1")
set.seed(123)

flag = sample(1:2100,1400)
head(flag)

#criar os dois arquivos
lrn<- dados[flag,]
tst<- dados[-flag,]

set.seed(111)
ad= rpart(data= lrn, PR0M0~PROD_A+PROD_B+PROD_C+REGIAO+SETOR+TMPCLI+SEDE+FUNC) #daria na mesma ad=rpart(data=lrn,cancel~.)

printcp(ad)


PROD_A <- 18
PROD_B<- 10
PROD_C <- 5
REGIAO <- 'NORTE'
SETOR <- 'IND'
SEDE <- 1
FUNC <- 8

tst <- data.frame(PROD_A,PROD_B,PROD_C,REGIAO,SETOR,TMPCLI,SEDE,FUNC)

prob <- predict(ad,newdata = tst, type="prob")
prob
