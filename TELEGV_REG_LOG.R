#Analise TELEGV

set.seed(3035)

dados <- telegv_cat
View(dados)
dados <- dados[-c(10:17)]

dados$PR0M0 <- ifelse(dados$PR0M0== 1, "SIM","NAO")
dados$ALVO <- ifelse(dados$PR0M0== "SIM", 1,0)

View(dados)

library(arules)
kprodA <- discretize(dados$PROD_A  , method = 'frequency', categories=5)
kprodB <-discretize(dados$PROD_B  , method = 'frequency', categories=5)
kprodC <-discretize(dados$PROD_C  , method = 'frequency', categories=5)
kfunc <-discretize(dados$FUNC  , method = 'frequency', categories=5)

table(kprodA, dados$PR0M0) 
table(kprodB, dados$PR0M0)  
table(kprodC, dados$PR0M0)  
table(kfunc, dados$PR0M0)  

#par(mfrow=c(1,4)) 
# tres gráficos alinhados (1 linha, 3 colunas)
#boxplot(dados$PROD_A~dados$PR0M0, xlab="PROD_A") 
#boxplot(dados$PROD_B~dados$PR0M0, xlab="PROD_B") 
#boxplot(dados$PROD_C~dados$PR0M0, xlab="PROD_C")
#boxplot(dados$FUNC~dados$PR0M0, xlab="FUNC")




#CrossTable(dados$SETOR,tst$PR0M0,prop.c = FALSE,prop.t = FALSE,prop.chisq = FALSE)
#CrossTable(dados$REGIAO,tst$PR0M0,prop.c = FALSE,prop.t = FALSE,prop.chisq = FALSE)
#CrossTable(dados$TMPCLI,tst$PR0M0,prop.c = FALSE,prop.t = FALSE,prop.chisq = FALSE)


# vamos analisar as vars
dados$SEDE=as.factor(dados$SEDE)

m=table(dados$SEDE,dados$PR0M0)
mp=prop.table(m,1)
m;mp



col_list<- paste(colnames(dados),collapse = "+")
col_list

flag=sample(1:2100,1400)
head(flag)

lrn<-dados[flag,]
tst<-dados[-flag,]

#View(lrn)
#View(tst)

#vamos rodar a reg log
fit=glm(data=lrn,ALVO~PROD_A+PROD_B+PROD_C+REGIAO+SETOR+TMPCLI+SEDE+FUNC, family = binomial())

summary(fit)

print(fit,digits=3)

fit2=step(fit)

fit2
summary(fit2)

fit2=glm(data=lrn,ALVO~PROD_A+PROD_B+TMPCLI+SEDE+FUNC, family = binomial())

tst$psim <- predict(fit2,newdata=tst,type="response")

tst$psim
View(tst)

#vamos adotar como ponto de corte 

PC= 0.5

tst$klass= ifelse(tst$psim>PC,"yes","no")

table(tst$ALVO,tst$klass)

#no yes
#nao 481  31
#sim  69  80

Taxa=(31+69)/800

#no yes
#nao 564  51
#sim  94  91
#Taxa=(51+94)/800    VERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR


#vamos bater os olhos para a aderencia 

install.packages("arules")
library(arules)

Kprob<- discretize(tst$psim,method="frequency", breaks = 7)

install.packages("gmodels")
library(gmodels)

CrossTable(Kprob,tst$PR0M0,prop.c = FALSE,prop.t = FALSE,prop.chisq = FALSE)


#Obs:::  Crosfit serve para se encontrar a taxa de erro.

install.packages("hmeasure")
library(hmeasure)

HMeasure(tst$ALVO,tst$psim)$metrics




#----------------------------



##Load of packages:
install.packages("ggplot2")
library(ggplot2)
install.packages("rpart")
library(rpart)
install.packages("rpart.plot")
library(rpart.plot)
library(readxl)

TELEGV <- telegv_cat
qq_com_lixo<-TELEGV

View(qq_com_lixo)
#Tira as colunas nao usadas
qq<-qq_com_lixo[,-c(10:17)]

#qq$SEDEPROP<-ifelse(qq$SEDE==1,"SIM","NAO")
#qq$FUNCQTD<-ifelse(qq$FUNC>12,"ENTRE 13 E 25","AT? 12")
View(qq)
###dividir o arquivo em duas partes
set.seed(3035)
flag=sample(1:2100,1400)
head(flag)

###Criar os dois arquivos 
#Um arquivo lrn somente com o campo flag
lrn=qq[flag,] 
View(lrn)
#Um arquivo tst com todos os campos exceto o campo flag
tst=qq[-flag,]
View(tst)
tgv<-tst

prop.table(table(lrn$PR0M0))
prop.table(table(tst$PR0M0))

#2)With tgv I will analyse variables relationships, first taking a look on the structure
str(tgv)
## in Excel file, of each variable I wrote down their variable type
## and which graphs, test you can run to see relation with aim variable


#3)Gerando os gr?ficos e testes das vari?veis Quantitativas vs Quali (Alvo = Promo = Vari?vel Quali)

##PROMO x PROD_A :
boxplot(tgv$PROD_A~tgv$PR0M0,
        data=tgv,
        main="Rela??o Promo vs Prod_A",
        xlab="Retorno ao E-Mail (0 = N?o retornou  1 = Retornou)",
        ylab="Compras do produto A nos ?ltimos 12 meses")

t.test(tgv$PROD_A~tgv$PR0M0)


##PROMO x PROD_B:
boxplot(tgv$PROD_B~tgv$PR0M0,
        data=tgv,
        main="Rela??o Promo vs Prod_B",
        xlab="Retorno ao E-Mail (0 = N?o retornou  1 = Retornou)",
        ylab="Compras do produto B nos ?ltimos 12 meses")

t.test(tgv$PROD_B~tgv$PR0M0)


##PROMO x PROD_C:
boxplot(tgv$PROD_C~tgv$PR0M0,
        data=tgv,
        main="Rela??o Promo vs Prod_C",
        xlab="Retorno ao E-Mail (0 = N?o retornou  1 = Retornou)",
        ylab="Compras do produto C nos ?ltimos 12 meses")

t.test(tgv$PROD_C~tgv$PR0M0)


##PROMO x FUNC:
boxplot(tgv$FUNC~tgv$PR0M0,
        data=tgv,
        main="Rela??o Promo vs Funcion?rios",
        xlab="Retorno ao E-Mail (0 = N?o retornou  1 = Retornou)",
        ylab="N?mero de funcionarios")


t.test(tgv$FUNC~tgv$PR0M0)



#4)Gerando os gr?ficos e testes das vari?veis Qualitativas vs Qualitativas (Alvo = Promo = Vari?vel Quali)
##Para criar um barplot e fazer o teste CHI?, eu tinha que criar sempre um table onde somente tem as duas vari?veis que quero verificar!

##PROMO x REGIAO:
PR0 <- table(tgv$PR0M0, tgv$REGIAO)

barplot(PR0, 
        main = "Frequ?ncia de E-Mail respondido e n?o respondido em rela??o a regi?o",
        xlab = "Regi?o",
        ylab = "Frequ?ncia",
        col = c("black","white"), 
        beside = TRUE)

legend("topright", inset = .00001, c("E-Mail n?o respondido","E-Mail respondido"), fill = c("black","white"),
       cex = 0.70)

chisq.test(PR0)


##PROMO x SETOR:
PS0 <- table(tgv$PR0M0, tgv$SETOR)

barplot(PS0, 
        main = "Frequ?ncia de E-Mail respondido e n?o respondido em rela??o ao Setor",
        xlab = "Setor",
        ylab = "Frequ?ncia",
        col = c("black","white"),
        beside = TRUE)

legend("topleft", inset = .005, c("0 = E-Mail n?o respondido","1 = E-Mail respondido"), fill = c("black","white"),
       cex = 0.9)

chisq.test(PS0)


##PROMO x SEDE:
PSE0 <- table(tgv$PR0M0, tgv$SEDE)

barplot(PSE0, 
        main = "Frequ?ncia de E-Mail respondido e n?o respondido em rela??o a Sede",
        xlab = "Sede",
        ylab = "Frequ?ncia",
        col = c("black","white"),
        beside = TRUE)

legend("topleft", inset = .005, c("E-Mail n?o respondido","E-Mail respondido"), fill = c("black","white"),
       cex = 0.7)
legend(locator() , inset = .005, c("0 = N?o tem sede pr?pria","1 = Tem sede pr?pria"), cex =0.7)

chisq.test(PSE0)

##PROMO x TMPCLI:
PTMPCLI0 <- table(tgv$PR0M0, tgv$TMPCLI)

barplot(PTMPCLI0, 
        main = "Frequ?ncia de E-Mail respondido e n?o respondido em rela??o ao tempo do cliente",
        xlab = "Tempo de Cliente",
        ylab = "Frequ?ncia",
        col = c("black","white"),
        beside = TRUE)

legend("topleft", inset = .005, c("E-Mail n?o respondido","E-Mail respondido"), fill = c("black","white"),
       cex = 0.9)

chisq.test(PTMPCLI0)

##Excluir a coluna desnecess?ria Regi?o
tgv1 <- tgv[,-4]
View(tgv1)

###Rodar uma arvore de classifica??o
#no caso abaixo o alvo ? o cancel (por isso ele vem antes do ~)
ad<-rpart(data=tgv1, PR0M0~PROD_A+PROD_B+PROD_C+SETOR+TMPCLI+SEDE+FUNC)
#obs.: poderiamos usar o ~. para trazer todos os campos, mas ? mais acertivo descrever cada variavel que ser? utilizada
#poderia usar ad=rpart(data=lrn,cancel~.)
View(ad) 
ad

#Plotando a arvore
prp(ad, type = 2, extra = 1, nn=T, fallen.leaves = T, branch.col = "red", branch.lty = 5,
    box.col = "white")
printcp(ad)
#podar a ad pois o cross validation (xerror) sugere overfit
ad2<-prune(ad, cp=0.010929)
prp(ad2, type = 2, extra = 1, nn=T, fallen.leaves = T, branch.col = "red", branch.lty = 5,
    box.col = "white")
printcp(ad2)

#previsao 
prob=predict(ad2, newdata = tst, type = "prob")
head(prob)
p_nao=prob[,1]
tst$p_nao=prob[,1]
tst$p_sim=prob[,2]

# #classificar usando como ponto de corte .5
tst$klas=ifelse(tst$p_sim>.5,"yes","no")
#cruza variavel original com a classificada
m<-table(tst$cancel,tst$klas)
m
View(lrn)


#vamos rodar a reg log 
fit=glm(data=lrn,PR0M0~PROD_A+PROD_B+PROD_C+SETOR+TMPCLI+SEDE+FUNC, family = binomial())

summary(fit) 
print(fit,digits=3) 

#vamos selecionar as variaveis 
fit2=step(fit) 
fit2 


#A TELEGV prefere "paparicar" o cliente errado do que n?o "pararicar" ent?o vamos diminuir o ponto de corte para .4
# tst$klas=ifelse(tst$p_sim>.4,"yes","no")
# m<-table(tst$cancel,tst$klas)
# m
# #A TELEGV prefere "paparicar" o cliente errado do que n?o "pararicar" ent?o vamos diminuir o ponto de corte para .3 (se houver 30% de chance de ele cancelar, ser? "paparicado")
# tst$klas=ifelse(tst$p_sim>.3,"yes","no")
# m<-table(tst$cancel,tst$klas)
# m
# #para melhorar a analise, pode-se combinar variaveis (tomar cuidado ao criar outras novas)

CrossTable(Kprob,tst$PR0M0,prop.c = FALSE,prop.t = FALSE,prop.chisq = FALSE)


#Obs:::  Crosfit serve para se encontrar a taxa de erro.

install.packages("hmeasure")
library(hmeasure)

HMeasure(tst$PR0M0,tst$psim)$metrics


