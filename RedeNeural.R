ww<-TEBA[,-c(1,12:15)]
View(ww)

#na.omit() comando para remover missing value

#agora vamos gerar as variaveis dummies
wwd <- model.matrix(data=ww,~.)
View(wwd)
colnames(wwd)

#descartar primeira coluna
wwd<- wwd[,-1]
wwd <- as.data.frame(wwd)

#padronizar os dados entre 0 e 1
min <- apply(wwd,2,min)
max <- apply(wwd,2,max)

#agora vamos padronizar os dados
wws <- scale(wwd,center=min,scale= max-min)
View(wws)

wws = as.data.frame(wws)
head(wws)

set.seed(1234)

ind <- sample(1:2000,1200)

lrn <- wws[ind,]
tst <- wws[-ind,]

lixo <- paste(colnames(lrn), collapse="+")
lixo

install.packages("neuralnet")
library(neuralnet)

set.seed(911)

rn <- neuralnet(data=lrn,hidden=(3,4,5),cancelsim~idade+linhas+temp_cli+renda+fatura+temp_rsd+localB+localC+localD+tvcabosim+debautsim,
                lifesign = "minimal",linear.output = FALSE, rep=1)



#calcular as saidas da rede neural
pr <- compute(rn,tst[,-12]) #---> isto é uma lista

View(tst)

#para indentificar as saidas propriamente ditas:  
output = pr$net.result

head(output)

arq_saida <- cbind(tst,output)# ---> gerando uma unica matriz com as colunas  de tst + output

#vamos analisar os resultados
#matriz de classificação utilizando 0.5 como corte
klass= ifelse(output> .5, "yes","no")

table(tst$cancelsim,klass)

?neuralnet

set.seed(911)

install.packages("hmeasure")
library(hmeasure)

HMeasure(tst$cancelsim,output)$metrics

plot(rn)
 

#vamos estimar as probabiliddes de cancelsim
install.packages("arules")
library(arules)
kk<-discretize(output,method="frequency",breaks=8)

library(gmodels)

CrossTable(kk,tst$cancelsim,prop.c = FALSE, prop.t = FALSE, prop.chisq = FALSE)
