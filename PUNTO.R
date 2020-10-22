pp=PUNTO

View(PUNTO)
# vamos analisar as vars
pp$RESID=as.factor(pp$RESID)
pp$SEXO=as.factor(pp$SEXO)

m=table(pp$RESID,pp$STATUS)
mp=prop.table(m,1)
m;mp

m=table(pp$SEXO,pp$STATUS)
mp=prop.table(m,1)
m;mp

#vamos analisar RENDA e discretizá-la para analisar a "linearidade"
library(arules)
pp$krenda=discretize(pp$RENDA, method = "frequency", categories = 5)
m=table(pp$krenda,pp$STATUS)
mp=prop.table(m,1)
m;mp
plot(pp$krenda)

m=table(pp$ESCOL,pp$STATUS)
mp=prop.table(m,1)
m;mp

# vamos separa as amostras

#vamos separar as amostras
pp$alvo=ifelse(pp$STATUS=="turismo",1,0)
set.seed(1934)
flag=sample(1:1000, 500, replace = F)
ppl=pp[flag,]
ppt=pp[-flag,]

#agora vamos rodar a reg log

fit=glm(data = ppl, alvo~RENDA+ESCOL+RESID+SEXO, family = binomial())
summary(fit)

#selecionar vars
fit2=step(fit)
summary(fit2)

#---------------------------------------------------------------------
#--------------------------------------------------------------------
#vou analisar o modelo utilizando a amostra teste
#vamos calcular p(turismo) para cada individuo da amostra teste --> ptur

ppt$ptur= predict(fit2, newdata=ppt, type="response")

#verificar o ajuste do modelo aos dados (modelo calibrado)
#teste de zeuyeuv
kptur=discretize(ppt$ptur, method = "frequency", categories = 5)
m=table(kptur,ppt$STATUS)
mp=prop.table(m,1)
mp

#calcular a taxa de erro
ppt$klas=ifelse(ppt$ptur>.7, "tur_hat", "out_hat")
m=table(ppt$klas,ppt$STATUS)
m

library(hmeasure)
HMeasure(ppt$STATUS,ppt$ptur)$metrics

install.packages("pROC")
library(pROC)
xx=roc(ppt$STATUS,ppt$ptur)
plot(xx)
xx










