set.seed(NULL)

TEBA <- TEBA

View(TEBA)

zz<-TEBA[,-c(1,12:15)]

View(zz)

#reg logistica não pode ter alvo qualitativo

zz$alvo <- ifelse(zz$cancel=="nao",1,0)
View(zz)


col_list<- paste(colnames(zz),collapse = "+")
col_list

#flag = sample(1:2000,1200)? usa?
flag = sample(1:2000,1200)

#vamos dividir o arquivo em 2 partes.
lrn<-zz[flag,]
tst<-zz[-flag,]

#vamos rodar a reg log
fit=glm(data=lrn,alvo~idade+linhas+temp_cli+renda+fatura+temp_rsd+local+tvcabo+debaut, family = binomial())

summary(fit)

print(fit,digits=3)

#vamos selecionar as variaveis
fit2=step(fit)

fit2

#da saida vamos escrever (formula de z)
#z=-1.564295 +0.041068*idade+0.197745*temp_cli-0.002098*fatura-2.431947*localB+0.151507*localC-1.220053*localD

z

#juquinha= idade =20, tmp_cli =2, fatura=800, mora com B
z=-1.564+0.041*20+0.198*2-0.0021*800+2.432*1+0.152*0-1.220*0
z
z=-4.46

pjuquinhanaocancelar= 1/(1+exp(-z))
pjuquinhanaocancelar

#Calcular a probabilidade dos individuos na amostra tst

tst$pnao <- predict(fit2,newdata=tst,type="response")

tst$pnao
View(tst)

#vamos adotar como ponto de corte 

PC= 0.5

tst$klass= ifelse(tst$pnao>PC,"no","yes")

table(tst$cancel,tst$klass)

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

Kprob<- discretize(tst$pnao,method="frequency", breaks = 8)

install.packages("gmodels")
library(gmodels)

CrossTable(Kprob,tst$cancel,prop.c = FALSE,prop.t = FALSE,prop.chisq = FALSE)


#Obs:::  Crosfit serve para se encontrar a taxa de erro.

install.packages("hmeasure")
library(hmeasure)

HMeasure(tst$cancel,tst$pnao)$metrics



