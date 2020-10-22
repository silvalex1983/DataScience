rm(list=ls())
install.packages("quantmod")
install.packages("ggplot2")
library(quantmod)
library(ggplot2)


papeis <- c('ABEV3.SA', 'AZUL4.SA', 'B3SA3.SA', 'BBAS3.SA', 'BBDC3.SA', 'BBSE3.SA')

pbr <- getSymbols(papeis, src = "yahoo", from = "2019-01-01", to = "2019-06-01")
head(pbr)
print(pbr)

P <- NULL
for(papel in papeis) {
  # help(Cl) # A função Cl serve para extrair e transformar colunas de series 
  # temporais de objetos OHLC (Open-High-Low-Close), que é como as informações
  # financeiras são geralmente disponibilizadas.
  # cl é a função do pacote quantmod para ficar só com o valor de fechamento 
  # (close).
  # help(eval) # A função eval resolve o seu argumento principal dentro de um
  # ambiente.
  #--------------------------------------------------------------------------
  #---------------------------- Exemplos de eval ----------------------------
  
  #eval(2 ^ 2 ^ 3)
  #mEx <- expression(2^2^3); mEx; 1 + eval(mEx)
  #eval({ xx <- pi; xx^2}) ; xx
  
  #--------------------------------------------------------------------------
  # help(parse) # A função parse pede para analisar uma palavra (parse=analisar).
  tmp <- Cl(to.monthly(eval(parse(text = papel))))
  P <- cbind(P, tmp)
}
head(P)

colnames(P) <- papeis

head(P)

retornos <- (diff(P)/lag(P,k=1)) * 100
retornos <- retornos[-1,]



media_retornos <- colMeans(retornos, na.rm = TRUE)
cov_retornos <- cov(retornos)


require(lattice)
levelplot(cov_retornos)


cor_retornos <-cor(retornos)
require(lattice)
levelplot(cor_retornos)


# Risco X Retorno de papéis isolados.
desvpad_retornos <- sqrt(diag(cov_retornos))
risco_retorno <- cbind(media_retornos,desvpad_retornos)
risco_retorno_df <- data.frame(risco_retorno)
ativos <- rownames(risco_retorno_df)
risco_retorno.condicao <- data.frame(matrix("Papéis isolados", nrow = nrow(risco_retorno)))
risco_retorno_df <- cbind(risco_retorno.condicao, ativos, risco_retorno_df)
rownames(risco_retorno_df) <- NULL
names(risco_retorno_df)[1]<-paste("condicao")
names(risco_retorno_df)[2]<-paste("carteira")
names(risco_retorno_df)[3]<-paste("retorno_carteira")
names(risco_retorno_df)[4]<-paste("desvpad_carteira")
head(risco_retorno_df)
#sapply(risco_retorno_df, class)
library("ggplot2")
ggplot(data = risco_retorno_df, aes(x = desvpad_carteira, y = retorno_carteira)) + 
  geom_point(data = risco_retorno_df, colour = "blue") + 
  geom_text(data = risco_retorno_df,aes(label=carteira),hjust=0, vjust=0, colour = "blue") + 
  ggtitle("Risco X Retorno dos papéis considerados") + 
  labs(x = "Risco (Desvio-padrão)", y = "Retorno Esperado")



# Simulando carteiras aleatórias.
simulacoes <- matrix(0,ncol = 3 + ncol(retornos))
dimnames(simulacoes) <- list(NULL, c("cart.simulada",papeis,"retorno_carteira","desvpad_carteira"))
cart.simulada <- 0
for (i in 1:1000) {
  cart.simulada <<- cart.simulada + 1
  w <- c(runif(ncol(retornos)))
  if (sum(w) == 0) { w <- w + 1e-2 }
  w <- w / sum(w)
  retorno_carteira <- w %*% media_retornos
  desvpad_carteira <- sqrt(w %*% cov_retornos %*% w)
  w <- matrix(w,nrow = 1)
  linhanova <- cbind(cart.simulada, w, retorno_carteira, desvpad_carteira)
  simulacoes <<- rbind(simulacoes, linhanova)
}
#head(simulacoes)
simulacoes <- simulacoes[-1,]
head(simulacoes)
simulacoes.carteira <- simulacoes[,1]
simulacoes.metricas <- simulacoes[, (ncol(simulacoes)-1):ncol(simulacoes)]
risco_retorno.sims <-cbind(simulacoes.carteira,simulacoes.metricas)
Simulacoes.condicao <- data.frame(matrix("Carteiras simuladas", nrow = nrow(simulacoes)))
risco_retorno.sims_df <- data.frame(risco_retorno.sims)
risco_retorno.sims_df <- cbind(Simulacoes.condicao, risco_retorno.sims_df)
names(risco_retorno.sims_df)[1]<-paste("condicao")
names(risco_retorno.sims_df)[2]<-paste("carteira")
#head(risco_retorno.sims_df)
#sapply(risco_retorno.sims_df, class)
risco_retorno.sims_df$carteira <- as.factor(risco_retorno.sims_df$carteira)
#sapply(risco_retorno.sims_df, class)
head(risco_retorno.sims_df)
ggplot(data = risco_retorno.sims_df, aes(x = desvpad_carteira, y = retorno_carteira)) + 
  geom_point(data = risco_retorno.sims_df, colour = "red") + 
  geom_text(data = risco_retorno.sims_df,aes(label=simulacoes.carteira),hjust=0, vjust=0, colour = "red") +
  ggtitle("Risco X Retorno das carteiras simuladas") + 
  labs(x = "Risco (Desvio-padrão)", y = "Retorno Esperado")



# Observação1: o sinal de <<- é diferente de <-. <- atua somente sobre o ambiente
# em que é definido. Já <<- atua também atualizando ambientes "pais".
# Observação2: O sinal de = é diferente de ==. = é sinal de atribuição, usado dentro
# de algumas funções como argumento. Já == é um operador lógico que verifica a igual-
# dade e retorna TRUE ou FALSE.
# Observação3: Se você quiser ver como funciona a função objetivo, usar 
# w <- rep(0.2, ncol(retornos)).
funcao_obj <- function(w) {
  fn.call <<- fn.call + 1
  if (sum(w) == 0) { w <- w + 1e-2 }
  w <- w / sum(w)
  retorno_carteira <- w %*% media_retornos
  desvpad_carteira <- sqrt(w %*% cov_retornos %*% w)
  obj <- desvpad_carteira-retorno_carteira
  return(obj)
}

 #install.packages("GenSA")
library(GenSA)
# help(GenSA)
# ls('package:GenSA')



# help(GenSA)
set.seed(1234)
fn.call <<- 0
tempo_inicial <- Sys.time()
resultado_GenSA <- GenSA(fn = funcao_obj, lower = rep(0, ncol(retornos)), upper = rep(1, ncol(retornos)), control = list(smooth = FALSE, max.call = 100000))
tempo_final <- Sys.time()
tempo_execucao <- tempo_final - tempo_inicial
tempo_execucao



fn.call_GenSA <- fn.call
resultado_GenSA$counts
cat("GenSA chamou a função objetivo", fn.call_GenSA, "vezes.\n")



resultado_GenSA$value


w.otimo_GenSA <- resultado_GenSA$par
w.otimo_GenSA <- w.otimo_GenSA / sum(w.otimo_GenSA)
w.otimo_GenSA_ <- cbind(papeis, w.otimo = round(100 * w.otimo_GenSA, 2))
w.otimo_GenSA_

retorno_carteira_otima <- sum(w.otimo_GenSA * media_retornos)
retorno_carteira_otima

desvpad_carteira_otima <- sqrt(w.otimo_GenSA %*% cov_retornos %*% w.otimo_GenSA)
desvpad_carteira_otima

risco_retorno_otimo_df <- data.frame(matrix(c(retorno_carteira_otima,desvpad_carteira_otima),1,2))
risco_retorno_otimo_df <- cbind("Carteira ótima","*",risco_retorno_otimo_df)
names(risco_retorno_otimo_df)[1]<-paste("condicao")
names(risco_retorno_otimo_df)[2]<-paste("carteira")
names(risco_retorno_otimo_df)[3]<-paste("retorno_carteira")
names(risco_retorno_otimo_df)[4]<-paste("desvpad_carteira")
#sapply(risco_retorno_otimo_df, class)
risco_retorno_total_df <- rbind(risco_retorno_df,risco_retorno.sims_df,risco_retorno_otimo_df)
risco_retorno_total_df

ggplot(data = risco_retorno_total_df, aes(x = desvpad_carteira, y = retorno_carteira, colour = condicao)) + 
  geom_point(data = risco_retorno_total_df) +
  ggtitle("Risco X Retorno das ações e carteiras") + 
  labs(x = "Risco (Desvio-padrão)", y = "Retorno Esperado")



funcao_obj2 <- function(w) {
  fn.call <<- fn.call + 1
  if (sum(w) == 0) { w <- w + 1e-2 }
  w <- w / sum(w)
  retorno_carteira <- w %*% media_retornos
  desvpad_carteira <- sqrt(w %*% cov_retornos %*% w)
  w_matriz <- matrix(w,1,ncol(retornos))
  GenSA_linha <- cbind(fn.call, w_matriz, retorno_carteira, desvpad_carteira)
  GenSA_passoapasso <<- rbind(GenSA_passoapasso,GenSA_linha)
  obj <- desvpad_carteira-retorno_carteira
  return(obj)
}

set.seed(1234)
fn.call <- 0
w_matriz <- matrix(rep(0.0, ncol(retornos)),1,ncol(retornos))
retorno_carteira <- 0
desvpad_carteira <- 0
GenSA_passoapasso <- cbind(fn.call, w_matriz, retorno_carteira, desvpad_carteira)
resultado_GenSA <- GenSA(fn = funcao_obj2, lower = rep(0, ncol(retornos)), upper = rep(1, ncol(retornos)),
                         control = list(smooth = FALSE, max.call = 10000))

#head(GenSA_passoapasso)
GenSA_passoapasso <- GenSA_passoapasso[-1,]
head(GenSA_passoapasso)

head(risco_retorno_total_df)



carteira_GenSA_pap <- matrix(GenSA_passoapasso[,1],nrow(GenSA_passoapasso),1)
#head(carteira_GenSA_pap)


retorno_carteira_GenSA_pap <- matrix(GenSA_passoapasso[,(2+ncol(w_matriz))],nrow(GenSA_passoapasso),1)
head(retorno_carteira_GenSA_pap)

desvpad_carteira_GenSA_pap <- matrix(GenSA_passoapasso[,(3+ncol(w_matriz))],nrow(GenSA_passoapasso),1)
head(desvpad_carteira_GenSA_pap)


risco_retorno_GenSA_pap_df <- data.frame(cbind(carteira_GenSA_pap,retorno_carteira_GenSA_pap,desvpad_carteira_GenSA_pap))
#head(risco_retorno_GenSA_pap_df)
risco_retorno_GenSA_pap_df <- cbind("Passos do GenSA",risco_retorno_GenSA_pap_df)
#head(risco_retorno_GenSA_pap_df)
names(risco_retorno_GenSA_pap_df)[1]<-paste("condicao")
names(risco_retorno_GenSA_pap_df)[2]<-paste("carteira")
names(risco_retorno_GenSA_pap_df)[3]<-paste("retorno_carteira")
names(risco_retorno_GenSA_pap_df)[4]<-paste("desvpad_carteira")
#sapply(risco_retorno_GenSA_pap_df, class)
risco_retorno_GenSA_pap_df$carteira <- as.factor(risco_retorno_GenSA_pap_df$carteira)
#sapply(risco_retorno_GenSA_pap_df, class)
risco_retorno_total_df_2 <- rbind(risco_retorno_total_df,risco_retorno_GenSA_pap_df)
head(risco_retorno_total_df_2)

ggplot(data = risco_retorno_total_df_2, aes(x = desvpad_carteira, y = retorno_carteira, colour = condicao)) + 
  geom_point(data = risco_retorno_total_df_2) +
  ggtitle("Risco X Retorno das ações e carteiras") + 
  labs(x = "Risco (Desvio-padrão)", y = "Retorno Esperado")



