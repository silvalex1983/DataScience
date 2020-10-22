# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 12:35:20 2019
   Remarks: "\" can be used as a continuation line character
@author: gusta
"""
#%%
#########################################################
# Preparação do ambiente                                #
#########################################################
# Esta célula deve ser executada sempre que o kernel for#
#  reinicializado. Ela irá preparar o ambiente para uso #
#  carregando variáveis globais e bibliotecas           #
#  apropriadas. Isto facilitará a execução dos demais   #
#  códigos durante o desenvolvimento.                   #
#########################################################

#########################################################
# Dicionário de parâmetros (variáveis globais) (início) #
#########################################################
parametros = { 
                #########################################
                # Usar um número de epocas de treino    # 
                #  padronizado para facilitar o ajuste  #
                #  entre a geração  dos dados de treino #
                #  e os gráficos animados               #
                #########################################
                'epocas' : 50,

                #########################################
                # Espessura é o quanto os eixos x e y   #
                #  serão maiores que os valores mínimo e#
                #  máximo dos seus respectivos datasets #                                             #
                #########################################
                'espessura' : 0.5,

                #########################################
                # Usar um número padronizado de quadros #
                #  para as funções gráficas com animação#
                #########################################
                'quadros':50,

                #########################################
                # Utilizar a mesma seed para todos os   #
                #  estados de número aleatório ao longo #
                #  do script. A mistura de uma chamada  #
                #  np.random.seed(1234) com um          # 
                #  random_state=42 produziu diferenças  #
                #  entre os dados de treino e os valores#
                #  que foram apresentados nos gráficos  #
                #########################################
                'semente':42,

                #########################################
                # Taxa de aprendizado default para os   #
                #  modelos de machine learning          #
                #########################################
                'taxa_aprend':0.1}

#########################################################
# Carregamento das bibliotecas de setup e debug         #
#########################################################
import os; os.getcwd()
import sys; sys.path
import copy; a = 1; b = copy.deepcopy(a); del(a,b)

#########################################################
# Carregamento das bibliotecas de cálculo               #
#########################################################
import numpy as np
np.set_printoptions(precision=3)
np.random.seed(parametros['semente'])

#########################################################
# Carregamento das bibliotecas gráficas                 #
#########################################################
from matplotlib import pyplot as plt
import pandas as pd 
import seaborn as sns

#########################################################
# Carregamento e setup das bibliotecas de animação graf.#
#########################################################
from matplotlib.animation import FuncAnimation
plt.rcParams["figure.facecolor"] = "w"
plt.rcParams["animation.html"] = "html5" #ou "jshtml"

#########################################################
# Carregamento dos pacotes de dados e funções de M.L.   #
#########################################################
from sklearn.datasets import make_circles, make_moons, \
                             load_iris
from sklearn.model_selection import train_test_split
from tensorflow import keras

#%%
#########################################################
# 1a aula: Rede Neural em Keras                         #
#########################################################

#########################################################
# Dados e separação treino / teste                      #
#########################################################
x,y = make_circles(n_samples=2000, noise=0.1,
                   random_state=parametros['semente'], 
                   factor=0.3)

x_treino, x_teste, y_treino, y_teste = \
      train_test_split(x, y, test_size=0.2)

#%%
#########################################################
# Para simplificar o desenvolvimento iremos trabalhar   #
#  apenas com os arrays "x" e "y" completos. Devemos    #
#  lembrar no entanto que em uma situação real, devemos #
#  dividir o conjunto de dados em treino e teste, tal   #
#  como mostrado acima, usando por exemplo a função     #
#  train_test_split do scikit-learn.                    #
#########################################################
del(x_treino, x_teste, y_treino, y_teste)

#%%
#########################################################
# Visualização do conjunto de dados                     #
#########################################################
dados = np.column_stack([x, y])
dados = pd.DataFrame(data=dados,
                      columns=['x1','x2','y'])
plt.close()
sns.scatterplot(data=dados, x='x1', y='x2',
                hue='y')
plt.show()

#%%
#########################################################
# Também para simplificar o desenvolvimento vamos criar #
#  uma função que irá receber os arrays numpy x e y e   #
#  irá criar um dataframe pandas com colunas x0, x1, x2 #
#  e y. A coluna x0 será a coluna de 1s dos modelos de  #
#  regressão linear. Iremos em seguida deletar os arrays#
#  x e y os quais serão recuperados a partir do pandas  #
#  dataframe "dados" conforme necessário. Como sempre o #
#  objetivo é manter o número de variáveis livres na    #
#  memória ao mínimo                                    #
#########################################################
def gera_dados(x,y):
    n = len(x)
    x0 = np.ones(n).reshape(-1,1)
    matriz = np.column_stack([x0,x,y.reshape(-1,1)])
    dados = pd.DataFrame(data=matriz,
                         columns=['x0','x1','x2','y'])
    return(dados)

dados = gera_dados(x,y)
del(x,y)

#%%
#########################################################
# Preparação do grid de desenho                         #
#########################################################
# Vamos criar uma função que irá receber um array X com #
#  variáveis x1 e x2 e irá devolver um grid_2d, nxm     #
#########################################################
def gera_grid(X=np.array(dados[['x1','x2']]),
              espessura=parametros['espessura'],
              n=100, m=100):
    x1_min = np.min(X[:,0]) - espessura    
    x1_max = np.max(X[:,0]) + espessura
    eixo_x1 = np.linspace(x1_min,x1_max,n)

    x2_min = np.min(X[:,1]) - espessura
    x2_max = np.max(X[:,1]) + espessura
    eixo_x2 = np.linspace(x2_min,x2_max,m)

    grid_2d = np.meshgrid(eixo_x1,eixo_x2)
    grid_2d = np.array(grid_2d)

    return(grid_2d)

grid_2d = gera_grid(np.array(dados[['x1','x2']]))

#%%
#########################################################
#array_entrada é composto da combinação de todos os va- #
# lores de x1 no intervalo x1_min a x1_max, e x2 no in- #
# tervalo x2_min a x2_max.                              #
#########################################################
array_entrada = grid_2d.reshape(2,-1).T

#%%
#########################################################
# Criação de um modelo através do Keras                 #
#########################################################
model = keras.models.Sequential()
model.add(keras.layers.Dense(4, input_dim=2,activation='relu'))
model.add(keras.layers.Dense(6, activation='relu'))
model.add(keras.layers.Dense(6, activation='relu'))
model.add(keras.layers.Dense(4, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

#%%
#########################################################
# Preparação para execução (compilação)                 #
#########################################################
model.compile(loss='binary_crossentropy', 
              optimizer='adamax', 
              metrics=['accuracy'])

#%%
###########################################################
# Array numpy com os arrays de histórico de probabilidade #
###########################################################
dados_probs = []

#########################################################
# Função para salvamento do Histórico do Treino         #
#########################################################
def salva_epoca(epoca, logs, 
                array_entrada = array_entrada,
                dados=dados, 
                dados_probs = dados_probs):
    probs_previstas = \
    model.predict_proba(array_entrada, 
                        batch_size=32, 
                        verbose=0)
    #########################################################
    # O array com o histórico de probabilidades para os     #
    #  valores de y a cada época do modelo é armazenado em  #
    #  uma lista específica com 10.000 linhas no caso padrão#
    #########################################################
    dados_probs.append(probs_previstas)
    return(dados_probs)

#%%
#########################################################
# Chamada pela keras da função de salvamento do Treino  #
#  Será executada ao final de cada época a função       #
#  salva_epoca. Olhando acima a definição de salva_epoca#
#  vemos que ela irá receber o número da epoca e os logs#
#  de status do treino por default. Esta função então   #
#  irá calcular as probabilidades dos pontos no array   #
#  numpy array_entrada pertencerem ao conjunto com y=1 e#
#  em seguida irá salvar este bloco de valores na posi- #
#  ção final do array numpy de nome dados_probs.        #
#########################################################
historico = \
 keras.callbacks.LambdaCallback(on_epoch_end=salva_epoca)

#%%
#########################################################
# Treino do modelo via keras                            #
######################################################### 
x = np.array(dados[['x1','x2']])
y = np.array(dados['y'])
historia = model.fit(x, y, epochs=parametros['epocas'], 
                     verbose=0, 
                     callbacks=[historico])
del(x, y)

#%%
#########################################################
# Visualização dinâmica do treino                       #
#########################################################
plt.close()
fig, ax = plt.subplots()

def animate(i, ax = ax, dados = dados, 
            dados_probs = dados_probs, 
            n=100, m=100, 
            espessura=parametros['espessura']):
    
    x1_min = dados['x1'].min() - espessura
    x1_max = dados['x1'].max() + espessura
    eixo_x1 = np.linspace(x1_min, x1_max, n)
    
    x2_min = dados['x2'].min() - espessura
    x2_max = dados['x2'].max() + espessura
    eixo_x2 = np.linspace(x2_min, x2_max, n)
    
    plt.clf()
    ax = sns.scatterplot(data=dados, x='x1', y='x2',
                hue='y', palette=['red','blue'],
                legend=False)
    
    array = np.array(dados_probs[i]).reshape(100,100)
    ax = plt.contourf(eixo_x1, eixo_x2, array, 
                      alpha=0.5, cmap=plt.cm.Spectral)
    fig.suptitle("Aprendizado do Modelo Keras. Época = " +
                 str(i+1))
    return(ax)

animation = FuncAnimation(fig, animate, 
                          range(parametros['epocas']),
                          repeat = True, interval = 1000)

animation

#%%
plt.close()
del(grid_2d, array_entrada, dados_probs)
del(historico, historia)
del(fig, ax, animation, animate)

#%%
#########################################################
# 2a aula: Convergência de Um Perceptron                #
#########################################################

#%%
#########################################################
# Geração dos Dados em formato de meia-lua              #
#########################################################
x, y = make_moons(n_samples = 2000, noise=0.15, 
                  random_state=parametros['semente'])

dados = gera_dados(x,y)
del(x,y)

plt.close()
sns.scatterplot(data=dados, x='x1', y='x2', 
                hue='y', palette=['red','blue'])

#%%
#########################################################
# Inversão para posicionar a classe "1" na parte de     #
#  cima do gráfico                                      #
#########################################################
dados['y_inv'] = np.logical_not(dados['y'])*1
plt.close()
sns.scatterplot(data=dados, x='x1', y='x2', 
                hue='y_inv', palette=['red','blue'])

#%%
#########################################################
# Deslocamento de x2 para cima e x1 para esquerda para  #
#  permitir a separação linear dos grupos               #
#########################################################
dados['x1_desloc'] = dados['x1'] -0.25 * dados['y_inv']
dados['x2_desloc'] = dados['x2'] +1.60 * dados['y_inv']

plt.close()
sns.scatterplot(data=dados, x='x1_desloc', y='x2_desloc', 
                hue='y_inv', palette=['red','blue'])

#%%
#############################################################
# Giro de 45o para criar uma reta inclinada de separação    #
#############################################################
def giro(vetores,graus=45):
    rads = graus*np.pi/180
    Mgiro = np.array([[np.cos(rads),np.sin(rads)],
                     [-np.sin(rads),np.cos(rads)]])
    vetores_novos = np.matmul(Mgiro,vetores.T)
    return(vetores_novos)

vetores = np.array(dados[['x1_desloc','x2_desloc']]) 
vetores = giro(vetores).T
dados['x1_giro'] = vetores[:,0]
dados['x2_giro'] = vetores[:,1]
del(vetores)

plt.close()
sns.scatterplot(data=dados, x='x1_giro', y='x2_giro', 
                hue='y_inv', palette=['red','blue'])

#%%
#########################################################
# 2a aula, 2a parte: Convergência de Um Perceptron      #
#########################################################

#########################################################
# Função para apresentação do gráfico dos dados de      #
#  treino e da linha de separação a partir do vetor wT  #
#  via seaborn                                          #
#########################################################
def separacao(dados, wT, cor, titulo):
    plt.close()
    #Linha de separação
    a = wT[0,0]
    b = wT[0,1]
    c = wT[0,2]
    sns.lineplot(data=dados, x=dados.iloc[:,0], 
                 y=-(a+b*dados.iloc[:,0])/c, 
                 color=cor).set_title(titulo)
                                 
    sns.scatterplot(data=dados, x=dados.iloc[:,0], 
                    y=dados.iloc[:,1], 
                    hue=dados.iloc[:,2], 
                    palette=('red','blue'), legend=False)
    plt.show()

#%%
#########################################################
# Execução do gráfico                                   #
#########################################################
wT_target = np.array([[-3.2,2,3]])
separacao(dados[['x1_giro','x2_giro','y_inv']], 
          wT_target, 'black','Target')
del(wT_target)


#%%
#########################################################
# Função de Treino do Perceptron                        #
#########################################################
def treina_perceptron(dados, taxa_aprend):
    epoca = 0
    precisao = 0

    ###########################################################
    # Inicialização de wT. Trabalharemos com matrizes X e Y   #
    #  as quais serão geradas a partir dos dados armazenados  #
    #  no pandas dataframe "dados".                           #
    ###########################################################
    wT = np.random.randn(1,3)
    wT_historico = []
    wT_historico.append(wT)

    X = np.array(dados.iloc[:,0:3])
    Y = np.array(dados.iloc[:,3]).reshape(-1,1)
    n = len(dados)
    while ((precisao < 1) and (epoca < 1000)):
        for i in range(n):
            teste = np.matmul(wT,X.T[:,i])
            if (teste>0) and (Y.T[0,i]==0):
                wT = wT - taxa_aprend*X.T[:,i]
            elif (teste<0) and (Y.T[0,i]==1):
                wT = wT + taxa_aprend*X.T[:,i]
            else:
                pass
            wT_historico.append(wT)
        
        Y_hat = (np.matmul(wT,X.T)>0)*1
        precisao = np.sum(Y_hat==Y.T)/n
        epoca = epoca + 1

    wT_historico = np.array(wT_historico)
    return(wT_historico)

#%%
#########################################################
# Treino do Perceptron                                  #
#########################################################
matriz = dados[['x0','x1_giro','x2_giro','y_inv']]
    
wT_historico = treina_perceptron(matriz, 
                                 parametros['taxa_aprend'])

del(matriz)

#%%
#########################################################
# Linha de Separação Inicial                            #
#########################################################
wT_inicial = wT_historico[0]

separacao(dados[['x1_giro','x2_giro','y_inv']], 
          wT_inicial, 'black','Target')

del(wT_inicial)

#%%
#########################################################
# Linha de Separação Final                              #
#########################################################
wT_final = wT_historico[len(wT_historico)-1]

separacao(dados[['x1_giro','x2_giro','y_inv']], 
          wT_final, 'yellow','Final')

del(wT_final)

#%%
#########################################################
# Evolução da Linha de Separação ao longo das Épocas    #
#########################################################
plt.close()
fig, ax = plt.subplots()
ax.set_xlim(dados['x1_giro'].min()-0.25,
            dados['x1_giro'].max()+0.25)
ax.set_ylim(dados['x2_giro'].min()-0.25,
            dados['x2_giro'].max()+0.25)

def faz_linha(k, wT_historico = wT_historico,
              dados = dados):
    n = len(wT_historico)-1
    
    plt.clf()
    ax = sns.scatterplot(data=dados, x='x1_giro', 
                         y='x2_giro', hue='y', 
                         palette=('red','blue'),
                         legend=False)
    i = int((k+1)*n/50)
    a = wT_historico[i,0,0]
    b = wT_historico[i,0,1]
    c = wT_historico[i,0,2]
    ax = sns.lineplot(data=dados, x='x1_giro', 
                      y=-(a + b * dados['x1_giro']) / c,
                      color = 'black')
    ax.set_title('Convergência do Perceptron - i = ' +
                 str(i) + " de " + str(n))
    return(ax)

animacao = FuncAnimation(fig, faz_linha, 
                         parametros['quadros'], 
                         repeat = False,
                         interval = 400)
animacao
plt.show()

#%%
#########################################################
# Observação final: dado que wT é atualizado a cada     #
#  iteração com X[i], esta lógica de ajuste não pode    #
#  ser implementada através de uma operação matricial   #
#  única. Devemos obrigatoriamente percorrer a matriz   #
#  vetor por vetor e ir ajustando wT.                   #
#########################################################
del(wT_historico)

