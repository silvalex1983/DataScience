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
                #  máximo dos seus respectivos datasets #
                #########################################
                'espessura' : 0.5,
                
                #########################################
                # n_pontos é o número de pontos que     #
                #  serão utilizados para dividir os     #
                #  eixos coordenados quando estiver     #
                #  sendo produzido grid (meshgrid) de   #
                #  desenho para um gráfico de contorno  #
                #########################################
                'n_pontos' : 100,
                
                #########################################
                # Precisão é o número de casas decimais #
                #  utilizado nas impressões em geral    #
                #########################################
                'precisao' : 3,

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
from time import time

#########################################################
# Carregamento das bibliotecas de cálculo               #
#########################################################
import numpy as np
np.set_printoptions(precision=parametros['precisao'])
np.random.seed(parametros['semente'])

#########################################################
# Carregamento das bibliotecas gráficas                 #
#########################################################
from matplotlib import pyplot as plt
import pandas as pd 
import seaborn as sns
sns.set_style("darkgrid")
from matplotlib.colors import ListedColormap

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

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

#########################################################
# Pacote para importação de dados de outras plataformas #
#  R por exemplo                                        #
#########################################################
import statsmodels.api as sm

#########################################################
# Pacote com recursos de matemática avançados.          #
#  Utilizaremos o recurso math.inf (infinito)           #
#########################################################
import math

#########################################################
# Funções do usuário                                    #
#########################################################

#########################################################
# apaga_var recebe uma string ou uma lista de strings e #
#  verifica se a variável existe no escopo global do    #
#  script apagando a mesma da memória. Caso a string não#
#  exista na memória a função prossegue sem a geração de#
#  um erro de execução.                                 #
#########################################################
def apaga_var(vars):
    if type(vars) == str:
        if vars in globals():
            del globals()[vars]
    else:
        for x in vars:
            if x in globals():
                del globals()[x]
                    
#%%            
#########################################################
# 1a aula - Support Vector Machines via scikit-learn    #
#########################################################

#%%
#########################################################
# 1a etapa - Definição de objetivos: Qual é a 'pergunta #
# que desejamos responder'?                             #
#########################################################

#%%
#########################################################
# Desejamos ser capazes de prever a categoria de uma    #
#  variável denominada "species" a partir de até quatro #
#  fatores, a saber "petal_length", "petal_width",      #
#  "sepal_length", "sepal_width"                        #
#########################################################
                
#%%
#########################################################
# 2a etapa - Obtenção e ajuste dos dados                #
#########################################################

#%%
#########################################################
# Obtenção dos dados                                    #
#########################################################
                
#%%
#########################################################
# Vamos carregar novamente o dataset "iris" o qual      #
#  servirá de base para o próximo exercício. Para       #
#  referência futura, vamos importar o dataset a partir #
#  do R. Devemos conhecer o nome do dataset (neste caso #
#  "iris" e o pacote R que carrega o mesmo (neste caso o#
#  pacote chama-se "datasets".                          #
#########################################################
# A biblioteca python que permite tal funcionalidade foi#
#  carregada através do comando:                        #
# import statsmodels.api as sm                          #
#########################################################
dataset_iris = sm.datasets.get_rdataset(dataname='iris',
                                        package='datasets')

#%%
#########################################################
# Os conjuntos de dados importados do R são carregados  # 
#  normalmente como dicionários. Para determinar quais  #
#  são as chaves associadas ao mesmo, utilizamos o      #
#  método ".keys()" tal como pode ser visto a seguir:   #
#########################################################
print(dataset_iris.keys())

#%%
#########################################################
# As seguintes chaves podem ser acessadas para obter-se #
#  informações a respeito do pacote em si:              #
#########################################################

#%%
#########################################################
# Nome do pacote R a partir do qual o dataset foi       #
#  carregado, no caso "datasets"                        #
######################################################### 
print(dataset_iris['package'])

#%%
#########################################################
# Nome oficial do dataset carregado, no caso:           #
#  "Edgar Anderson's Iris Data"                         #
######################################################### 
print(dataset_iris['title']) 

#%%
#########################################################
# Boolean, que determina se o pacote foi carregado      #
#  diretamente da internet ou do cache em disco         #
#########################################################
print(dataset_iris['from_cache'])

#%%
#########################################################
# Descrição dos campos do pacote                        #
#########################################################
print(dataset_iris['__doc__'])

#%%
#########################################################
# Os dados em sí, encontram-se disponíveis na chave     #
#  'data', a qual retorna os mesmos em um formato de um #
#  pandas dataframe                                     #
#########################################################
print(dataset_iris['data'])

#%%
#########################################################
# Por conveniência vamos passar esta parte do objeto    #
#  dataset_iris para uma variável simplesmente de nome  #
#  iris. A variável iris será um pandas dataframe. Vamos#
#  recarregar os dados do modelo diretamente da internet#
#  e eliminar o dicionário dataset_iris                 #
#########################################################
apaga_var('dataset_iris')
iris = sm.datasets.get_rdataset(\
            dataname='iris', package='datasets')['data']

#%%
#########################################################
# Ajuste dos dados - nomes de colunas                   #
#########################################################

#%%
#########################################################
# Para determinar os nomes das colunas do dataframe     #
#  pandas "iris" utilizamos o comando "list" (no R      #
#  utilizaríamos o comando "names"). Atente para o fato #
#  que o comando "list" no Python é bem mais genérico   #
#  que o comando "names" no R. "list" é utilizado para  #
#  criar uma lista a partir de um grupo de valores.     #
#########################################################
print(list(iris))

#%%
#########################################################
# Observe que o nome das colunas no dataset iris        #
#  importado do R é diferente do iris carregado pela    #
#  seaborn. Vamos ajustar as chaves no dicionário de    #
#  nomes para compatibilizar as funções. Precisamos     #
#  baixar a caixa das letras e trocar o "." que separa  #
#  os nomes por um "_".                                 #
#########################################################

#%%
#########################################################
# Podemos fazer isso chave por chave, tal como a seguir #
# PORÉM isto deve ser feito pelo método equivalente ao  #
# "names" do R, o qual é ".columns". Observe que        #
# "iris.keys()" é uma função e como tal ela "devolve"   #
#  valores. Portanto não podemos alterar os nomes das   #
#  colunas do data frame pandas por meio dela. Para isso#
#  precisaremos atribuir um valor a iris.columns. Além  #
#  disso, iris.columns tem de ser alterada em bloco (   #
#  deve-se passar uma lista diretamente para a variável)#
#  Operações individuais de atribuição (iris.columns[0] #
#  por exemplo) produzirão um erro. Outra forma de      #
#  obter uma lista de valores dos nomes das colunas que #
#  possam ser alterados de forma individual é copiar o  #
#  o valor de iris.columns.values. PORÉM lembre-se que  #
#  não devemos alterar os nomes das colunas alterando   #
#  os valores em iris.columns.values pois isso fará     #
#  com que a atribuição de valores a partir das colunas #
#  do dataframe iris nos gráficos da seaborn deixe de   #
#  funcionar. Sendo assim primeiro criamos uma cópia por#
#  valor dos valores dos títulos das colunas e em       #
#  seguida manipulamos esta variável (para preservar    #
#  por enquanto o valor original no dataframe           #                          #
#########################################################
titulo = copy.deepcopy(iris.columns.values)

#%%
titulo[0] = 'sepal_length'
titulo[1] = 'sepal_width'
titulo[2] = 'petal_length'
titulo[3] = 'petal_width'
titulo[4] = 'species'

print("\n", "Nomes da lista = ", titulo)
print("\n", "Nomes das colunas = ", iris.columns)

#%%
apaga_var('titulo')

#%%
#########################################################
# Cabe aqui um "pequeno" parênteses a respeito da       #
#  passagem de parâmetros no python                     # 
#########################################################
# Explicação da passagem por designação no Python       #
#########################################################

#%%
#########################################################
# Criaremos aqui uma lista que estará no nível "global" #
#########################################################
lista = [1,2,3,4,5]

#%%
#########################################################
# Em seguida criamos uma função que irá receber um      #
#  argumento e irá alterar completamente o mesmo através#
#  da redefinição do seu valor (em inglês "rebinding")  #
#########################################################
def teste(valor):
    #####################################################
    #Neste caso redefinimos "valor" por completo dentro #
    # da função                                         #
    #####################################################
    valor = [-1,-2,-3,-4,-5]
    return(valor)

#%%
#########################################################
# Observe que a função retorna um valor mas o objeto    #
#  externo lista, não é alterado pela mudança dentro da #
#  função. Isto ocorre porque redefinimos o objeto      #
#  inteiro dentro da função, isto é o nome dele o qual é#
#  passado por valor                                    #
#########################################################
print("\n")
print("var.externa lista antes = ", lista)
print("retorno da função com alteração", \
      "do nome do objeto = ", teste(lista))
print("var.externa lista depois = ", lista)

#%%
#########################################################
# Agora vamos alterar os elementos do objeto recebido   #
#  como argumento dentro da função.                     #
#########################################################
def teste2(valor):
    #####################################################
    # Observe que a instrução a seguir altera cada um   #
    # dos elementos de valor                            #
    #####################################################
    valor[:] = range(-1,-6,-1)
    return(valor)

#########################################################
# Como o objeto é passado por referência alterações em  #
# elementos dele dentro da função são refletidas fora   #
# dela.                                                 #
#########################################################
print("\n")
print("var.externa lista antes = ", lista)
print("retorno da função com alteração", \
      "dos elementos do objeto =", teste2(lista))
print("var.externa lista depois = ", lista)

#########################################################
# Esta combinação de passagem do objeto em si por       #
#  referência e do nome dele por valor é que se chama   #
#  passagem por designação                              #
#########################################################

#%%
apaga_var('lista')

#%%
#########################################################
# Vamos agora criar uma função para executar as         #
#  operações anteriores (troca de "." por "_", troca de #
#  maiúscula por mínuscula) e uma operação extra (troca #
#  de " " por "_") de forma automática a partir do array#
#  com os nomes das colunas.                            #
#########################################################

#%%
#########################################################
# A transformação dos caracteres de uma string de caixa #
#  alta para caixa baixa é feita de forma automática com#
#  o método .lower()                                    #
#########################################################
nome = 'Gustavo Mirapalheta'
print("\n")
print("nome = ", nome)
print("nome.lower() = ", nome.lower())
print("\n") 

#%%
apaga_var('nome')

#%%
#########################################################
# A transformação de uma string numa lista de caracteres#
#  é também feita de forma automática pela função list  #
#########################################################
nome = 'Gustavo Mirapalheta'
print(list(nome))

#%%
apaga_var('nome')

#%%
#########################################################
# A troca de todos os caracteres em uma string por outro#
#  pode ser feita transformando a string em um numpy    #
#  array de caracteres, analisando o array caracter por #
#  caracter, efetuando a troca se for o caso e juntando #
#  a sequência ao final novamente                       #
#########################################################
lista = list('Gustavo.Correa.Mirapalheta')

while "." in lista:
    posicao = lista.index(".")
    lista[posicao] = "_"

print("\n", "".join(lista))

#%%
apaga_var(['lista','posicao'])

#%%
#########################################################
# Um loop for com uma lista de elementos a serem        #
#  substituidos mais o construto anterior permite trocar#
#  uma série de elementos por um "_" por exemplo        #
#########################################################
elemento = "Gustavo.Correa.Mirapalheta Nascido_em:RS-1968"
lista = list(elemento)
a_trocar = [" ", ".", "-", ";",":"]

for ele1 in a_trocar:
    while ele1 in lista:
        posicao = lista.index(ele1)
        lista[posicao] = "_"

elemento = "".join(lista)
print(elemento)

#%%
apaga_var(['elemento', 'lista', 'a_trocar', 'ele1', 
           'posicao'])

#%%
#########################################################
# E se juntarmos o operador "enumerate" podemos trocar  #
# os elementos de um conjunto pelos elementos de mesma  #
# posicao em outro conjunto                             #
#########################################################
elemento = "Gustavo.Correa.Mirapalheta Nascido_em:RS-1968"
lista = list(elemento)
a_trocar   = [" ", ".", "-", ";", ":", "_"]
trocar_por = [" espaco ", " ponto ", 
              " traço ", " ponto e virgula ", 
              " dois pontos ", " sublinhado "]

for indx, ele1 in enumerate(a_trocar):
    while ele1 in lista:
        posicao = lista.index(ele1)
        lista[posicao] = trocar_por[indx]

elemento = "".join(lista)
print(elemento)

#%%
apaga_var(['elemento','lista','a_trocar','trocar_por',
           'indx', 'ele1', 'posicao'])

#%%
#########################################################
# Também podemos criar uma função para efetuar as trocas#
#  a partir de duas listas de valores, baixando a caixa #
#  dos caracteres na string e usar o recurso  de        #
#  compreensão de lista para executar esta operação em  #
#  todos os elementos de uma lista de strings           #
#########################################################
a_trocar   = [" ", ".", "-", ";", ":","_"]
trocar_por = ["_", "_", "_", "_", "_","_"]

def troca_valor(elemento, a_trocar, trocar_por):
    lista = list(elemento)

    for indice, valor in enumerate(a_trocar):
        ####################################
        #Para evitar um loop infinito se   #
        # o valor estiver nas duas listas  #
        ####################################
        if valor != trocar_por[indice]:
            while valor in lista:
                posicao = lista.index(valor)
                lista[posicao] = trocar_por[indice]
            
    elemento = "".join(lista)
    elemento = elemento.lower()
    return(elemento)
    
a = "Gustavo Correa Mirapalheta"
b = "Gustavo_Correa_Mirapalheta"
c = "Gustavo.Correa.Mirapalheta"
d = "Gustavo Correa.Mirapalheta"
e = "Gustavo;Correa-Mirapalheta-Nascido:RS"
lista = [a,b,c,d,e]

lista2 = [troca_valor(x, a_trocar, trocar_por) \
                                  for x in lista]

print("\n")
print("lista original = ", lista)
print("\n")
print("lista final = ", lista2)

#%%
apaga_var(['a','b','c','d','e','lista','lista2', 'x', 
           'a_trocar', 'trocar_por','indx'])

#%%
#########################################################
# Vamos então criar a função completa "troca_valor"     #
#  Ela irá receber uma lista de strings, um array de    #
#  valores a serem substituidos e outro array de valores#
#  a trocar. O valor de retorno será a nova lista com as#
#  strings alteradas. Como bonus vamos dar a opção do   #
#  usuário passar como valor de destino uma lista com   #
#  apenas um caracter, caso este em que todos os valores#
#  da lista original serão trocados por este caracter   #
#########################################################
a_trocar   = [" ", ".", "-", ";", ":","_"]
trocar_por = ["_", "_", "_", "_", "_","_"]

a = "Gustavo Correa Mirapalheta"
b = "Gustavo_Correa_Mirapalheta"
c = "Gustavo.Correa.Mirapalheta"
d = "Gustavo Correa.Mirapalheta"
e = "Gustavo;Correa-Mirapalheta-Nascido:RS"
lista = [a,b,c,d,e]

def troca_valor(lista, a_trocar, trocar_por):

    if len(trocar_por)==1:
        elem = trocar_por[0]
        quant = len(a_trocar)
        trocar_por = list(np.repeat(elem, quant))
    
    def troca_elem(elemento, 
                   a_trocar = a_trocar, 
                   trocar_por = trocar_por):
        
        elemento = elemento.lower()
        list_elem = list(elemento)    
        
        for indice, valor_orig in enumerate(a_trocar):
            novo_valor = trocar_por[indice]
            
            if valor_orig != novo_valor:
                
                while valor_orig in list_elem:
                    posicao = list_elem.index(valor_orig)
                    list_elem[posicao] = novo_valor
                    
        elem_novo = "".join(list_elem)            
        return(elem_novo)
        
    lista = [troca_elem(x) for x in lista]
    return(lista)        
    
troca_valor(lista, a_trocar, trocar_por)

#%%
apaga_var(['a', 'a_trocar', 'b', 'c', 'd', 'e', 
          'lista', 'trocar_por', 'lista2', 'r', 'apagar',
          'new_names','old_names', 'vars','indx'])

    
#%%
#########################################################
# Utilizando então a função para efetuar a troca do nome#
#  das colunas do dataframe iris                        #
#########################################################
iris.columns = troca_valor(iris.columns, ["."], ["_"])

iris.columns

#%%
#########################################################
# Ajuste dos dados - categorias numéricas               #
#########################################################

#%%
#########################################################
# As previsões criadas por um modelo que utilize na     #
#  etapa de treino a coluna 'species' serão categóricas.#
#  Para efeito de visualização, o ideal é que tenhamos  #
#  valores numéricos correspondentes. Sendo assim vamos #
#  uma outra coluna no data_frame iris de nome          #
#  species_no com valores numéricos para as categorias  #
#  em species                                           #
#########################################################

#%%
#########################################################
# Esta operação pode ser implementada diretamente com   #
#  listas, através de numpy arrays ou por meio de um    #
#  dicionário. Vamos criar um dataframe de nome 'tempos'#
#  com colunas 'metodo' e 'tempo' para armazenar os     #
#  tempos de execução de cada método. Para calcular os  #
#  tempos de execução vamos utilizar a função time() a  #
#  qual foi carregada na etapa de setup deste script com#
#  o comando "from time import time. Ela apresenta      #
#  tempos que até a faixa de 0.1 micro segundos (100    #
#  nano segundos). Como temos apenas 150 valores para   #
#  atualizar, é possível que time não consiga mensurar  #
#  uma diferença de tempo tão pequena. Sendo assim      #
#  vamos repetir a operação 1.000 vezes e medir o tempo #
#  total em cada método.                                #
#########################################################

#%%
#########################################################
# Dataframe para armazenagem dos tempos de execução     #
#########################################################
tempos = pd.DataFrame(columns = ['metodo','tempo'])

#%%
#########################################################
# Por listas                                            #
#########################################################

#%%
#########################################################
# Transformar uma lista em um set tem o efeito de gerar #
#  os valores unicos da lista. Os sets não são ordenados#
#  tal como as listas, portanto após gerar os valores   #
#  únicos deve-se transformar o set resultante          #
#  novamente em uma lista e classificar o resultado     #
#  através do método .sort()                            #
#########################################################
species_names = list(set(iris['species']))
species_names.sort()
inicio = time()
for i in range(1000):
    species_no = []

    for nome in iris['species']:
        numero = species_names.index(nome)
        species_no.append(numero)
fim = time()
tempo = fim - inicio

#%%
#########################################################
# Vamos inserir o numpy array species_no_listas no      #
#  dataframe iris em uma coluna de mesmo nome para      #
#  depois comparar se os métodos de geração das         #
#  categorias numéricas estão produzing resultados      #
#  consistentes entre si                                #
######################################################### 
iris['species_no_listas'] = species_no

#%%
#########################################################
# Salvamos o resultado do tempo de execução pelo método #
#  que utiliza listas no dataframe tempos. Observe que  #
#  neste caso estamos adicionando uma nova linha ao     #
#  dataframe                                            #
######################################################### 
df = pd.DataFrame(data = [['lista',tempo]],
                  columns = ['metodo','tempo'])

#%%
tempos = tempos.append(df, ignore_index=True)

#%%
#########################################################
# Por numpy arrays                                      #
#########################################################

#%%
#########################################################
# Para executar a operação de comparação é necessário   #
#  converter a lista de nomes original em um numpy      #
#  array. Caso contrário a comparação não será feita    #
#  elemento a elemento tal como desejamos. Neste caso   #
#  não é necessário classificar a lista de nomes pois a #
#  função np.unique devolve os valores únicos           #
#  classificados em ordem crescente                     #
#########################################################
species_names_np = np.array(np.unique(iris['species']))

inicio = time()
for i in range(1000):
    species_no = np.array([])

    for nome in iris['species']:
        numero = np.where(species_names_np == nome)[0][0]
        species_no = np.append(species_no, numero)
        
fim = time()
tempo = fim - inicio

#%%
#########################################################
# Mesmos procedimentos anteriores. Criar nova coluna em #
#  iris e acrescentar uma linha em tempos               #
#########################################################
iris['species_no_arrays'] = species_no

#%%
#########################################################
# Inserção da nova linha em tempos                      #
#########################################################
df = pd.DataFrame(data = [['array_append',tempo]],
                  columns = tempos.columns)

#%%
tempos = tempos.append(df, ignore_index=True)

#%%
#########################################################
# Por dicionários                                       #
#########################################################

#%%
#########################################################
# Primeiro vamos escrever uma função que irá criar o    #
#  dicionário para nós a partir dos valores presentes   #
#  na coluna species. Os valores únicos serão obtidos   #
#  através de set e sua reconversão para lista e a      #
#  classificação das chaves através do método .sort     #
#  nativo para as listas no python.                     #
#########################################################
def gera_dicionario(lista):
    dicionario = {}
    chaves = list(set(lista))
    chaves.sort()
    n = len(chaves)
    valores = range(n)

    def insere(chave,valor,dicionario=dicionario):
        dicionario[chave] = valor
        return(dicionario)

    [insere(chaves[x], valores[x]) for x in range(n)]
    return(dicionario)

dicionario = gera_dicionario(iris['species'])

#%%
#########################################################
# Em seguida usamos a compreensão de lista para gerar os#
#  valores correspondentes às chaves presentes na coluna#
#  'species' do dataframe iris                          #
#########################################################
inicio = time()
for i in range(1000):
    species_no = []    
    [species_no.append(dicionario[chave]) \
                       for chave in iris['species']]
fim = time()
tempo = fim - inicio

#%%
#########################################################
# Mesmos procedimentos anteriores. Criar nova coluna em #
#  iris e acrescentar uma linha em tempos               #
#########################################################
iris['species_no_dicionarios'] = species_no

#%%
#########################################################
# Inserção da nova linha em tempos                      #
#########################################################
df = pd.DataFrame(data = [['dicionarios',tempo]],
                  columns = tempos.columns)

#%%
tempos = tempos.append(df, ignore_index=True)

#%%
##########################################################
# Verificando se as conversões foram consistentes entre  #
#  si                                                    #
##########################################################
result1 = (iris['species_no_listas'] == \
           iris['species_no_arrays']).all()

result2 = (iris['species_no_arrays'] == \
           iris['species_no_dicionarios']).all()

print(result1, result2)

#%%
##########################################################
# Comparando os resultados vemos que a opção mais rápida #
# através do uso de dicionários                          #
##########################################################
tempos.sort_values('tempo')

#%%
##########################################################
# Criando uma coluna de nome 'species_no'                #
##########################################################
iris['species_no'] = iris['species_no_dicionarios']

#%%
##########################################################
# Deletando as colunas species_no_listas,                #
# species_no_arrays, species_no_dicionarios              #
##########################################################
iris.drop(['species_no_listas', 'species_no_arrays', 
           'species_no_dicionarios'], axis=1, inplace=True)

#%%
##########################################################
# Eliminando as variáveis                                #
##########################################################
apagar = ['df','dicionario','fim','i','inicio','nome',
          'numero','result1','result2','species_names',
          'species_names_np','species_no','tempo', 
          'nomes','apagar']
apaga_var(apagar)

#%%
#########################################################
# É importante salientar que através das etapas         #
#  de ajuste de dados revisamos diversos (se não a      #
#  maioria) dos recursos da linguagem Python: funções,  #
#  métodos para listas de caracteres, diferença entre   #
#  string e lista de caracteres, loops regulares, loops #
#  com enumerate, compreensão de lista, passagem de     #
#  parâmetros por designação além de criação de         #
#  dataframes, inserção de linhas e colunas, deleção de #
#  colunas e acesso por .loc e .iloc.                   #
#########################################################

#%%
#########################################################
# 3a etapa - Desenvolvimento e ajuste (fit) do modelo   #
#########################################################

#%%
#########################################################
# Façamos agora um gráfico de pares para observar como  #
#  as variáveis podem ser utilizadas para classificar a #
#  espécie a qual pertence o registro. Não devemos      #
#  utilizar a coluna species_no neste gráfico           #
#########################################################
plt.close()
sns.pairplot(data=iris.iloc[:,0:5], hue='species')
plt.show()

#%%
#########################################################
# 1o Modelo - Gerar um modelo de classificação para     #
#  as espécies utilizando o modelo de support vector    #
#  machines em modo linear com as variável petal_length #
#  e petal_width. A variável de saida será species      #
#########################################################
# A biblioteca correspondente foi carregada com:        #
# from sklearn import svm                               #
#########################################################

#%%
#########################################################
# O processo segue o mesmo padrão dos modelos de        #
#  regressão. Primeiro criamos o modelo "vazio". Neste  #
#  primeiro exemplo vamos utilizar o modelo com kernel  #
#  linear. Nos demais modelos (SVC e NuSVC) pode-se     #
#  especificar o tipo de kernel (função de ajuste) a ser#
#  utilizada na etapa de fit. Neste exercício, a função #
#  escolhida (LinearSVC) admite somente kernels lineares#
#########################################################
svm_linear = svm.LinearSVC(\
                    random_state=parametros['semente'],
                    C = 1)

#%%
#########################################################
# Criamos as matrizes com os dados a serem ajustados. A #
#  matrix X terá n linhas e 2 colunas neste exemplo pois#
#  utilizaremos duas variáveis para prever a espécie,   #
#  lembrando: petal_length e petal_width. A matriz y    #
#  é 1D com n elementos e será formada pela coluna de   #
#  nome "species" no dataframe iris.                    #
#########################################################
X_treino = iris[['petal_length','petal_width']]
Y_treino = iris['species']

#%%
#########################################################
# Ajustamos o modelo aos dados                          #
#########################################################
svm_linear.fit(X_treino,Y_treino)

#%%
#########################################################
# E obtemos as previsões para os dados originais de     #
#  entrada armazenando as mesmas em uma coluna de nome  #
#  'species_svml' no nosso dataframe pandas 'iris'      #
#########################################################
iris['species_svml'] = svm_linear.predict(X_treino)

#%%
#########################################################
# Vamos incluir uma coluna de nome 'species_svml_result'#
# a qual irá indicar se a classificação do modelo está  #
# correta (True) ou não (False)                         #
#########################################################
iris['species_svml_result'] = \
    iris['species'] == iris['species_svml']

#%%
#########################################################
# E também uma coluna para indicar o número da categoria#
#  prevista pelo modelo. O nome da coluna será          #
#  'species_svml_no' e será produzida pelo método com   #
#  dicionários conforme abaixo                          #
#########################################################
nomes_especies = {'setosa':1,
                  'versicolor':2,
                  'virginica':3}

species_svml_no = []
for i, valor in enumerate(iris['species_svml']):
    numero = nomes_especies[valor]
    species_svml_no.append(numero)
    
iris['species_svml_no'] = species_svml_no

#%%
apaga_var(['i', 'valor', 'numero', 'species_svml_no',
           'nomes_especies'])
    
#%%
#########################################################
# Eliminamos as variáveis intermediárias X_treino e     #
#  Y_treino                                             #
#########################################################
apaga_var(['X_treino','Y_treino'])

#%%
#########################################################
# Após obter as previsões vamos criar um grid com uma   #
#  série de pontos novos nos intervalos das variáveis de#
#  treino e em seguida criar um gráfico de contorno com #
#  previsões e um scatter plot com os valores observados#
#  das espécies.                                        #
#########################################################

#%%
#########################################################
# Usuários de R: Lembrem-se que para os pandas          #
# dataframes ficarem "parecidos" com os dataframes do R #
# deve-se utilizar os métodos .loc e .iloc (.loc para   #
# filtragem condicional e .iloc para  filtragem por     #
# número, isto é por "slice" no jargão do Python)       #
#########################################################
 
#%%
#########################################################
# Valores das variáveis de entrada do modelo presentes  #
#  no conjunto de dados utilizado para treino do modelo #
#########################################################
X_treino = iris[['petal_length','petal_width']]

#%%
#########################################################
# Criação do eixo coordenado x com n pontos de x_min a  #
#  x_max                                                #
#########################################################
x_min = X_treino.iloc[:,0].min() - parametros['espessura']
x_max = X_treino.iloc[:,0].max() + parametros['espessura']
eixo_x = np.linspace(x_min, x_max, parametros['n_pontos'])
apaga_var(['x_min', 'x_max'])

#%%
#########################################################
# Criação do eixo coordenado y com n pontos de y_min a  #
#  y_max                                                #
#########################################################
y_min = X_treino.iloc[:,1].min() - parametros['espessura']
y_max = X_treino.iloc[:,1].max() + parametros['espessura']
eixo_y = np.linspace(y_min, y_max, parametros['n_pontos'])
apaga_var(['y_min', 'y_max'])

#%%
#########################################################
# Eliminação de X_treino                                #
######################################################### 
apaga_var(['X_treino'])

#%%
#########################################################
# Criação do quadriculado de desenho no plano (x,y) com #
# n x n pontos. Serão gerados dois dataframes n x n. Em #
# um deles (o do eixo x, denominado x neste caso, os    #
# valores serão constantes para cada coluna, variando ao#
# longo das linhas. No outro (relativo ao eixo y, y     #
# neste exemplo) eles serão constantes por linha e      #
# por coluna, tal como ocorreria se desenhassemos as    #
# posições no plano                                     #
#########################################################
x, y = np.meshgrid(eixo_x, eixo_y)
apaga_var(['eixo_x', 'eixo_y'])

#%%
#########################################################
# Por último para gerar todas as combinações de pares xy#
# que serão utilizadas nas previsões da variável espécie#
# primeiro "desmontamos" cada um dos arrays, com ravel()#
# colocamos os mesmos desmontados lado a lado,          #
# reformatamos o mesmo para (2,-1) (2,nxn) e pedimos    #
# que o array resultante seja transposto.               #
# IMPORTANTE: neste método é essencial reformatar       #
# primeiro e depois transpor. Não deve-se tentar uma    #
# reformatação "direta" para (-1,2) pois isto vai       #
# embaralhar os pontos tornando os mapas de cores com   #
# as previsões sem nenhum sentido.                      #
#########################################################
X_new = np.array([x.ravel(), y.ravel()]).reshape(2,-1).T

#%%
#########################################################
# Vamos armazenar estes novos valores de petal_length e #
#  petal_width em um dataframe de nome iris_new         #
#########################################################
iris_new = pd.DataFrame(data = X_new, 
                        columns = ['petal_length',
                                   'petal_width'])
#%%
#########################################################
# Eliminamos a variável X_new intermediária, pois agora #
#  podemos buscar estes valores diretamente das colunas #
#  petal_length e petal_width do dataframe iris_new     #
#########################################################
apaga_var('X_new')

#%%
#########################################################
# Em seguida com o array dos pares x,y criado (com o    #
# nome de X_new) podemos gerar as previsões da variável #
# species, armazenando estes resultados diretamente no  #
# dataframe pandas iris_new em uma coluna de nome       #
# 'species'. Observe que agora estamos buscando os      #
# valores de X_new diretamente nas colunas do dataframe #
# iris_new                                              #
#########################################################
iris_new['species'] = \
     svm_linear.predict( \
                iris_new[['petal_length','petal_width']]) 

#%%
#########################################################
# Geração dos número de categoria diretamente a partir  #
#  do modelo. O modelo original foi ajustado a partir   #
#  de uma coluna com valores categóricos nas variáveis  #
#  Se ajustarmos o modelo novamente, desta vez com      #
#  valores numéricos na saida, ele passará a produzir   #
#  como resultado não mais os nomes das categorias mas  #
#  sim os valores designados para elas. Observe que     #
#  mesmo com valores numéricos na variável de saida,    #
#  o modelo de support vector machines do scikit-learn  #
#  considerará as mesmas como variáveis categóricas não #
#  ordinais                                             #
#########################################################

#%%     
#########################################################
# Após o modelo ter sido ajustado ao novo padrão de     #
#  saida de dados, geramos as previsões correspondentes #
#########################################################
     
#%%
#########################################################
# Uma vez que calculamos os tempos de execução gerar os #
#  números de categoria com listas, numpy arrays e      #
#  dicionários na etapa anterior, vamos também calcular #
#  estes tempos aqui para o procedimento de fit e       #
#  e predict. lá foram feitas 1000 x 150 = 150.000      #
#  inserções. Como no dataframe de previsões são 10.000 #
#  linhas temos de fazer 10 rodadas                     #
#########################################################
# Lembrando que o modelo "vazio" foi gerado com os      #
#  comandos:                                            #
# svm_linear = svm.LinearSVC(\                          #
#                  random_state=parametros['semente'],  #
#                  C = 1)                               #
#########################################################
svm_linear.fit(iris[['petal_length','petal_width']],\
               iris['species_no'])

inicio = time()
for i in range(10):
    iris_new['species_no'] = \
             svm_linear.predict( \
                iris_new[['petal_length','petal_width']])
fim = time()

tempo = fim - inicio

#%%
#########################################################
# Armazenamos os valores de duração no dataframe tempos #
#########################################################
df = pd.DataFrame( data = [['modelo_svml',tempo]],
                   columns = tempos.columns)

tempos = tempos.append(df, ignore_index=True)

tempos

#%%
#########################################################
# Impressionante não? É mais rápido gerar as previsões  #
#  pelo scikit-learn novamente do que fazer a inserção  #
#  por dicionários                                      #
#########################################################
#%%
apaga_var(['i', 'inicio', 'fim', 'df', 'tempo'])
     
#%%
#########################################################
# 4a etapa - Visualização de Resultados                 #
#########################################################
 
#%%
#########################################################
# O próximo passo é criar o gráfico de dispersão de     #
#  petal_length x petal_width e colorir os pontos de    #
#  acordo com o valor de species                        #
#########################################################
plt.close()
sns.scatterplot(data = iris, x='petal_length',
                             y='petal_width',
                             hue='species')
plt.show()

#%%
#########################################################
# Vamos acrescentar um título ao gráfico. Para isso será#
#  necessário mesclar recursos da matplotlib com a      #
#  seaborn. Lembre-se também que a seaborn na verdade é #
#  um conjunto de chamadas à matplotlib, que visam      #
#  tornar a criação de alguns tipos de gráficos mais    #
#  fácil de ser executada.                              #
#########################################################
plt.close()

fig, ax = plt.subplots()
ax.set_title("Previsão x Real - SVM Kernel Linear")
sns.scatterplot(data = iris, x = 'petal_length',
                             y = 'petal_width',
                             hue = 'species')
plt.show()

#%%
#########################################################
# Incluiremos uma borda espessa nos pontos que foram    #
#  classificados de forma incorreta, salientando os     #
#  mesmos no gráfico                                    #
#########################################################

#%%
#########################################################
# Para obter este efeito primeiro mapeamos os valores na#
#  coluna 'species_svml_result' para uma variável com   #
#  um nome mais curto. Isto é feito por mera            #
#  conveniência                                         #
#########################################################
svml_result = iris['species_svml_result']

#%%
#########################################################
# Em seguida criamos duas correspondências a partir de  #
#  dicionários. A primeira é para a cor das bordas. Se  #
#  o ponto tiver sido classificado errado pelo modelo   #
#  o valor de svml_result será False. Caso contrário    #
#  será True. Se for False o dicionário indicará a cor  #
#  preta ('black'). Se for True indicará sem cor        #
#  ('none'). Este dicionário é então mapeado para a     #
#  lista bordas_pontos através do recurso de            #
#  compreensão de lista e em seguida salvo no dataframe #
#  iris, seguindo nossa estratégia de manter o número de#
#  variáveis livres o menor possível                    #
#########################################################
bordas = {False:'black', True:'none'}
bordas_pontos = [bordas[x] for x in svml_result]
iris['svml_bordas'] = bordas_pontos
apaga_var(['bordas', 'bordas_pontos'])

#%%
#########################################################
# Um processo similar é executado para as larguras.     #
#  Pode-se pensar esta etapa (a qual é similar a        #
#  anterior) como a criação de uma nova coluna no       #
#  dataframe iris                                       #
#########################################################
largura = {False:2, True:0}
larg_pontos = [largura[x] for x in svml_result] 
iris['svml_larg'] = larg_pontos
apaga_var(['largura','larg_pontos', 'svml_result'])

#%%
#########################################################
# Vamos agora desenhar o gráfico. As larguras das bordas#
# são obtidas com linewidth e as cores das mesmas são   #
# obtidas com edgecolor. No caso do tamanho dos pontos  #
# a função scatterplot tem a opção de indicar o código  #
# para o tamanho com size (e neste caso nós indicamos a #
# coluna species_svml_result) e um mapeamento para estes#
# códigos (que neste exemplo são False e True). O       #
# mapeamento foi indicado pelo parâmetro sizes o qual   #
# recebeu uma lista com dois valores 90 e 30.           #
#########################################################

#%%
#########################################################
# A preparação acima e as opções size e sizes da        #
# scatterplot na prática fazem com que o ponto se tiver #
# sido classificado de forma errada (tendo False como   #
# valor no campo species_svml_result) tenha: tamanho 90 #
# borda de cor preta e largura igual a 2. Caso ele tenha#
# sido classificado corretamente ele terá borda sem cor,#
# com largura 0 e tamanho 30                            #
#########################################################
plt.close()
fig, ax = plt.subplots()
ax.set_title("Previsão x Real - SVM Kernel Linear")
sns.scatterplot(data = iris, 
    x = 'petal_length', y = 'petal_width', hue = 'species',
    size='species_svml_result', sizes=[90,30],
    linewidth=iris['svml_larg'], edgecolor=iris['svml_bordas'])

plt.show()

#%%
#########################################################
# Agora é hora de criar o gráfico de contorno com os    #
#  os valores previstos da variável species (os quais   #
#  foram produzidos a partir de X_new e estão salvos no #
#  dataframe iris_new, colunas species e species_no     #
#########################################################
plt.close()

minhas_cores = ['blue','green','red']
fig, ax = plt.subplots()
ax.set_title("Previsão x Real - SVM Kernel Linear")

sns.scatterplot(data = iris, 
    x = 'petal_length', y = 'petal_width', hue = 'species',
    palette = minhas_cores, ax=ax,
    size='species_svml_result', sizes=[90,30],
    linewidth=iris['svml_larg'], edgecolor=iris['svml_bordas'])

ax.contourf(x, y, 
            np.array(iris_new['species_no']).reshape(100,100),
            cmap = ListedColormap(minhas_cores), alpha = 0.1)
apaga_var('minhas_cores')
plt.show()
