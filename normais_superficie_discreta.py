# IMPORTAR AS BIBLIOTECAS
import random
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn import neighbors
from sklearn.decomposition import PCA


# CONTANTES
TOTAL_PONTOS = 100


# E FUNÇÕES
def paraboloide_hiperbolico(u, v):
  return np.array([u, v, u**2-v**2])

def normal_PH(u, v):
  d = np.sqrt(1 + 4*v**2 + 4*u**2)
  return np.array([-2*u / d, 2*v / d, 1.0 / d])

def produto_vetorial(a, b):
  return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])

def angulo_interno(a, b):
  return np.arccos((a[0]*b[0]+a[1]*b[1]+a[2]*b[2])/(modulo(a)*modulo(b)))

def vetor_unitario(v):
  if (v == [0, 0, 0]).all == True:
    return np.array([0, 0, 0])

  return v / modulo(v)

def modulo(v):
  return (v[0]**2 + v[1]**2 + v[2]**2)**0.5

def mapLinear(i, i0, i1, f0, f1):
  return (i - i0) * (f1 - f0) / (i1 - i0) + f0 


# GERA PONTOS ALEATORIOS NO ESPAÇO 2D E SALVA EM "pontos_aleatorios.txt"
arquivo = open('pontos_aleatorios.txt','w')

for i in range(TOTAL_PONTOS):
  u = random.uniform(-1,1)
  v = random.uniform(-1,1)
  arquivo.write(f'{u};{v}\n')

arquivo.close()

dados = np.loadtxt('pontos_aleatorios.txt', delimiter=';')

# APLICA OS PONTOS GERADOS NA SUPERFÍCIE
superficie = paraboloide_hiperbolico(dados[:,0],dados[:,1])
superficie = superficie.T
normal_matematico = normal_PH(dados[:,0],dados[:,1])
normal_matematico = normal_matematico.T


# GERA UM GRAFO CONEXO COM A SUPERFÍCIE DISCRETA
total_vizinhos = 1
while True:
  model = neighbors.kneighbors_graph(superficie,total_vizinhos,mode='distance')
  grafo = nx.Graph(model)

  if nx.number_connected_components(grafo) == 1:
    break
  
  total_vizinhos += 1


normal_discreto = np.ndarray((TOTAL_PONTOS, 3))
for i in range(TOTAL_PONTOS):
  # Pego o ponto i e seus vizinhos
  vizinhanca = list(grafo[i].keys()) + [i]

  # Aplicando a Analise de Componentes Principais (PCA) em i
  pca = PCA(n_components=3) # em 3D
  pca.fit(superficie[vizinhanca])

  # Pego o ultimo vetor (onde os dados estam menos espalhados)
  normal = pca.components_[2]
  normal_discreto[i] = vetor_unitario(normal)


diferencas = np.ndarray((TOTAL_PONTOS))
for i in range(TOTAL_PONTOS):
  theta = min(angulo_interno(normal_discreto[i], normal_matematico[i]),\
              angulo_interno(-normal_discreto[i], normal_matematico[i]))

  diferencas[i] = theta

media = np.mean(diferencas)
desvio_padrao = np.std(diferencas)

# Cria uma figura 3d
figura = plt.figure()
eixos = figura.add_subplot(111, projection="3d")


# arestas = np.array([(superficie[u], superficie[v]) for u, v in grafo.edges()])
# for aresta in arestas:
#     eixos.plot(*aresta.T, color="tab:gray")

eixos.quiver(*superficie.T, *normal_matematico.T, length=0.5, color=(0, 0, 1))
eixos.quiver(*superficie.T, *normal_discreto.T, length=0.5, color=(1, 0, 0))
# eixos.scatter(*superficie.T, s=10, c="r")



print("Cada ponto tem no mínimo {} vizinhos para formar um grafo continuo".format(TOTAL_PONTOS))
print("A média de erros angulares é de {} rad com um desvio padrão de {} rad".format(media, desvio_padrao))

plt.show()
