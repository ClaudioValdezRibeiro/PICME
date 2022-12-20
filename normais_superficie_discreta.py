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

def angulo_interno(a, b):
  return np.arccos((a[0]*b[0]+a[1]*b[1]+a[2]*b[2])/(modulo(a)*modulo(b)))

def modulo(v):
  return (v[0]**2 + v[1]**2 + v[2]**2)**0.5


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


# CALCULA O VETOR NORMAL PARA CADA PONTO
normal_discreto = np.ndarray((TOTAL_PONTOS, 3))
for i in range(TOTAL_PONTOS):
  vizinhanca = list(grafo[i].keys()) + [i]

  pca = PCA(n_components=3)
  pca.fit(superficie[vizinhanca])

  normal = pca.components_[2]
  normal_discreto[i] = vetor_unitario(normal)


# CALCULA A MEDIA E DESVIO PADRAO COM BASE NO ANGULO ENTRE O DISCRETO E O MATECATICO
diferencas = np.ndarray((TOTAL_PONTOS))
for i in range(TOTAL_PONTOS):
  theta = min(angulo_interno(normal_discreto[i], normal_matematico[i]),\
              angulo_interno(-normal_discreto[i], normal_matematico[i]))

  diferencas[i] = theta

media = np.mean(diferencas)
desvio_padrao = np.std(diferencas)




figura = plt.figure()
eixos = figura.add_subplot(111, projection="3d")

# arestas = np.array([(superficie[u], superficie[v]) for u, v in grafo.edges()])
# for aresta in arestas:
#     eixos.plot(*aresta.T, color="tab:gray")
# eixos.scatter(*superficie.T, s=10, c="r")

eixos.quiver(*superficie.T, *normal_matematico.T, length=0.5, color=(0, 0, 1))
eixos.quiver(*superficie.T, *normal_discreto.T, length=0.5, color=(1, 0, 0))

print("Cada ponto tem no mínimo {} vizinhos para formar um grafo continuo".format(TOTAL_PONTOS))
print("A média de erros angulares é de {} rad com um desvio padrão de {} rad".format(media, desvio_padrao))

plt.show()
