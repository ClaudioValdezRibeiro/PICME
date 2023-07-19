# IMPORTAR AS BIBLIOTECAS NECESSÁRIAS
import random
random.seed(10)

import plotly.graph_objects as go
import numpy as np
import networkx as nx

from numpy import pi, cos, sin, tan
from sklearn import neighbors
from queue import PriorityQueue


# FUNÇÕES AUXILIARES
class Paraboloide_hiperbolico:
  def __repr__(self):
    return 'paraboloide_hiperbolico'
  def definition(self, u, v):
    u_=u*4-2
    v_=v*4-2
    return np.array([u_, v_, (u_)**2 - (v_)**2])

class Esfera:
  def __repr__(self):
    return 'esfera'

  def definition(self, u, v):
    # u = (0, pi), v = (0, 2pi)
    return np.array([sin(u*pi)*cos(v*pi), sin(u*pi)*sin(v*pi), cos(u*pi)])

class Plano:
  def __repr__(self):
    return 'plano'
  def definition(self, u, v):
    return np.array([u, v, u+v])


def dijkstra(grafo, inicial, final):
  anterior = dict()
  visitados = set()
  dist = {vertice: float('inf') for vertice in grafo.nodes()}
  dist[inicial] = 0

  pq = PriorityQueue()
  pq.put((dist[inicial], inicial))

  while not pq.empty():
    dist_atual, atual = pq.get()

    visitados.add(atual)

    if atual == final:
      break

    for vizinho in dict(grafo.adjacency()).get(atual):
      nova_dist = dist_atual + grafo[atual][vizinho]['weight']
      if nova_dist < dist[vizinho]:
        dist[vizinho] = nova_dist
        anterior[vizinho] = atual

        if vizinho not in visitados:
          pq.put((dist[vizinho], vizinho))


  vertice = final
  caminho = []
  while vertice != inicial:
      caminho.append(vertice)
      vertice = anterior[vertice]
  caminho.append(vertice)
  caminho.reverse()
  return (dist[final], caminho)

# GERA PONTOS ALEATORIOS NO ESPAÇO 2D E SALVA EM "pontos_aleatorios.txt"
# with open('pontos_aleatorios.txt','w') as arquivo:
#   for i in range(TOTAL_PONTOS):
#     # u = random.uniform(0,1)
#     # v = random.uniform(0,1)
#     while 0 > (u := random.gauss(0.5, 1/6)) or u > 1:
#       pass
#     while 0 > (v := random.gauss(0.5, 1/6)) or v > 1:
#       pass
#     arquivo.write(f'{u};{v}\n')


# CONTANTES
TOTAL_PONTOS = 300
PONTO_INICIAL = 0
PONTO_FINAL = 2
SURFACE = Plano()


dados = np.loadtxt('pontos_aleatorios.txt', delimiter=';')

# APLICA OS PONTOS GERADOS NA SUPERFÍCIE
superficie = SURFACE.definition(*dados.T).T
# GERA UM GRAFO CONEXO COM A SUPERFÍCIE DISCRETA
total_vizinhos = 1
while True:
  model = neighbors.kneighbors_graph(superficie,total_vizinhos,mode='distance')
  grafo = nx.Graph(model)

  if nx.number_connected_components(grafo) == 1:
    break

  total_vizinhos += 1

distancia, caminho_minimo = dijkstra(grafo, PONTO_INICIAL, PONTO_FINAL)

# CRIA O CAMINHO MÍNIMO DA SUPERFÍCIE DISCRETA
arestas_caminho_minimo = list(zip(caminho_minimo, caminho_minimo[1:]))
aresta_superficie = np.array([[superficie[u], superficie[v]] for u, v in arestas_caminho_minimo])
gradiante = np.linspace(0, 1, len(aresta_superficie))

fig = go.Figure()

for aresta in np.array([[superficie[u], superficie[v]] for u, v in grafo.edges()]):
  fig.add_trace(go.Scatter3d(
                            x=aresta[:,0],
                            y=aresta[:,1],
                            z=aresta[:,2],
                            mode='lines',
                            legendgroup="superficie",
                            showlegend=False,
                            line_color="#aaa"))

for aresta, t in zip(aresta_superficie, gradiante):
  fig.add_trace(go.Scatter3d(
                            x=aresta[:,0],
                            y=aresta[:,1],
                            z=aresta[:,2],
                            mode='lines',
                            legendgroup="caminho mínimo",
                            showlegend=False,
                            line_color=f"rgb({t*255},0,{255-255*t})",
                            line_width=5))

print("Cada ponto tem no mínimo {} vizinhos para formar um grafo conexo.".format(total_vizinhos))
print("A distância encontrada foi de {}.".format(distancia))
print("E este caminho é representado saindo do azul para o vermenho no gráfico.")
fig.update_layout(
    margin=dict(r=20, l=10, b=10, t=10))

fig.show()
