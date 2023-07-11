# IMPORTAR AS BIBLIOTECAS NECESSÁRIAS
import random
import plotly.graph_objects as go
import numpy as np
import networkx as nx
from sklearn import neighbors
from queue import PriorityQueue

# CONTANTES
TOTAL_PONTOS = 100
PONTO_INICIAL = 0
PONTO_FINAL = 1

# FUNÇÕES AUXILIARES
def SUPERFICIE(u, v):
  return np.array([np.cos(2*u), np.sin(2*u), v])
  # return np.array([u, v, u**2-v**2])

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
# arquivo = open('pontos_aleatorios.txt','w')

# for i in range(TOTAL_PONTOS):
#   u = random.uniform(-1,1)
#   v = random.uniform(-1,1)
#   arquivo.write(f'{u};{v}\n')

# arquivo.close()

dados = np.loadtxt('pontos_aleatorios.txt', delimiter=';')

# APLICA OS PONTOS GERADOS NA SUPERFÍCIE
superficie = SUPERFICIE(dados[:,0],dados[:,1])
superficie = superficie.T

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
                            line=dict(
                              color="#aaa",   # choose a colorscale
                            )
                             ))

for aresta, t in zip(aresta_superficie, gradiante):
  fig.add_trace(go.Scatter3d(
                            x=aresta[:,0],
                            y=aresta[:,1],
                            z=aresta[:,2],
                            mode='lines',
                            line=dict(
                              color=f"rgb({t*255},0,{255-255*t})",   # choose a colorscale
                            )
                             ))

print("Cada ponto tem no mínimo {} vizinhos para formar um grafo conexo.".format(total_vizinhos))
print("A distância encontrada foi de {}.".format(distancia))
print("E este caminho é representado saindo do azul para o vermenho no gráfico.")

fig.show()
