import matplotlib.pyplot as plt
import numpy as np
from math import cos, sin, pi

fig = plt.figure()
ax = fig.gca(projection='3d')

# Constantes
NUMERO_DE_PONTOS = 20
CURVA = 0 # CURVA A SER UTILIZADA (0 - 2)
T0 = 0
T1 = pi
L = 0.1   # tamanho da seta

def derivar(array):
  n = array.shape[0]
  m = array.shape[1]
  new_array = np.zeros((n - 1, m))
  for i in range(n - 1):
    new_array[i, :] = (array[i + 1] - array[i]) / 2
  
  return new_array

def vetor_unitario(v):
  if (v == [0, 0, 0]).all == True:
    return [0, 0, 0]

  return v / modulo(v)

def modulo(v):
  return (v[0]**2 + v[1]**2 + v[2]**2)**0.5

def f1(t: float):
  return [cos(t), sin(t), t]

def f1T(t):
  return np.array([-sin(t), cos(t), 1]) / (2)**0.5
def f1N(t):
  return np.array([-cos(t), -sin(t), 0])
def f1B(t):
  return np.array([sin(t), -cos(t), 1]) / (2)**0.5



def f2(t: float):
  return [t, t*t, t**3]

def f2T(t):
  return np.array([1, 2*t, 3*t**2]) / (1 + 4*t**2 + 9*t**4)**0.5
def f2N(t):
  return np.array([-18*t**3 - 4*t, 2 - 18*t**4, 16*t**3 + 6*t]) / (4 + 52*t**2 + 264*t**4 + 580*t**6 + 324*t**8)**0.5
def f2B(t):
  return np.array([6*t**2 + 32*t**4 + 54*t**6, -6*t - 28*t**3 - 54*t**5, 2 + 8*t**2 + 18*t**4]) / (4 + 58*t**2 + 508*t**4 + 2104*t**6 + 5020*t**8 + 6516*t**10 + 2916*t**12)**0.5


def f3(t: float):
  return [sin(t), cos(t), 1]

def f3T(t):
  return np.array([cos(t), -sin(t), 0])
def f3N(t):
  return np.array([-sin(t), -cos(t), 0])
def f3B(t):
  return np.array([0, 0, -1])


dt = (T1 - T0) / NUMERO_DE_PONTOS
pos = np.zeros((NUMERO_DE_PONTOS, 3))
tanM = np.zeros((NUMERO_DE_PONTOS - 1, 3))
norM = np.zeros((NUMERO_DE_PONTOS - 2, 3))
binM = np.zeros((NUMERO_DE_PONTOS - 2, 3))

for i in range(NUMERO_DE_PONTOS):
  t = T0 + i * dt
  exec(F'pos[i, :] = f{CURVA}(t)')

  if i > 0:
    exec(F'tanM[i-1, :] = f{CURVA}(t)')
  if i > 1:
    exec(F'norM[i-2, :] = f{CURVA}(t)')
    exec(F'binM[i-2, :] = f{CURVA}(t)')


# Calculo do Triedo de Frenet
dpos_dt = derivar(pos)

Tangente = np.zeros((NUMERO_DE_PONTOS - 1, 3))

for i in range(NUMERO_DE_PONTOS - 1):
  Tangente[i, :] = vetor_unitario(dpos_dt[i, :])

dTangente_dt = derivar(Tangente)
Normal = np.zeros((NUMERO_DE_PONTOS - 2, 3))
for i in range(NUMERO_DE_PONTOS - 2):
  Normal[i, :] = vetor_unitario(dTangente_dt[i, :])

Binormal = np.cross(Tangente[1:, :], Normal[:, :])

# print("tangente")
# for i in range(NUMERO_DE_PONTOS - 1):
#   print(modulo(tanM[i,:] - Tangente[i,:]),'\t', modulo(tanM[i,:] - Tangente[i,:]) / modulo(tanM[i,:])*100,'%')

# print("normal")
# for i in range(NUMERO_DE_PONTOS - 2):
#   print(modulo(norM[i,:] - Normal[i,:]),'\t', modulo(norM[i,:] - Normal[i,:]) / modulo(norM[i,:])*100,'%')

# print("binormal")
# for i in range(NUMERO_DE_PONTOS - 2):
#   print(modulo(binM[i,:] - Binormal[i,:]),'\t', modulo(binM[i,:] - Binormal[i,:]) / modulo(binM[i,:])*100,'%')


ax.quiver(*pos[1:].T, *Tangente.T, length=L, color=(0, 0, 1))
ax.quiver(*pos[2:].T, *Normal.T,   length=L, color=(0, 1, 0))
ax.quiver(*pos[2:].T, *Binormal.T, length=L, color=(1, 0, 0))

plt.show()
