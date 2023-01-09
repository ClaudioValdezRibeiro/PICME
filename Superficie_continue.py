import matplotlib.pyplot as plt
import numpy as np
from math import cos, sin, pi, e

fig = plt.figure()
ax = fig.gca(projection='3d')

def fn(n, u, v):
  if n == 1:
    return ([sin(u) * cos(v), sin(u) * sin(v), cos(u)],\
            [cos(u) * cos(v), cos(u) * sin(v), - sin(u)],\
            [- sin(u) * sin(v), sin(u) * cos(v), 0],\
            [sin(u) * cos(v), sin(u) * sin(v), cos(u)])
    
  if n == 2:
    return ([u + v, u - v, u*v],\
            [1, 1, v],\
            [1, -1, u],\
            [u + v, v - u, -2])

  if n == 3:
    return ([e**u * cos(v), e**u * sin(v), u**2],\
            [e**u * cos(v), e**u * sin(v), 2 * u],\
            [- e**u * sin(v), e**u * cos(v), 0],\
            [- 2 * u * e**u * cos(v), - 2 * u * e**u * sin(v), e**(2*u)])
    
def vetor_unitario(v):
  v = np.asarray(v)
  if (v == [0, 0, 0]).all == True:
    return [0, 0, 0]

  return v / modulo(v)

def modulo(v):
  return (v[0]**2 + v[1]**2 + v[2]**2)**0.5

def mapLinear(i, i0, i1, f0, f1):
  return (i - i0) * (f1 - f0) / (i1 - i0) + f0 

# Constantes
U0 = 0
U1 = pi / 2
V0 = -pi/2
V1 = 0
NU = 4
NV = 6
DESVIO = 0.001
LENGTH = 0.5

TIPO_FUNCAO = 3

TOTAL_PONTOS = NU * NV

pos    = np.zeros((TOTAL_PONTOS, 3))
tan_u  = np.zeros((TOTAL_PONTOS, 3))
tan_v  = np.zeros((TOTAL_PONTOS, 3))
normal = np.zeros((TOTAL_PONTOS, 3))

for i in range(TOTAL_PONTOS):
  u = mapLinear(i % NU, 0, NU - 1, U0 + DESVIO, U1 - DESVIO)
  v = mapLinear(i // NU, 0, NV - 1, V0 + DESVIO, V1 - DESVIO)

  pos[i, :], tan_u[i, :], tan_v[i, :], normal[i, :] = fn(TIPO_FUNCAO, u, v)


DU = (U1 - U0) / (NU - 1) 
DV = (V1 - V0) / (NV - 1)
x = np.ndarray((NU,NV))
y = np.ndarray((NU,NV))
z = np.ndarray((NU,NV))
for j in range(NV):
  for i in range(NU):
    u = U0 + i * DU + DESVIO
    v = V0 + j * DV - DESVIO

    x[i,j], y[i,j], z[i,j] = fn(TIPO_FUNCAO, u, v)[0]


ax.plot_surface(x, y, z) 

ax.quiver(*pos.T, *tan_u.T, color=(0, 0, 1), normalize=True, length = LENGTH)
ax.quiver(*pos.T, *tan_v.T, color=(0, 1, 0), normalize=True, length = LENGTH)
ax.quiver(*pos.T, *normal.T, color=(1, 0, 0), normalize=True, length = LENGTH)

plt.show
