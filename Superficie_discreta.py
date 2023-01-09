import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin, pi, e

fig = plt.figure()
ax = fig.gca(projection='3d')

def fn(n, u, v):
  if n == 1:
    return (np.array([sin(u) * cos(v), sin(u) * sin(v), cos(u)]),\
            [cos(u) * cos(v), cos(u) * sin(v), - sin(u)],\
            [- sin(u) * sin(v), sin(u) * cos(v), 0],\
            vetor_unitario([sin(u) * sin(u) * cos(v), sin(u) * sin(u) * sin(v), cos(u) * sin(u)]))
    
  if n == 2:
    return (np.array([u + v, u - v, u*v]),\
            [1, 1, v],\
            [1, -1, u],\
            vetor_unitario([u + v, v - u, -2]))

  if n == 3:
    return (np.array([e**u * cos(v), e**u * sin(v), u**2]),\
            [e**u * cos(v), e**u * sin(v), 2 * u],\
            [- e**u * sin(v), e**u * cos(v), 0],\
            vetor_unitario([- 2 * u * e**u * cos(v), - 2 * u * e**u * sin(v), e**(2*u)]))

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
U1 = pi /2 
V0 = -pi/2
V1 = 0
NU = 10
NV = 10
DESVIO = 0.001
LENGTH = 0.3

TIPO_FUNCAO = 3 # 1 - 3

TOTAL = NU * NV

pos    = np.zeros((TOTAL, 3))
tan_u  = np.zeros((TOTAL, 3))
tan_v  = np.zeros((TOTAL, 3))
normal = np.zeros((TOTAL, 3))
tu = np.zeros((TOTAL, 3))
tv = np.zeros((TOTAL, 3))
nr = np.zeros((TOTAL, 3))

for i in range(TOTAL):
  u = mapLinear(i % NU, 0, NU - 1, U0 + DESVIO, U1 - DESVIO)
  v = mapLinear(i // NU, 0, NV - 1, V0 + DESVIO, V1 - DESVIO)
  pos[i, :], tan_u[i, :], tan_v[i, :], normal[i, :] = fn(TIPO_FUNCAO, u, v)

DU = (U1 - U0) / NU
DV = (V1 - V0) / NV

for j in range(NV):
  for i in range(NU):
    u = mapLinear(i, 0, NU - 1, U0 + DESVIO, U1 - DESVIO)
    v = mapLinear(j, 0, NV - 1, V0 + DESVIO, V1 - DESVIO)
    tu[i + j * NU, :] = (fn(TIPO_FUNCAO, u, v)[0] - fn(TIPO_FUNCAO, u - DU, v)[0]) / DU 

for i in range(NU):
  for j in range(NV):
    u = mapLinear(i, 0, NU - 1, U0 + DESVIO, U1 - DESVIO)
    v = mapLinear(j, 0, NV - 1, V0 + DESVIO, V1 - DESVIO)
    tv[i + j * NU, :] = (fn(TIPO_FUNCAO, u, v)[0] - fn(TIPO_FUNCAO, u, v - DV)[0]) / DV

nr = vetor_unitario(np.cross(tu, tv))
  
# for i in range(TOTAL):
#   print(f"{modulo(tan_u[i,:] - tu[i,:]):10.6f}  {modulo(tan_u[i,:]-tu[i,:])/modulo(tan_u[i,:])*100:10.6f}% | \
#           {modulo(tan_v[i,:] - tv[i,:]):10.6f}  {modulo(tan_v[i,:]-tv[i,:])/modulo(tan_v[i,:])*100:10.6f}% | \
#           {modulo(normal[i,:] - nr[i,:]):10.6f}  {modulo(normal[i,:]-nr[i,:])/modulo(normal[i,:])*100:10.6f}%")

ax.quiver(*pos.T, *tu.T, color=(0, 0, 1), normalize=True,length = LENGTH)
ax.quiver(*pos.T, *tv.T, color=(0, 1, 0), normalize=True,length = LENGTH)
ax.quiver(*pos.T, *nr.T, color=(1, 0, 0), normalize=True,length = LENGTH)

plt.show()
