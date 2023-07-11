import sympy
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from math import cos, sin, pi
from sympy import diff, Matrix, symbols, sin, cos, simplify, sqrt

_t = symbols("t")

# Constantes
TOTAL_POINTS = 20
T_0 = 1
T_1 = pi

def curve(t):
  return [sin(t), cos(t), t]
  # return [sin(t), cos(t), 1]
  # return [t, t**2, t**3]

def diferenca_finita(array):
  n = array.shape[0]
  return 0.5*(array[1:] - array[:n-1])

def vetor_unitario(v):
  if (v == [0, 0, 0]).all == True:
    return [0, 0, 0]

  return v/np.sqrt(np.diag(v.dot(v.T))).reshape((v.shape[0],1))



F = Matrix(curve(_t))
T = simplify(F.diff(_t))
T = simplify(T/sqrt(T.dot(T)))
N = simplify(T.diff(_t))
N = simplify(N/sqrt(N.dot(N)))
B = simplify(T.cross(N))

dt = (T_1 - T_0) / TOTAL_POINTS
POSITION = np.zeros((TOTAL_POINTS, 3))
m_TANGENT = np.zeros((TOTAL_POINTS - 1, 3))
m_NORMAL = np.zeros((TOTAL_POINTS - 2, 3))
m_BINORMAL = np.zeros((TOTAL_POINTS - 2, 3))

for i in range(TOTAL_POINTS):
  t = T_0 + i * dt
  POSITION[i, :] = np.ravel(F.subs(_t, t))

  if i > 0:
    m_TANGENT[i-1, :] = np.ravel(T.subs(_t, t))
  if i > 1:
    m_NORMAL[i-2, :] = np.ravel(N.subs(_t, t))
    m_BINORMAL[i-2, :] = np.ravel(B.subs(_t, t))

# Calculo do Triedo de Frenet
dpos = diferenca_finita(POSITION)
c_TANGENT = np.zeros((TOTAL_POINTS - 1, 3))
c_TANGENT = vetor_unitario(dpos)

for i in c_TANGENT:
  (i, np.linalg.norm(i))
dtan = diferenca_finita(c_TANGENT)
c_NORMAL = np.zeros((TOTAL_POINTS - 2, 3))
c_NORMAL = vetor_unitario(dtan)
c_BINORMAL = np.cross(c_TANGENT[1:], c_NORMAL)

fig = go.Figure()
fig.add_trace(go.Scatter3d(
                    x=POSITION[:,0],
                    y=POSITION[:,1],
                    z=POSITION[:,2],
                    mode='lines'
                  ))

fig.add_trace(go.Cone(
                    colorscale= [[0, "blue"], [1, "blue"]],
                    x=POSITION[1:,0],
                    y=POSITION[1:,1],
                    z=POSITION[1:,2],
                    u=c_TANGENT[:,0],
                    v=c_TANGENT[:,1],
                    w=c_TANGENT[:,2],
                    anchor= "tip",
                  ))

fig.add_trace(go.Cone(
                    colorscale= [[0, "red"], [1, "red"]],
                    x=POSITION[2:,0],
                    y=POSITION[2:,1],
                    z=POSITION[2:,2],
                    u=c_NORMAL[:,0],
                    v=c_NORMAL[:,1],
                    w=c_NORMAL[:,2],
                    anchor= "tip",
                  ))

fig.add_trace(go.Cone(
                    colorscale= [[0, "green"], [1, "green"]],
                    x=POSITION[2:,0],
                    y=POSITION[2:,1],
                    z=POSITION[2:,2],
                    u=c_BINORMAL[:,0],
                    v=c_BINORMAL[:,1],
                    w=c_BINORMAL[:,2],
                    anchor= "tip",
                  ))

# fig.add_trace(go.Cone(
#                     colorscale= [[0, "rgb(0, 255, 255)"], [1, "rgb(0, 255, 255)"]],
#                     x=POSITION[1:,0],
#                     y=POSITION[1:,1],
#                     z=POSITION[1:,2],
#                     u=m_TANGENT[:,0],
#                     v=m_TANGENT[:,1],
#                     w=m_TANGENT[:,2],
#                     anchor= "tip",
#                   ))

# fig.add_trace(go.Cone(
#                     colorscale= [[0, "rgb(255, 0, 255)"], [1, "rgb(255, 0, 255)"]],
#                     x=POSITION[2:,0],
#                     y=POSITION[2:,1],
#                     z=POSITION[2:,2],
#                     u=m_NORMAL[:,0],
#                     v=m_NORMAL[:,1],
#                     w=m_NORMAL[:,2],
#                     anchor= "tip",
#                   ))

# fig.add_trace(go.Cone(
#                     colorscale= [[0, "rgb(255, 255, 0)"], [1, "rgb(255, 255, 0)"]],
#                     x=POSITION[2:,0],
#                     y=POSITION[2:,1],
#                     z=POSITION[2:,2],
#                     u=m_BINORMAL[:,0],
#                     v=m_BINORMAL[:,1],
#                     w=m_BINORMAL[:,2],
#                     anchor= "tip",
#                   ))


fig.update_layout(
    width=700,
    margin=dict(r=20, l=10, b=10, t=10))

fig.show()
