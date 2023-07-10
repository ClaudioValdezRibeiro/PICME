from sympy import diff, Matrix, symbols, sin, cos, simplify

_u, _v = symbols("u v")

def get_christofell(X):
  global _u, _v
  Xu = diff(X, _u)
  Xv = diff(X, _v)

  E = Xu.dot(Xu)
  Eu = diff(E, _u)
  Ev = diff(E, _v)

  F = Xu.dot(Xv)
  Fu = diff(F, _u)
  Fv = diff(F, _v)

  G = Xv.dot(Xv)
  Gu = diff(G, _u)
  Gv = diff(G, _v)

  k = E*G - F**2

  # Gamma_ij^k
  GAMMA111 = simplify((G*Eu - 2*F*Fu + F*Ev) / (2*k))
  GAMMA112 = simplify((2*E*Fu - E*Ev - F*Eu) / (2*k))
  GAMMA121 = simplify((G*Ev - F*Gu) / (2*k))
  GAMMA122 = simplify((E*Gu - F*Ev) / (2*k))
  GAMMA221 = simplify((2*G*Fv - G*Gu - F*Gv) / (2*k))
  GAMMA222 = simplify((E*Gv - 2*F*Fv + F*Gu) / (2*k))

  return GAMMA111, GAMMA121, GAMMA221, GAMMA112, GAMMA122, GAMMA222


PARABOLOIDE_HIPERBOLICO = Matrix([_u, _v, _u**2 - _v**2])
ESFERA = Matrix([sin(_u)*cos(_v), sin(_u)*sin(_v), cos(_u)])
PLANO = Matrix([_u, _v, 0])

print(*get_christofell(PARABOLOIDE_HIPERBOLICO),sep='\t')
print(*get_christofell(ESFERA),sep='\t')
print(*get_christofell(PLANO),sep='\t')
