from sympy import diff, factor, Matrix, symbols, sympify


def get_christofell(X):
    Xu = diff(X, u)
    Xv = diff(X, v)

    E = Xu.dot(Xu)
    Eu = diff(E, u)
    Ev = diff(E, v)

    F = Xu.dot(Xv)
    Fu = diff(F, u)
    Fv = diff(F, v)

    G = Xv.dot(Xv)
    Gu = diff(G, u)
    Gv = diff(G, v)

    k = E*G - F**2

    # Gamma_ij^k
    GAMMA111 = factor((G*Eu - 2*F*Fu + F*Ev) / (2*k))
    GAMMA112 = factor((2*E*Fu - E*Ev - F*Eu) / (2*k))
    GAMMA121 = factor((G*Ev - F*Gu) / (2*k))
    GAMMA122 = factor((E*Gu - F*Ev) / (2*k))
    GAMMA221 = factor((2*G*Fv - G*Gu - F*Gv) / (2*k))
    GAMMA222 = factor((E*Gv - 2*F*Fv + F*Gu) / (2*k))

    return GAMMA111, GAMMA112, GAMMA121, GAMMA122, GAMMA221, GAMMA222


u, v = symbols("u v")

# PARABOLOIDE HIPERBOLICO
PH = Matrix([u, v, u ** 2 - v ** 2])

# SELA DE MACACO
SM = Matrix([u, v, u**3 - 3*u*v**2])

print(get_christofell(PH))

print(get_christofell(SM))
