def rint(x):
    print(x)
    return x

def ppcm_lm(P, Q):
    p, q = max(P), max(Q)
    return (max(p[0], q[0]), max(p[1], q[1]))

def S_poly(P, Q):
    p, a, b = ppcm_lm(P, Q), max(P), max(Q)
    c, d = P[a], Q[b]
    e, f = (p[0] - a[0], p[1] - a[1], c[1], c[0]), (p[0] - b[0], p[1] - b[1], d[1], d[0])
    P2, Q2 = {}, {}
    for m, t in P.items():
        P2[(m[0]+e[0], m[1]+e[1])] = (t[0]*e[2], t[1]*e[3])
    for m, t in Q.items():
        Q2[(m[0]+f[0], m[1]+f[1])] = (-t[0]*f[2], t[1]*f[3])
    return addition(P2, Q2)

def pgcd(x, y):
    while y:
        x, y = y, x % y
    return x

def addition(P, Q):
    R = {}
    for m in P.copy():
        if m in Q:
            a, b = P.pop(m), Q.pop(m)
            if a[0]*b[1] + b[0]*a[1] != 0:
                n, d = a[0]*b[1] + b[0]*a[1], a[1]*b[1]
                p = pgcd(n, d)
                R[m] = (n//p, d//p)
    return R | P | Q

def division(P, A):
    mds = {}
    for Q in A:
        m = max(Q)
        R = Q.copy()
        a, b = R.pop(m)
        for i in R:
            R[i] = (-R[i][0]*b, R[i][1]*a)
        mds[m] = R
    m = 1
    while m:
        m = 0
        for md in mds:
            for mp in list(P):
                a, b = mp[0] - md[0], mp[1] - md[1]
                if a >= 0 and b >= 0:
                    c, d = P.pop(mp)
                    nP = {(a+e, b+f): (c*g, d*h) for (e, f), (g, h) in mds[md].items()}
                    P = addition(P, nP)
                    m = 1
    return P

print(division({(4, 0):(1, 1), (3, 1):(3, 1)}, [{(3, 0):(1, 1), (1, 1):(-2, 1)}]))
print(division({(4, 0):(1, 1), (3, 1):(3, 1)}, [{(4, 0):(1, 1), (3, 1):(3, 1)}]))

def buchberger(polys):
    G = polys
    paires_critiques = []
    for i in range(len(polys)):
        for j in range(i):
            paires_critiques.append((j, i))
    while paires_critiques:
        i, j = paires_critiques.pop()
        P, Q = G[i], G[j]
        S = S_poly(P, Q)
        R = division(S, G)
        if R != {} and R not in G:
            for i in range(len(G)):
                paires_critiques.append((i, len(G)))
            G.append(R)
    return G

def est_base_gröbner(G):
    for i, P in enumerate(G):
        for Q in G[:i]:
            if division(S_poly(P, Q), G) != {}:
                return False
    return True

def optimisation(G):
    une_var, deux_var = [], []
    for P in G:
        if max(c[0] for c in P)*max(c[1] for c in P) == 0:
            une_var.append(P)
        else:
            deux_var.append(P)
    for P in une_var:
        for Q in deux_var:
            if est_base_gröbner((P, Q)):
                return (P, Q)
    print('bizzare...')
    return G
    
from time import perf_counter_ns as t
z = t()
G = buchberger([{(3, 0):(1, 1), (1, 1):(-2, 1)}, {(2, 1):(1, 1), (0, 2):(-2, 1), (1, 0):(1, 1)}])
print(t() - z)
print(G)
z = t()
print(optimisation(G))
print(t() -z)
def grob(A):
    G = buchberger(A)
    G = optimisation(G)
    return G
from sympy import groebner
from sympy.abc import x, y
p1s = [{(0, 0): (4607011922895687, 1180591620717411303424), (0, 1): (-2628655315693055, 2305843009213693952), (0, 2): (-281686387051785, 2251799813685248), (1, 0): (2514568243717833, 576460752303423488), (1, 1): (-6285575001443129, 18014398509481984), (2, 0): (1, 1)}, {(0, 0): (153804764260449, 4611686018427387904), (0, 1): (362685610540071, 288230376151711744), (0, 2): (-3961153186376467, 2251799813685248), (1, 0): (478082912761703, 36028797018963968), (1, 1): (-3473243904899643, 4503599627370496), (2, 0): (1, 1)}]
p2s = ['4607011922895687/1180591620717411303424+y**1*-2628655315693055/2305843009213693952+y**2*-281686387051785/2251799813685248+x**1*2514568243717833/576460752303423488+x**1*y**1*-6285575001443129/18014398509481984+x**2*1', '153804764260449/4611686018427387904+y**1*362685610540071/288230376151711744+y**2*-3961153186376467/2251799813685248+x**1*478082912761703/36028797018963968+x**1*y**1*-3473243904899643/4503599627370496+x**2*1']
z = t()
groebner(p2s, x, y)
print(f'sympy met {(t() - z)/1000000}ms')
z = t()
grob(p1s)
print(f'mon grobner met {(t() - z)/1000000}ms')
