import numpy
from math import sqrt

def norm(coord):
    '''renvoie les coordonnées normalisés (x/Z, y/Z) de (x,y,z)'''
    if coord[2]==0:
        return coord
    return (coord[0]/coord[2], coord[1]/coord[2], 1)

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
    
def grob(A):
    G = buchberger(A)
    G = optimisation(G)
    return G


class Polynome:
    
    def __init__(self, mat):
        self.coefs = []
        if isinstance(mat, dict):
            if len(list(mat.keys())[0]) == 2:
                h, l = max(mat)[0], max(mat, key = lambda x: x[1])[1]
                mat2 = nouv_coords = [[0]*(l + 1) for i in range(h + 1)]
                for a, b in mat.items():
                    mat2[a[0]][a[1]] = b[0]/b[1]
                mat = mat2
            elif len(list(mat.keys())[0]) == 3:
                h, l, m = max(mat)[0], max(mat, key = lambda x: x[1])[1], max(mat, key = lambda x: x[2])[2]
                mat2 = nouv_coords = [[[0]*(m + 1) for j in range(l + 1)] for i in range(h + 1)]
                for a, b in mat.items():
                    mat2[a[0]][a[1]][a[2]] = b[0]/b[1]
                mat = mat2
        mat = list(mat)
        while mat != [] and mat[-1] == 0: mat.pop()
        for coef in mat:
            if isinstance(coef, numpy.float64):
                self.coefs.append(coef)
            elif hasattr(coef, '__getitem__'):
                self.coefs.append(Polynome(coef))
            else:
                self.coefs.append(coef)

    def __getitem__(self, ind):
        if isinstance(ind, slice):
            return self.coefs[ind]
        if ind < len(self.coefs):
            return self.coefs[ind]
        return 0
    
    def __setitem__(self, ind, value):
        while ind >= len(self.coefs):
            self.coefs.append(0)
        self.coefs[ind] = value
    
    def __iter__(self):
        return iter(self.coefs)
    
    def __repr__(self):
        return str(self.coefs)
    
    def __add__(self, other):
        if other == 0: return self
        if isinstance(other, int):
            return Polynome([self[0]+other] + self[1:])
        if not isinstance(other, Polynome):
            other = Polynome((other,))
        i, a, b, l = 0, 1, 1, []
        while i <= max(len(self.coefs), len(other.coefs)):
            a, b = self[i], other[i]
            l.append(a + b)
            i += 1
        return Polynome(l)
        
    def __neg__(self):
        raise Exception(f'Négation non autorisée pour les polynomes. Polynome en question : {self}')
        
    def __div__(self, other):
        raise Exception(f'Division non autorisée pour les polynomes. Polynome en question : {self}')
        
    def __sub__(self, other):
        return self + (-1)*other
    
    def __rsub__(self, other):
        return (-1)*self + other
    
    def __mul__(self, other):
        if isinstance(other, (int, float, complex, numpy.float64)):
            return Polynome([coef*other for coef in self])
        if isinstance(other, Polynome):
            return Polynome([sum([self[j]*other[i-j] for j in range(i + 1)]) for i in range(self.deg + other.deg + 1)])
        raise Exception(f'Multiplication non autorisée pour les {type(other)}. Objet en question : {other}')

    def __pow__(self, other):
        if other == 0:
            return Polynome((1,))
        if other == 1:
            return self
        if isinstance(other, int) and other > 1:
            return self ** (other - 1) * self

    def __call__(self, arg):
        if self.coefs == []: return 0
        if arg == float('inf'):
            return float('inf')*self.coef_domin
        if arg == -float('inf'):
            if self.deg % 2 == 0:
                return float('inf')*self.coef_domin
            return -float('inf')*self.coef_domin
        image = self.coefs[-1]
        for coef in self.coefs[-2::-1]:
            image *= arg
            image += coef
        return image
    
    def substitution(self, P):
        '''x -> P(x)'''
        return sum([Polynome([c*coef for c in P**i]) for i, coef in enumerate(self)])
    
    def coefficients(self):
        return self.coefs
    
    @property
    def coef_domin(self):
        return self[self.deg]
    
    @property
    def deg(self):
        return max(e + self[e].deg if isinstance(self[e], Polynome) else e for e, coef in enumerate(self.coefs)) if self.coefs != [] else 0
    
    def composante_homogene(self, d):
        return Polynome({(i, d - i):(self[i][d - i], 1) for i in range(d + 1)})
    
    def change_variables(self):
        '''P(x,y) -> P(y,x), P(x,y,z) -> P(y,x,z)''' 
        for e in range(self.deg + 1):
            if not isinstance(self[e], Polynome):
                self[e] = Polynome((self[e],))
        return Polynome([Polynome([coef[e] for coef in self]) for e in range(max(poly.deg for poly in self) + 1)])
            
    def change_variables3(self):
        '''P(x,y,z) -> P(x,z,y)'''
        for e in range(self.deg + 1):
            if not isinstance(self[e], Polynome):
                self[e] = Polynome((self[e],))
        return Polynome([coef.change_variables() for coef in self])
    
    def change_variables32(self):
        '''P(x,y,z) -> P(z,x,y)'''
        return self.change_variables3().change_variables()
        
    def derivee(self):
        return Polynome([e*self[e] for e in range(1, self.deg + 1)])
    
    def resoudre(self):
        if self.deg == 1:
            return [-self[0]/self[1]], []
        if self.deg == 2:
            a, b, c = self[2], self[1], self[0]
            det = b**2 - 4*a*c
            if det < 0:
                return [], [-b/(2*a)]
            if det == 0:
                return [-b/(2*a)], [-b/(2*a)]
            if det > 0:
                return [-(b+sqrt(det))/(2*a), -(b-sqrt(det))/(2*a)], [-b/(2*a)]
        return [x.real for x in numpy.roots(self.coefs[::-1]) if abs(x.imag)<1e-16], []
        derivee = self.derivee()
        maximas, max_derivee = derivee.resoudre()
        maximas = [-float('inf')] + maximas + [float('inf')]
        images = [self(m) for m in maximas]
        intervalles = []
        for i in range(len(images)-1):
            if images[i]*images[i+1] <= 0:
                intervalles.append((maximas[i], maximas[i+1]))
        solutions = []
        for inter in intervalles:
            if inter[0] == -float('inf') and inter[1] == float('inf'):
                deb = 0
            elif inter[0] == -float('inf'):
                deb = inter[1] - 1
            elif inter[1] == float('inf'):
                deb = inter[0] + 1
            else:
                for m in max_derivee:
                    if inter[0] < m < inter[1]:
                        break
                deb = m
            x = deb
            while True:
                d = self(x) / derivee(x)
                if d < 1e-15:
                    break
                x -= d
                if not inter[0] <= x <= inter[1]:
                    print('aie')
                    x = deb
                    inf, sup = inter[0], inter[1]
                    if inf != -float('inf') and sup != float('inf'):
                        for i in range(100):
                            if self(x)*self(inf) <= 0:
                                x = (x + inf)/2
                                sup = x
                            else:
                                x, inf = (x + sup)/2, x
            solutions.append(x)
        return solutions, maximas

    def expr_rationals(self, variables, join = 1):
        liste = []
        for e in range(self.deg + 1):
            if isinstance(self[e], Polynome):
                txt = self[e].expr_rationals(variables[1:], 0)
                liste.append('+'.join(map(lambda x: (f'{variables[0]}**{e}*' if e != 0 else '') + x, txt)))
            else:
                p, q = self[e].as_integer_ratio()
                liste.append((f'{variables[0]}**{e}*' if e != 0 else '') + f'{p}/{q}')
        return '+'.join(liste) if join else liste
    
    def expr_dict_monomes(self):
        l = {}
        for i, poly in enumerate(self):
            for j, coef in enumerate(poly):
                p, q = coef.as_integer_ratio()
                l[(i, j)] = (p, q)
        return l
            
    def parametrisation(self, point):
        a, b = norm(point)[:2]
        coords = self.substitution(Polynome((a, 1)))
        coords = coords.change_variables()
        coords = coords.substitution(Polynome((b, 1)))
        coords = coords.change_variables()
        g1, g2 = coords.composante_homogene(1)(1), coords.composante_homogene(2)(1)
        p1, p2 = a*g2 - g1, b*g2 - g1*Polynome((0, 1))
        return p1, p2, g2
    
    __radd__ = __add__
    __rmul__ = __mul__
    cv = change_variables
    
    
def resoudre_systeme(P, Q):
    p, q = P.expr_dict_monomes(), Q.expr_dict_monomes()
    print('gröbner en cours')
    base = grob([p, q])
    print('fin')
    p1 = Polynome([base[0][(0, e)][0]/base[0][(0, e)][1] if (0,e) in base[0] else 0 for e in range(len(base[0]))])
    racines = p1.resoudre()[0]
    l = []
    for y in racines:
        p2 = Polynome(base[1]).change_variables()(y)
        for x in p2.resoudre()[0]:
            l.append((x, y, 1))
    l.sort(key = lambda v: abs(P(v[0])(v[1]))+abs(Q(v[0])(v[1])))
    return l[:len(l)//2]


