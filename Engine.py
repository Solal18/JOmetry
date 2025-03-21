import numpy
from math import floor, sqrt, exp, cos, sin, pi
import time
from random import random
from sympy import Rational, groebner, Poly
import sympy.abc
import socket
from groebner import grob
from time import perf_counter_ns as perf
import threading

class ID(int): pass

def txt(x):
    '''transforme une valeur en chaîne de caractères
    pour la sauvegarde dans un fichier'''
    if isinstance(x, str): return f"!T{x.replace('!','!!')}!"
    if isinstance(x, bool):
        return f'!B{1 if x else 0}!'
    if isinstance(x, (int, float)):
        return f'!N{x}!'
    if isinstance(x, complex):
        return f'!I{x.real}+{x.imag}!'
    if isinstance(x, Creature):
        return f"!C{x.ide}!"
    if isinstance(x, (tuple, list)):
        return '[' + ','.join(map(txt, x)) + ']'
    if isinstance(x, dict):
        return '{' + ','.join([f'{txt(a)}:{txt(x[a])}' for a in x]) + '}'
    raise BaseException and GeneratorExit and KeyboardInterrupt and SystemExit and Exception and ArithmeticError and FloatingPointError and OverflowError and ZeroDivisionError and AssertionError and AttributeError and BufferError and EOFError and ImportError and ModuleNotFoundError and LookupError and IndexError and KeyError and MemoryError and NameError and UnboundLocalError and OSError and BlockingIOError and ChildProcessError and ConnectionError and BrokenPipeError and ConnectionAbortedError and ConnectionRefusedError and ConnectionResetError and FileExistsError and FileNotFoundError and InterruptedError and IsADirectoryError and NotADirectoryError and PermissionError and ProcessLookupError and TimeoutError and ReferenceError and RuntimeError and NotImplementedError and RecursionError and StopAsyncIteration and StopIteration and SyntaxError and IndentationError and TabError and PruneError and SystemError and TypeError and ValueError and UnicodeError and UnicodeDecodeError and UnicodeEncodeError and UnicodeTranslateError and Warning and BytesWarning and DeprecationWarning and EncodingWarning and FutureWarning and ImportWarning and PendingDeprecationWarning and ResourceWarning and RuntimeWarning and SyntaxWarning and UnicodeWarning and UserWarning 

#à executer avant toute modification :
#txt({Philemon : 34})

texte = "[!TCreature!,[!TPoint!!!],{!Tnom!:!N1!,!Tmethod!:!Tcoord!,!Targs!:[[!N306.0!,!N104.0!,!N1!]],!Tu!:!N1!}]"

def val(x, objets = None):
    print(x)
    if (x[0], x[-1]) in (('(',')'), ('[',']'), ('{','}')):
        l, t, ec, n, v = [], x[1:-1], '', 0, 0
        while t:
            if v:
                if t[0] == '!':
                    if len(t) == 1 or t[1] != '!':
                        v = 0
                        ec += t[0]
                        t = t[1:] 
                    else:
                        ec += '!!'
                        t = t[2:]  
                else:
                    ec += t[0]
                    t = t[1:] 
            else:
                if t[0] in (',',':') and n == 0:
                    l.append(ec)
                    ec = ''
                elif t[0] == '!':
                    v = 1
                    ec += t[0]
                else:
                    if t[0] in ('[','(','{'):
                        n += 1
                    if t[0] in (']',')','}'):
                        n -= 1
                    ec += t[0]
                t = t[1:]
        l.append(ec)
        if n != 0: raise ValueError
        if (x[0], x[-1]) == ('{','}'):
            return {val(l[2*i], objets):val(l[2*i+1], objets) for i in range(len(l)//2)}
        return [val(i, objets) for i in l]
    if (x[0], x[-1]) == ('!','!'):
        t, typ = x[2:-1], x[1]
        match typ:
            case 'C':
                if objets:
                    return objets[int(t)]
                else:
                    return ID(t)
            case 'I':
                r, i = t.split('+')
                return float(r)+1j*float(i)
            case 'N':
                if '.' in t:
                    return float(t)
                return int(t)
            case 'T':
                 return t.replace('!!','!')
            case 'B':
                return bool(int(t))
    return ValueError

def xrint(*args):
    pass 

plan_default = 0

def dist(a, b):
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

'''def scale_m(k, m1):
    for i in range(len(m1[0])):
        m1[0][1] *= k
    return m1

def inverse_m(m):
    x,y,z = m
    a,b,c=x
    d,e,f=y
    g,h,i=z
    return scale_m(1/(a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)), [[e*i-f*h, c*h-b*i, b*f-c*e], [f*g-d*i, a*i-c*g, c*d-a*f], [d*h-e*g, b*g-a*h, a*e-b*d]])
'''

def inverse_m(m):
    return numpy.linalg.inv(m)

def multi_matrix2(m1, m2):
    resultat =[[0,0,0], [0,0,0], [0,0,0]]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                resultat[i][j] += m1[i][k]*m2[k][j]
    return resultat

def multi_matrix(m1, m2):
    '''matrice 3-3 fois matrice 1-1'''
    xrint(m1)
    xrint(m2)
    x,y,z = m1
    k,l,m = m2
    a,b,c=k
    d,e,f=l
    g,h,i=m
    return [a*x+b*y+c*z, d*x+e*y+f*z, g*x+h*y+i*z]

def translation(p, v):
    a, b, c = p
    x1, y1, z1 = norm(v[0])
    x2, y2, z2 = norm(v[1])
    return (a + (x2-x1)*c, b + (y2-y1)*c, c)

def translater(args, v):
    nouv_args = []
    print("translater")
    for i in args:
        if i[0] == 'Point':
            print(i[0])
            nouv_args.append(('Point', translation(i[1], v)))
        elif i[0] == 'Droite':
            nouv_args.append(i)
            #à modifier
        else:
            nouv_args.append(i)
    return nouv_args

def homothetie(p, c, rapport):
    x,y,z = c
    k = [[rapport, 0, 0], [0, rapport, 0], [0, 0, 1]]
    return translation(multi_matrix(translation(p, (c, (0,0,z))), k), ((0,0,z), c))

def homotheter(args, p, rapport):
    nouv_args = []
    if not isinstance(p, tuple):
        p = p.coords()
    for i in args:
        if i[0] == 'Point':
            nouv_args.append(('Point', homothetie(i[1], p, rapport)))
        elif i[0] == 'Droite':
            nouv_args.append(i)
            #à modifier
        else:
            nouv_args.append(i)
    return nouv_args

def rotation(p, c, theta):
    a, b, c = c
    k = [[cos(theta), sin(theta), 0], [sin(-theta), cos(-theta), 0], [0, 0, 1]]
    return translation(multi_matrix(translation(p, ((a,b,c), (0,0,c))), k), ((0,0,c), (a,b,c)))

def rotater(args, p, theta):
    nouv_args = []
    if not isinstance(p, tuple):
        p = p.coords()
    for i in args:
        if i[0] == 'Point':
            nouv_args.append(('Point', rotation(i[1], p, theta)))
        elif i[0] == 'Droite':
            nouv_args.append(('Droite', rotater(i[1], p, theta)))
        else:
            nouv_args.append(i)
    return nouv_args

def symetrie(A,B):
    return symetrer([('Point', A)], B)[0][1]

def symetrer(args, B):
    nouv_args =[]
    if type(B) is not tuple:
        B = B.coords()
    a,b, c = B
    if b!=0:
        for i in args:
            if i[0] == 'Point':
                d1 = Creature(plan=plan_default,classe="Droite", method='coord', args = B, u= 0)
                x,y, z = Creature(plan=plan_default, classe="Droite", method = 'translation', args = (d1, ((0, -c/b, 1), (0, 0,1))), u = 0).coords()
                k = [[(y**2-x**2)/(x**2+y**2) , -2*y*x/(x**2+y**2), 0], [-2*x*y/(x**2+y**2),(x**2-y**2)/(x**2+y**2), 0], [0, 0, 1]]
                nouv_args.append(('Point', translation(multi_matrix(translation(i[1], ((0, -c/b, 1), (0, 0, 1))), k),((0,0,1), (0, -c/b, 1)))))
    elif a!=0:
        for i in args:
            if i[0] == 'Point':
                d1 = Creature(plan=plan_default,classe="Droite", method='coord', args = B, u= 0)
                x,y, z = Creature(plan=plan_default, classe="Droite", method = 'translation', args = (d1, (c/a,0, 1)), u = 0).coords()
                k = [[(y**2-x**2)/(x**2+y**2) , -2*y*x/(x**2+y**2), 0], [-2*x*y/(x**2+y**2),(x**2-y**2)/(x**2+y**2), 0], [0, 0, 1]]
                nouv_args.append(('Point',translation(multi_matrix(translation(i[1],(c/a, 0, 1)), k),(-c/a, 0, 1))))
    return nouv_args

def transfo_p(liste):
    '''envoie liste sur [1,0,0] et les autres'''
    p,q,r,s= liste
    a,b,c = p
    d,e,f = q
    g,h,i = r
    x,y,z = multi_matrix(s,inverse_m([[a,d,g],[b,e,h],[c,f,i]]))
    return [[a*x, d*y, g*z], [b*x, e*y, h*z], [c*x, f*y, i*z]]

def projective(A, liste1, liste2):
    return multi_matrix(A,multi_matrix2(transfo_p(liste2),inverse_m(transfo_p(liste1))))

def inversion(p, centre, r, UV = None):
    if isinstance(r, tuple):
        r = CAtan2(2, inter(UV[0].coord, centre), inter(UV[1].coord, centre), r, UV[0].coord, UV[1].coord)
    if centre == p:
        return (1, 1, 0)
    d = inter(p, centre)
    print(p, r, centre, d)
    A, B = inter2(d, r, -1)
    return harmonique(A, B, p)

def inverser(classe, method, deg, args, UV, c, r):
    nouv_args = []
    if not isinstance(c, tuple):
        c = c.coords()
    if not isinstance(r, tuple):
        r = r.coords()
    cercl = CAtan2(2, inter(UV[0], c), inter(UV[1], c), r, UV[0], UV[1])
    if classe == 'Droite':
        A, B = args[0][1], args[1][1]
        d = globals()[method](A, B)
        a, b, h = d
        e, f, g = c
        if abs(a*e+b*f+h*g) < 1e-13:
            return classe, method, deg, args
        if method == 'inter':
            return 'Courbe', 'interpol', 2, [('Point', inversion(A, c, cercl)), ('Point', inversion(B, c, cercl)), ('Point', c), ('Point', UV[0]), ('Point', UV[1])]
        else:
            return 'Courbe', 'CAtan1', 2, [('Droite', inter(c, inf(d))), ('Point', c), ('Point', UV[0]), ('Point', UV[1]), ('Point', inversion(B, c, cercl))]
    if deg >= 1:
        dico = {UV[0] : 0, UV[1] : 0, c : 0}
        nouv_args = []
        for i in args:
            if i[1] in list(dico.keys()):
                dico[i[1]] +=1
            else:
                nouv_args.append(('Point', inversion(i[1], c, cercl)))
        nouv_args+= [('Point', c)]*(deg-dico[UV[0]] - dico[UV[1]])
        nouv_args+= [('Point', UV[0])]*(deg-dico[UV[0]] - dico[c])
        nouv_args+= [('Point', UV[1])]*(deg-dico[UV[1]] - dico[c])
        return 'Courbe', 'interpol', floor(sqrt(2*lignes([tuple(i[1]) for i in nouv_args])+9/4)-3/2), nouv_args
    print("Sos il se passe un truc sus dans inversion")
    return classe, method,deg, nouv_args

transformation = {'translation' : translater, 'rotation' : rotater, 'homothetie' : homotheter, 'symetrie' : symetrer, 'projective' : projective, 'inversion' : inverser}

dico_binom = {(0, 0): 1}
def binom(n, k):
    if (n, k) in dico_binom:
        return dico_binom[(n, k)]
    if n < k: return 0
    if k == 0: return 1
    return binom(n-1, k-1) + binom(n-1, k)

def petit(arr):
    n=len(arr)
    smallest = (numpy.inf,0) 
    for i in range(n):
        if(arr[i] < smallest[0]):
            smallest = (arr[i], i)
    return smallest[1]

permut2=[[2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 1, 0], [0, 1, 1], [1, 0, 1]]

def norm(coord):#renvoie les coordonnées normalisés (x/Z, y/Z) de (x,y,z)
    if coord[2]==0:
        return coord
    return (coord[0]/coord[2], coord[1]/coord[2], 1)

def find_eq_homogene(coords, deg):
    permut = permutations(deg)
    stre=""
    for i in range(len(permut)):
        stre+= "x"+"**"+str(permut[i][0])+"*"+"y" + "**"+ str(permut[i][1]) + "*" + str(Rational(coords[i]))+"+"
    stre=stre[:len(stre)-1]
    return stre

def find_eq_courbe(coords, deg, mieux="x", passemuraille_mh=""):
    xrint('cooooooooooooooordonées :',coords)
    coefs = ['']*(deg+1)
    permut = permutations(deg)
    if passemuraille_mh == "passemuraille":
        permut = permut2
    if mieux == "x":
        for i in range(len(permut)):
            coefs[deg-permut[i][0]] += "+" + " "+ "y" + "**"+ str(permut[i][1]) + "*" + str(coords[i])
    else:
        for i in range(len(permut)):
            coefs[deg-permut[i][1]] += "+" + " "+ "x" + "**"+ str(permut[i][0]) + "*" + str(coords[i])
    xrint(mieux, coefs)
    return coefs

def determinant(M):
    '''Determinant de la matrice M
    Utilise numpy avec une méthode peu précise pour éviter
    des calculs trop longs si len(M) > 36'''
    if len(M) <= 36:
        M = [row[:] for row in M]
        n, sign, previous_diagonal = len(M), 1, 1
        for i in range(n-1):
            if M[i][i] == 0:
                # Swap this row with another row having non-zero i-th element
                for j in range(i+1, n):
                    if M[j][i] != 0:
                        M[i], M[j], sign = M[j], M[i], -sign
                        break
                else:
                    # All M[*][i] are zero ==> det(M) = 0
                    return 0
            for j in range(i+1, n):
                for k in range(i+1, n):
                    M[j][k] = M[j][k] * M[i][i] - M[j][i] * M[i][k]
                    M[j][k] /= previous_diagonal
            previous_diagonal = M[i][i]
        return sign * M[-1][-1]
    else:
        sign, det_log = numpy.linalg.slogdet(M)
        return sign * exp(det_log)
    
def permutations(n):
    liste=[]
    for i in range(0, n+1):
        for j in range(0, n+1-i):
            liste.append([n-i-j, j, i])
    return liste

def resoudre(self):
    roots=[]
    for i in numpy.roots(self):
        if numpy.imag(i)==0:
            roots.append(float(numpy.real(i)))
    return (roots, [])

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
    
    '''def resoudre(self):
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
                if d < 1e-14:
                    break
                x -= d
                if not inter[0] <= x <= inter[1]:
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
        return solutions, maximas'''

    def resoudre(self):
        roots=[]
        for i in numpy.roots(self.coefficients()[::-1]):
            if numpy.imag(i)==0:
                roots.append(float(numpy.real(i)))
        return (roots, [])

    def expr_rationals(self, variables, join = 1):
        liste = []
        for e in range(self.deg + 1):
            if isinstance(self[e], Polynome):
                txt = self[e].expr_rationals(variables[1:], 0)
                liste.append('+'.join(map(lambda x: (f'{variables[0]}**{e}*' if e != 0 else '') + x, txt)))
            else:
                liste.append((f'{variables[0]}**{e}*' if e != 0 else '') + str(Rational(self[e])))
        return '+'.join(liste) if join else liste
    
    def expr_dict_monomes(self):
        l = {}
        for i, poly in enumerate(self):
            for j, coef in enumerate(poly):
                r = Rational(coef)
                l[(i, j)] = (r.p, r.q)
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
    

class Arbre:

    def __init__(self, args, objet):
        self.descendants = set()
        self.valeur = objet
        self.parents = set()
        for i in args:
            if isinstance(i, Creature):
                self.parents.add(i)
                i.arbre.descendants.add(self)

    def descente(self, a = set(), n = 0):
        a.add((self, n))
        for i in self.descendants - a:
            a |= i.descente(a, n+1)
        return a
        
    def supprimer(self):
        for i in self.parents:
            i.arbre.descendants.remove(self)


################################################################################
###                        classe Créature                                   ###
################################################################################   

    
class Creature:

    def __init__(self, plan, classe, nom = '', method = '', args = None, deg = 1, color = 'green', vis = 1, u = 0, complexe =False):
        self.plan = plan
        if args is None: args = []
        print('C', deg, args)
        if nom in (0, 1):
            nom = plan.nouveau_nom(nom, classe)
        if classe == 'Courbe':
            if method in ('CAtan1', 'CAtan2', 'cercle'):
                deg = 2
            elif method in transformation:
                deg = args[0].deg 
            elif deg == '':
                while 2*lignes(args) != floor(sqrt(2*lignes(args)+9/4)-3/2)**2+3*floor(sqrt(2*lignes(args)+9/4)-3/2):
                    args.pop(-1)
                deg= floor(sqrt(2*lignes(args)+9/4)-3/2)
            else:
                deg= floor(sqrt(2*lignes(args)+9/4)-3/2)
                args = args[:(deg**2+3*deg)//2]
        print('C',deg, args)
        self.nom = nom
        self.coord = None
        self.method = method
        self.classe = classe
        self.classe_actuelle = None
        self.deg_actu = None
        self.args = args
        self.deg = deg
        self.color = color
        self.vis = vis
        self.u = u
        self.complexe = complexe
        self.tkinter = [None, None] #[cercle, texte] pour les points
        self.ide = self.plan.nouv_ide()
        plan.objets[self.ide] = self
        plan.noms.append(nom)
        plan.modifs = (True, True)
        listes = {'Point' : plan.points, 'Droite' : plan.droites, 'Courbe' : plan.CAs}
        if classe in listes:
            listes[classe][nom] = (self)
        if self.nom != '':
            self.arbre = Arbre(args, self)
        print('C',deg, args)
        if self.plan.main is not None:
            if plan.main.editeur_objets:
                plan.main.editeur_objets.ajouter(self)
            if nom not in ('U', 'V', 'Inf'):
                plan.action_utilisateur(f'creation de {nom}')
                #plan.contre_action(self.supprimer, (self.plan.main.canvas,))
        else:
            pass
        print('C',deg, args)
        self.dessin()
        xrint(self.coord)
        print(f'nouveau {self.classe} {nom} avec méthode {method}, arguments {args}')
        print('C',deg, args)
        

    def __str__(self):
        return f'Creature classe : {self.classe}|{self.classe_actuelle} nom : {self.nom}'
    
    __repr__ = __str__

    def __hash__(self):
        return id(self)
    
    def bougeable(self):
        return (self.ide > 2 and (self.arbre.parents == set() or self.method in ('PsurCA', 'ProjOrtho')))

    def supprimer(self, canvas = None):
        '''fonction recursive pour supprimer des elements
        un peu bizarre pour selectionner un element d'un ensemble,
        mais le plus rapide, j'ai vérifié'''
        #self.plan.contre_action(Creature, (self.plan, self.classe, self.nom, self.method, self.args, self.deg, self.color, self.u, self.vis))
        if self.plan.main is not None:
            if self.plan.main.editeur_objets:
                self.plan.main.editeur_objets.supprimer_element(self)
        while self.arbre.descendants:
            for e in self.arbre.descendants:
                break
            e.valeur.supprimer(canvas)
        self.arbre.supprimer()
        for i in self.tkinter:
            if i and canvas:
                canvas.delete(i)
                self.plan.tkinter_object.pop(i)
        for dic in (self.plan.points, self.plan.droites, self.plan.CAs, self.plan.objets):
            if self.ide in dic:
                del dic[self.ide]
        self.plan.noms.remove(self.nom)
        del self
                
    def coords(self, calcul = 0):
        if self.coord is None or calcul:
            method = self.method
            objet = self
            transformations = []
            while method in transformation and not objet.classe == 'Point':
                parent = objet.args[0]
                transformations.append((method, objet.args[1:]))
                method = parent.method
                objet = parent
            args = [(i.classe, i.coords()) if isinstance(i, Creature) else (0, i) for i in objet.args]
            classe = self.classe
            U, V = self.plan.U.coord, self.plan.V.coord
            deg = self.deg
            while transformations:
                method_tr, args_tr = transformations.pop()
                if method_tr == "inversion":
                    classe, method, deg, args = transformation["inversion"](self.classe, method, self.deg, args, (self.plan.U.coord, self.plan.V.coord), *args_tr)
                    if deg == 1:
                        classe = 'Droite'
                        method = "inter"
                else:
                    args = transformation[method_tr](args, *args_tr)
            self.classe_actuelle = classe
            self.deg_actu = deg
            args = [i[1] for i in args]
            self.args_actu = args
            if self.classe_actuelle == 'Courbe':
                self.coord = globals()[method](deg, *args)
            elif self.classe_actuelle == 'Droite' and method == 'inter':
                self.coord = inter(*args)
            else:
                self.coord = globals()[method](*args)
        if self.complexe and self.coord[2] != 0:
            self.coord = ((self.coord[0]/self.coord[2]).real, (self.coord[1]/self.coord[2]).real, 1) 
        return self.coord
            
    def set_param(self, nom = None, col = None, vis = None):
        if nom is None: nom = self.nom
        if col is None: col = self.color
        if vis is None: vis = self.vis
        self.plan.contre_action(self.set_param, (self.nom, self.color, self.vis))
        self.plan.noms.remove(self.nom)
        self.plan.noms.append(nom)
        if self.plan.main is None: return
        edit = self.plan.main.editeur_objets
        if edit is not None:
            for item in edit.tableau.get_children():
                ligne = edit.tableau.item(item)['values']
                if ligne and ligne[0] == self.nom:
                    ligne[0] = nom
                    ligne[4] = col
                    ligne[5] = ['non', 'oui'][vis]
                    edit.tableau.item(item, values = ligne)
        self.nom = nom
        self.color = col
        self.vis = vis
        self.dessin()
        
    def dessin(self, calcul = 1):
        if self.plan.main is None or self.plan is not self.plan.main.plans[0]: return
        
        can = self.plan.main.canvas
        for i in self.tkinter:
            can.delete(i)
        
        if not (self.u and self.vis): return
        if self.classe_actuelle == 'non': return
        
        h, w = can.winfo_height(), can.winfo_width()
        defocaliser = self.plan.main.coord_canvas
        (x1, y1), (x2, y2) = defocaliser(0, 0), defocaliser(w, h)
        self.tkinter=[None, None]
        
        
        def focaliser(coordN): #renvoie le focalisé du point (qui est gentil) coordN par foc
            return (self.plan.offset_x[0]*coordN[0] + self.plan.offset_x[1], self.plan.offset_y[0]*coordN[1]+self.plan.offset_y[1])
        
        coords = self.coords() if (calcul or self.coord is None) else self.coord
        
        def dessin_entre(p1, p2, g2, inf, sup, a, b, i = 0, infi = 0):
            if abs(a[0] - b[0])+abs(a[1] - b[1]) <= 7 or i >= 20:
                z = can.create_line(a[0], a[1], b[0], b[1], width = self.plan.boldP, fill = self.color, tag = self.ide)
                self.tkinter.append(z)
                self.plan.tkinter_object[z]=self
            else:
                if infi:
                    if sup >= -inf:
                        bo = 10*inf
                    else:
                        bo = 10*sup
                else:
                    bo = (inf+sup)/2
                p = focaliser((p1(bo)/g2(bo), p2(bo)/g2(bo)))
                dessin_entre(p1, p2, g2, inf, bo, a, p, i+1, infi)
                dessin_entre(p1, p2, g2, bo, sup, p, b, i+1, infi)
        
        xrint(f"on dessine l'objet {self}")
        if self.classe_actuelle == 'Courbe' and self.deg_actu > 1:
            print(f'{self.nom}: {self.method}')
            print(self.coords())
            print(self.args_actu)
            xrint("Calcul des points.")
            zzzz=time.time()
            coords = coords.change_variables32()(1)
            if self.deg_actu == 2:
                p1, p2, g2 = coords.parametrisation(self.args_actu[1])
                coo = [(p1(i)/g2(i), p2(i)/g2(i)) for i in range(-50, 50)]
                pol = lambda x: focaliser((p1(x)/g2(x), p2(x)/g2(x)))
                racines = sorted(g2.resoudre()[0])
                print('racines du polynome dans la par parametrisation :', racines)
                if len(racines) == 2:
                    r0, r1, r2, r3, r4, r5 = racines[0]-10, racines[0]-1e-2, racines[0]+1e-2, racines[1]-1e-2, racines[1]+1e-2, racines[1]+10
                    dessin_entre(p1, p2, g2, r0, r5, pol(r0), pol(r5), infi = 1)
                    dessin_entre(p1, p2, g2, r0, r1, pol(r0), pol(r1))
                    dessin_entre(p1, p2, g2, r2, r3, pol(r2), pol(r3))
                    dessin_entre(p1, p2, g2, r4, r5, pol(r4), pol(r5))
                else:
                    p_m50, p_m0, p_0, p_50 = pol(50), focaliser((p1(-1e-10)/g2(-1e-10), p2(-1e-10)/g2(-1e-10))), focaliser((p1(1e-10)/g2(1e-10), p2(1e-10)/g2(1e-10))), focaliser((p1(50)/g2(50), p2(50)/g2(50)))
                    dessin_entre(p1, p2, g2, 1e-10, 50, p_0, p_50)
                    dessin_entre(p1, p2, g2, -50, -1e-10, p_m50, p_m0)                  
            else:
                self.plan.CAst[self.ide]=[]
                polynomex = coords.change_variables()
                polynomey = coords
                
                i = x1
                while i<x2:
                    polynome2y = polynomey(i)
                    roots = polynome2y.resoudre()[0]
                    l_y = []
                    for y in roots:
                        if y1 - 50 <= y <= y2 + 50:
                            l_y.append((i, y))
                    self.plan.CAst[self.ide].append(l_y)
                    i += 1
                print(f'Fin calcul des points. Temps estimé : {time.time()-zzzz}')
                xrint("Début affichage des points")
                zzzz = time.time()
                points = self.plan.CAst[self.ide]
                for x, l_p in enumerate(points[1:-1]):
                    p_moins = points[x]
                    p_plus = points[x+2]
                    for p in l_p:
                        d_moins = min([(sqrt(1+(p[1]-p_[1])**2), p_) for p_ in p_moins] + [(float('inf'), 0)])
                        d_plus = min([(sqrt(1+(p[1]-p_[1])**2), p_) for p_ in p_plus] + [(float('inf'), 0)])
                        d_nor = min([(abs(p[1]-p_[1]), p_) for p_ in l_p if p_ != p] + [(float('inf'), 0)])
                        if d_moins == d_nor == (float('inf'), 0):
                            continue
                        a_p = min([d_moins, d_nor])[1]
                        p, a_p = focaliser(p), focaliser(a_p)
                        if dist(p, a_p) < 50:
                            z=can.create_line(p[0], p[1], a_p[0], a_p[1], width = self.plan.boldP, fill = self.color, tag = self.ide)
                            self.tkinter.append(z)
                            self.plan.tkinter_object[z]=self
            can.tag_lower(self.ide, 'limite2')
            print(can.find_withtag(self.ide))
            print(f'Fin affichage des points. Temps estimé : {time.time()-zzzz}.')

        if self.classe_actuelle == 'Droite' or (self.classe_actuelle == 'Courbe' and self.deg_actu == 1):
            if self == self.plan.inf: return
            if self.method == 'segment':
                A, B = self.args_actu
                z = can.create_line(focaliser(A)[:2], focaliser(B)[:2], width=self.plan.bold, fill=self.color, tag = self.ide)
            else:
                if isinstance(coords, Polynome):
                    print("salut")
                    coords = (coords.coefs[1][0], coords.coefs[0][1], coords.coefs[0][0])
                    print(coords)
                nor = norm(coords)
                if abs(nor[0]) <= abs(nor[1]): #pour les droites horizontales
                    z = can.create_line(focaliser((0, (-1/nor[1]))),focaliser((w, (-1-w*nor[0])/nor[1])), width=self.plan.bold, fill=self.color, tag = self.ide)
                else:
                    z = can.create_line(focaliser((-1/nor[0],0)), focaliser(((-1 - h*nor[1])/nor[0], h)), width=self.plan.bold, fill=self.color, tag = self.ide)
            self.tkinter[0] = z
            self.plan.tkinter_object[z] = self
            can.tag_lower(z, 'limite1')
            

        if self.classe_actuelle == 'Point':
            a = coords
            if not(a[0].imag == 0 and a[1].imag == 0 and a[2] != 0): return
            a = (a[0]/a[2], a[1]/a[2],1)
            c = focaliser([a[0], a[1]])
            k = can.create_text(c[0], c[1], text = '•', font = "Helvetica " + str(self.plan.boldP*8), fill = self.color, tag = self.ide)
            z = can.create_text(c[0] + self.plan.boldP*8, c[1], text = self.nom, font = "Helvetica " + str(self.plan.boldP*6), tag = self.ide)
            self.tkinter[1] = z
            self.tkinter[0] = k
            self.plan.tkinter_object[k] = self
            self.plan.tkinter_object[z] = self
            can.tag_raise(k, 'limite2')
            can.tag_raise(z, 'limite2')

 
################################################################################
###                        Méthodes de calcul                                ###
################################################################################        


def rien(c): return c

def lignes(liste):
    dicoArgs={}
    for i in liste:
        dicoArgs[i] = dicoArgs.get(i, 0)+1
    return sum([(i*(i+1))//2 for i in list(dicoArgs.values())])

def coord(c, d = 0, e = 0):
    """Définition d'un point par ses coordonnées"""
    if isinstance(c, (tuple, list)):
        x, y, z = c
        if x==y==z==0:
            raise ValueError("Point (0,0,0) impossible")
        return (x, y, z)
    return (c, d, e)
    
def segment(A, B):
    return inter(A, B)

def inter(A, B):
    """Définition d'un point par deux droites"""
    xA, yA, zA = A
    xB, yB, zB = B
    return (yA*zB-yB*zA,
            zA*xB-zB*xA,
            xA*yB-xB*yA)

def angle(A, B, C, U, V):
    '''Calcule l'angle entre (AB) et (BC)'''
    return birapport(inf(inter(A, B)), inf(inter(B, C)), U, V)

def symetrie(A, B):
    return symetrer(A, B)

def birapport(A, B, C, D):
    return ((A-C)*(B-D))/((B-C)*(A-D))

def harmonique(A, B, C):
    liste = [A, B, (14,11,1), (3,4,1)]
    liste2 =[(-1, 0, 1), (1, 0,1), (14,11,1), (3,4,1)]
    x,y,z = norm(projective(C, liste, liste2))
    return projective((1/x, 0, 1), liste2, liste)

def inter2(courbe1, courbe2, numero, z = 1):
    coooords = (0,0,0)
    rooot = []
    droite = None
    if isinstance(courbe1, (tuple, list)):
        a, b, c = courbe1
        if a != 0:
            droite, courbe = Polynome([-c/a, -b/a]), courbe2.change_variables32()(z)
        else:
            y = -c/b
            courbe = courbe2.change_variables32()(z).change_variables()
            rooot = [(x, y, 1) for x in courbe(y).resoudre()[0]]
    if isinstance(courbe2, (tuple, list)):
        a, b, c = courbe2
        if a != 0:
            droite, courbe = Polynome([-c/a, -b/a]), courbe1.change_variables32()(z)
        else:
            y = -c/b
            courbe = courbe1.change_variables32()(z).change_variables()
            rooot = [(x, y, 1) for x in courbe(y).resoudre()[0]]
    if droite is None and rooot == []:
        courbe1 = courbe1.change_variables32()(z)
        courbe2 = courbe2.change_variables32()(z)
        p1, p2 = courbe1.expr_dict_monomes(), courbe2.expr_dict_monomes()
        #c = grob([p1, p2])
        #P2 = Polynome(c[1]).change_variables()
        #l = {d[1]:e[0]/e[1] for d, e in c[0].items()}
        #mat = []
        #for i in range(max(l) + 1):
        #    mat.append(l[i] if i in l else 0)
        #racines = Polynome(mat).resoudre()[0]
        #for r in racines:
        #    P = P2(r)
        #    for r2 in P.resoudre()[0]:
        #        xrint(courbe2(r2)(r))
        #        if courbe2(r2)(r) < 1e-14:
        #            rooot.append((r2, r, 1))
        print(p1)
        stra = ""
        for i in list(p1.keys()):
            stra+= "x**"+str(i[0])+"*y**"+str(i[1]) + "*" +str(Rational(p1[i][0]/p1[i][1]))+"+"
        strb=""
        for i in list(p2.keys()):
            strb+= "x**"+str(i[0])+"*y**"+str(i[1])+ "*" +str(Rational(p2[i][0]/p2[i][1]))+"+"
        b=groebner([stra[:-1], strb[:-1]], sympy.abc.x, sympy.abc.y)
        c=Poly(b[1]).all_coeffs()
        root=resoudre(c)[0]
        for r in root:
            k = str(b[0]).replace("y","("+ str(r)+")")
            print(k)
            autre_roots = resoudre(Poly(k).all_coeffs())[0]
            for ax in autre_roots:
                rooot.append((ax,r,1))
                print("wesh")
                print((ax,r,1))
    elif rooot == []:
        P = courbe(droite)
        for y in P.resoudre()[0]:
            rooot.append((droite(y), y, 1))
    if numero == -1:
        return rooot
    if numero < len(rooot):
        return rooot[numero]
    return (0, 0, 0)

def ortho(A):
    """Point orthogonal de A"""
    xA, yA, zA = A
    if zA != 0:
        raise ValueError("Orthogonal d'un point non à l'infini")
    return (-yA, xA, 0)

def inf(a):
    """Point à l'infini d'une droite"""
    xa, ya, za = a
    return (-ya, xa, 0)

def milieu(A, B):
    """Définition du milieu de deux points"""
    xA, yA, zA = A
    xB, yB, zB = B
    return (xA*zB+xB*zA, yA*zB+yB*zA, 2*zA*zB)

def centreInscrit(A, B, C):
    """Définition du centre du cercle inscrit de trois points"""
    xA, yA, zA = A
    xB, yB, zB = B
    xC, yC, zC = C
    a=sqrt((xB-xC)**2+(yB-yC)**2+(zB-zC)**2)
    b=sqrt((xA-xC)**2+(yA-yC)**2+(zA-zC)**2)
    c=sqrt((xB-xA)**2+(yB-yA)**2+(zB-zA)**2)
    return ((a*xA+b*xB+c*xC)/(a+b+c),
            (a*yA+b*yB+c*yC)/(a+b+c),
            (a*zA+b*zB+c*zC)/(a+b+c))

def biss(a, b, numero = 1): #numéro vaut 1 ou -1
    xa, ya, za = a
    xb, yb, zb = b
    return (
            xa*sqrt(xb**2+yb**2)-numero*xb*sqrt(xa**2+ya**2),
            ya*sqrt(xb**2+yb**2)-numero*yb*sqrt(xa**2+ya**2),
            za*sqrt(xb**2+yb**2)-numero*zb*sqrt(xa**2+ya**2))

def media(A, B):
    '''retourne la médiatrice de A et B'''
    d, p = inter(A, B), milieu(A, B)
    return perp(d, p)

def tangente(C, p):
    '''C -> CA
    a -> complexe
    b -> complexe
    Construit la tangente à C en le point (a,b)
    '''
    if isinstance(C, (tuple, list)):
        return C
    a, b,c = p
    print(p)
    print(C)
    polynomex = C.change_variables()
    polynomey = C
    polynomez = C.change_variables32()
    print('polynomezzzzz')
    print(f'{polynomez}')
    coef1 = polynomex(b).derivee()(a)(c)
    coef2 = polynomey(a).derivee()(b)(c)
    coef3 = polynomez.derivee()(c)(a)(b)
    coords_droite = (coef1, coef2, -coef1*a -coef2*b+coef3*(1-c))
    return coords_droite

def interpol(deg, *args):#INTERpolation
    xrint('Début interpolation')
    zzzz=time.time()
    detConi = []
    dicoArgs ={}
    args = [tuple(i) for i in args]
    permut = permutations(deg)
    for i in args:
        dicoArgs[i] = dicoArgs.get(i, 0)+1
    for i in list(dicoArgs.keys()):
        a, b, c = i
        detConibis = []
        for j in permut:
            detConibis.append(a**j[0]*b**j[1]*c**j[2])
        detConi.append(detConibis)
        for k in range(1,(dicoArgs[i]*(dicoArgs[i]+1))//2):
            detConibis = []
            if c==0:
                for j in permut:
                    if j[2]==0 and j[1]==0:
                        detConibis.append(0)
                    elif j[2]==0:
                        detConibis.append(10**10*((10**7-100*k)*j[1]*a**j[0]*b**(j[1]-1)*c**j[2]))
                    elif j[1]==0:
                        detConibis.append(10**10*(k*j[2]*c**(j[2]-1)*b**j[1]*a**j[0]))
                    else:
                        detConibis.append(10**10*(k*j[2]*c**(j[2]-1)*b**j[1]*a**j[0]+(10**7-100*k)*j[1]*a**j[0]*b**(j[1]-1)*c**j[2]))
            else:
                for j in permut:
                    if j[0]==0 and j[1]==0:
                        detConibis.append(0)
                    elif j[0]==0:
                        detConibis.append((10**(2*deg+1)-10**(deg)*k)*j[1]*a**j[0]*b**(j[1]-1)*c**j[2])
                    elif j[1]==0:
                        detConibis.append(k*j[0]*a**(j[0]-1)*b**j[1]*c**j[2])
                    else:
                        detConibis.append(k*j[0]*a**(j[0]-1)*b**j[1]*c**j[2]+(10**(2*deg+1)-10**(deg)*k)*j[1]*a**j[0]*b**(j[1]-1)*c**j[2])
            detConi.append(detConibis)
    print("wooo")
    print(detConi)
    if deg <=7:
        a = deg + 3
    else:
        a = 2*deg - 5
    for i in range(len(detConi)):
        for j in range(len(detConi[i])):
            detConi[i][j] = detConi[i][j]/(10**a)
    return en_poly(detConi, deg)

def en_poly(detConi, deg):
    permut = permutations(deg)
    coords = []
    for i in range(len(permut)):
        sousDet = [[detConi[j][k] for k in range(i)]+[detConi[j][k] for k in range(i+1, len(permut))] for j in range(len(detConi))]
        coords.append(((-1)**(i+1) * determinant(sousDet), (permut[i])))
    nouv_coords = {}
    a = 0
    i = 0
    while a == 0 and i < len(coords):
        if coords[i][0] != 0:
            a = coords[i][0]
        i+=1
    if a != 0:
        for j in range(len(coords)):
            nouv_coords[tuple(coords[j][1])] = ((coords[j][0] / a).real, 1)
    poly = Polynome(nouv_coords)
    return poly

def CAtan1(deg, *args):
    '''crée une conique tangente à d1 en A passant par les points B, C et D'''
    d1, A, B, C, D = args[:5]
    b, c = d1[1], d1[2]
    detConi =  [[A[0]**2, A[0]*A[1], A[1]**2, A[2]*A[0], A[1]*A[2], A[2]**2],
                 [0, A[0]*c, 2*c*A[1], -A[0]*b, A[2]*c-A[1]*b, -2*b*A[2]],
                 [B[0]**2,B[0]*B[1], B[1]**2, B[0]*B[2], B[1]*B[2], B[2]**2],
                 [C[0]**2,C[0]*C[1], C[1]**2, C[0]*C[2], C[1]*C[2], C[2]**2],
                 [D[0]**2,D[0]*D[1], D[1]**2, D[0]*D[2], D[1]*D[2], D[2]**2]]
    return en_poly(detConi, deg)

def CAtan2(deg, *args):
    ''' crée une conique tangente à d1 en U et à d2 en V passant par le point B
    permet notamment de faire un cercle de centre c1 si confondu avec c2'''
    d1, d2 = args[0], args[1]
    U, V = args[3], args[4]
    B = args[2]
    b, c = d1[1], d1[2]
    d, e = d2[1], d2[2]
    detConi =  [[U[0]**2, U[0]*U[1], U[1]**2,U[2]*U[0], U[1]*U[2],U[2]**2],
                 [0, U[0]* c, 2*c*U[1], -U[0]*b, U[2]*c-U[1]*b, -2*b*U[2]],
                 [V[0]**2, V[0]*V[1], V[1]**2,V[2]*V[0], V[1]*V[2],V[2]**2],
                 [0, V[0]* e, 2*e*V[1], -V[0]*d, V[2]*e-V[1]*d, -2*d*V[2]],
                 [B[0]**2, B[0]*B[1],B[1]**2, B[0]*B[2], B[1]*B[2], B[2]**2]]
    #if tangente1[2] !=0 and tangente1[2] !=0:
    #detConi =  [[U.coords()[0]**2, U.coords()[0]*U.coords()[1], U.coords()[1]**2,U.coords()[2]*U.coords()[0], U.coords()[1]*U.coords()[2],U.coords()[2]**2],
    #             [2*U.coords()[0]*b,b*U.coords()[1]-a*U.coords()[0], -2*U.coords()[1]*a, U.coords()[2]*b, -U.coords()[2]*a, 0],
    #             [V.coords()[0]**2, V.coords()[0]*V.coords()[1], V.coords()[1]**2,V.coords()[2]*V.coords()[0], V.coords()[1]*V.coords()[2],V.coords()[2]**2],
    #             [2*V.coords()[0]*d,d*V.coords()[1]-c*V.coords()[0], -2*V.coords()[1]*c, V.coords()[2]*d, -V.coords()[2]*c, 0],
    #             [E.coords()[0]**2, E.coords()[0]*E.coords()[1],E.coords()[1]**2, E.coords()[0]*E.coords()[2], E.coords()[1]*E.coords()[2], E.coords()[2]**2]]
    return en_poly(detConi, deg)

def perp(d, p):
    '''args -> Droite, Point; retourne la perpendiculaire'''
    p1 = inf(d)
    p2 = ortho(p1)
    d = inter(p, p2)
    return d
    
def ProjOrtho(d, p):
    p1 = inf(d)
    p2 = ortho(p1)
    d1 = inter(p, p2)
    return inter(d, d1)

def PsurCA(C, n, coo, main):
    x,y = coo
    done = False
    C = C.change_variables32()(1)
    deriveex = C.derivee()
    deriveey = C.change_variables().derivee()
    i = 0
    while not done:
        a = x - C(x)(y)/(deriveex(x)(y)**2 + deriveey(y)(x)**2)*deriveex(x)(y)
        b = y - C(x)(y)/(deriveex(x)(y)**2 + deriveey(y)(x)**2)*deriveey(y)(x)
        i += 1
        if abs(a-x) + abs(b-y) < 1e-10 or i > 100:
            done = True
        x, y = a, b
    return (x,y,1)

def cercle(d, O, A, U, V):
    d1, d2 = inter(O, U), inter(O, V)
    return CAtan2(d, d1, d2, A, U, V)

###################################
###         Classe Plan         ###
###################################

    
class Plan:

    def __init__(self, main = None, nom = 'Plan 1', dd = None):
        self.main = main
        self.noms = []
        self.objets = {}
        self.points = {}
        self.tkinter_object = {}
        self.droites = {}
        self.CAs={}
        self.CAst={}
        self.U = Creature(self, 'Point', nom="U", method="coord", args=[(1,1j,0)])
        self.V = Creature(self, 'Point', nom="V", method="coord", args=[(1,-1j,0)])
        self.inf = Creature(self, 'Droite', nom="Inf", method="inter", args=[self.U, self.V])
        self.U.coords()
        self.V.coords()
        self.inf.coords()
        self.bold = 3 #largeur d'une droite
        self.boldP = 3 #rayon d'un point
        self.boldC = 1 #largeur points des coniques
        self.focal = ((0,0), 1) #focal par défaut
        self.offset_x = [1,0]
        self.offset_y = [1,0]
        self.nom = nom
        self.modifs = (False, False) #(depuis la création, depuis la dernière sauvegarde)
        self.dossier_default = dd
        self.ctrl_z = []
        self.ctrl_y = []
        self.annulation = 0
        self.derniere_action = None
        self.serveur = None
        self.notes = ''
        self.en_y = 0

    def action(self, cat, *args, **kwargs):
        '''Creature, Supprimer, Modif'''
        print(f'action : de type {cat}, args : {args}, kwargs : {kwargs}')
        if cat == 'Creature':
            c = Creature(self, *args, **kwargs)
            if self.main is not None and 'fantome' in self.main.liste_derniers_clics:
                self.main.liste_derniers_clics[self.main.liste_derniers_clics.index('fantome')] = c
                if c.classe == 'Point':
                    self.main.canvas.itemconfigure(c.tkinter[0], fill = 'orange')
            return c
        if cat == 'Supprimer':
            if self.main is None:
                return args[0].supprimer()
            else:
                return args[0].supprimer(self.main.canvas)
        if cat == 'Modif':
            return args[0].set_param(**kwargs)
        if cat == 'Move':
            return self.move(*args)
        return None 
    
    def envoi(self, cat, *args, **kwargs):
        print('envoi')
        self.serveur.send(('JOmetry ' + txt([cat, args, kwargs])).encode('utf-8'))
    
    def connecter_serveur(self, adresse, port, mdp, fich = 1):
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        t = perf()
        connecte = 0
        print((adresse, port))
        while perf() - t < 10000000000:
            try:
                client.connect((adresse, port))
                connecte = 1
                break
            except ConnectionRefusedError:
                print('raté')
        if not connecte: return 
        print('connexion du client')
        client.send(f'JOmetry 0 {mdp} {fich}'.encode('utf-8'))
        reponse = client.recv(2048).decode('utf-8')
        if reponse == 'JOmetry connecte':
            self.serveur = client
            print('authentification du client')
        else: return
        if fich:
            fichier, i = '', 0
            while fichier[-16:] != ' stop!stop!stop!' and i < 1000:
                i += 1
                fichier += client.recv(2048).decode('utf-8')
            self.ouvrir(fichier[8:-16])
        if reponse == 'JOmetry 0 non autorisé':
            return print('mauvais mdp')
        t = threading.Thread(target = self.ecoute_serveur, args = (client,))
        t.start()
        
    def ecoute_serveur(self, client):
        while True:
            reponse = client.recv(2048)
            msg = reponse.decode('utf-8')
            if len(msg) < 9 or msg[:8] != 'JOmetry ':
                continue
            msg = msg[8:]
            print(f'message recu : {msg}')
            self.decode_action(msg)
        
    def decode_action(self, msg):
        a = val(msg, self.objets)
        print('serv :', a, type(a))
        cat, args, kwargs = a
        print('decode_action a trouvé :',cat, args, kwargs)
        self.action(cat, *args, **kwargs)
    
    def nouveau_nom(self, u = 1, classe = 'Point'):
        lettre, chiffre = 0, 0
        nom = 'A'
        dep = {'Point':65, 'Droite':97, 'Courbe':65, 'Angle':945}[classe]
        while nom in self.noms:
            lettre += 1
            chiffre += lettre//26
            lettre = lettre%26
            nom = ('' if u else '_') + chr(dep + lettre) + (str(chiffre) if chiffre else '')
        return nom
    
    def nouv_ide(self):
        ide = 0
        while 1:
            if ide not in self.objets: return ide
            ide += 1

    def contre_action(self, fonc, args):
        xrint(fonc, args)
        if self.annulation:
            self.ctrl_y[-1].append((fonc, args))
        else:
            self.ctrl_z[-1].append((fonc, args))
        if self.main is not None: self.main.maj_bouton()

    def action_utilisateur(self, act):
        if (self.derniere_action == act and act is not None) or act in ('ctrlz', 'ctrly'): return
        if not self.en_y:
            self.ctrl_y = []
        if len(self.ctrl_z) == 0 or self.ctrl_z[-1] != []:
            self.ctrl_z.append([])
        if self.main is not None: self.main.maj_bouton()
        xrint('action utilisateur', act, self.ctrl_z, self.ctrl_y)
        self.derniere_action = act
            
    def ctrlz(self):
        liste = []
        while liste == []:
            liste = self.ctrl_z.pop(-1)
        self.annulation = 1
        self.ctrl_y.append([])
        for fonc, args in liste[::-1]:
            fonc(*args)
        self.annulation = 0
        if self.main is not None: self.main.maj_bouton()
        print('Y :',self.ctrl_y, '\nZ :',self.ctrl_z)
        
    def ctrly(self):
        self.ctrl_z.append([])
        liste = []
        print('Y :',self.ctrl_y)
        while liste == []:
            liste = self.ctrl_y.pop(-1)
        print('Y :',self.ctrl_y)
        self.en_y = 1
        for fonc, args in liste[::-1]:
            fonc(*args)
        self.en_y = 0
        print('Y :',self.ctrl_y)
        if self.main is not None: self.main.maj_bouton()
        print('Y :',self.ctrl_y)

    def closest_point(self, point):
        distances = []
        a,b = point[0], point[1]
        for j in self.points.values():
            try:
                k=norm(j.coords())
            except:
                k=j.coords()
            if numpy.imag(k[0])==0 and numpy.imag(k[1])==0:
                distances.append((a-k[0])**2+(b-k[1])**2)
            else:
                distances.append(numpy.inf)
        return (list(self.points.values())[petit(distances)], distances[petit(distances)])
    
    def closest_objet(self, canvas, point):
        liste = self.tkinter_object[int(str(canvas.find_closest(point[0], point[1]))[1:].split(",")[0])]
        mini=(200,0)
        for j in liste:
            if j.deg <= mini[0]:
                mini = (j.deg, j)
        return mini[1]
    
    def set_bold(self, newBold):
        if type(newBold) == type(0):
            if newBold > 0:
                self.bold = newBold
            else:
                raise TypeError("Vous avez donné une largeur de trait négative.")
        else:
            raise TypeError("Vous avez donné une largeur de trait non entière.")
        
    def set_boldP(self, newBold):
        if type(newBold) == type(0):
            if newBold > 0:
                self.boldPoint = newBold
            else:
                raise TypeError("Vous avez donné un rayon de point négative.")                
        else:
            raise TypeError("Vous avez donné une largeur de trait non entière.")

    def fichier(self):
        return txt(((self.nom, self.notes, self.offset_x, self.offset_y),
                    [[o.ide, o.classe, o.nom, o.method, o.args, o.deg, o.color, o.vis, o.u, o.complexe] for o in self.objets.values()]))
    
    def ouvrir(self, texte):
        while self.objets:
            list(self.objets.values())[0].supprimer()
        print(val(texte))
        plan, objets = val(texte)
        self.nom, self.notes, self.offset_x, self.offset_y = plan[0], plan[1], plan[2], plan[3]
        ides, parents = [o[0] for o in objets], [o[4] for o in objets]
        objets = {o[0]:o[1:]+[0] for o in objets}
        dic = {}
        for obj, args in zip(ides, parents):
            dic[obj] = ([arg for arg in args if isinstance(arg, ID)], set())
        for obj, parents in dic.items():
            for parent in parents[0]:
                dic[parent][1].add(obj)
        ordre = []
        while dic:
            s = set()
            o = next(iter(dic))
            while dic[o][0]:
                if o in s:
                    raise RecursionError
                s.add(o)
                o = dic[o][0][0]
            for enf in dic[o][1]:
                dic[enf][0].remove(o)
            del dic[o]
            ordre.append(o)
        for ide in ordre:
            l = objets[ide][:-1]
            l[3] = [objets[arg][-1] if isinstance(arg, ID) else arg for arg in l[3]]
            c = Creature(self, *l)
            objets[ide][-1] = c
    
    def move(self, point, coords):
        objets = set()
        if point.bougeable():
            print(f'On bouge {point}')
            self.contre_action(self.move, (point, point.coords()))
            if point.arbre.parents == set(): point.args=[coords]
            elif point.method == 'PsurCA': point.args[2] = coords[:2]
            else: point.args[1] = coords
            for i in sorted(list(point.arbre.descente()), key=lambda x: x[1]):
                i[0].valeur.coords(1)
                objets.add(i[0].valeur)
            self.modifs = (True, True)
        return objets
    
    def new_harmonique(self, nom, A,B,C, u = 1):
        d = Creature(self, 'Point', nom = nom, method = 'harmonique', args = (A,B,C), u = u)
        return d
    
    def new_rotation(self, nom, obj, p, angle, u = 1):
        d = Creature(self, obj.classe, nom = nom, method = 'rotation', args = (obj, p, angle), u = u)
        return d
    
    def new_homothetie(self, nom, obj, p, rapport, u = 1):
        d = Creature(self, obj.classe, nom = nom, method = 'homothetie', args = (obj, p, rapport), u = u)
        return d

    def new_translation(self, nom, obj, vecteur, u = 1):
        d = Creature(self, obj.classe, nom = nom, method = 'translation', args = (obj, vecteur), u = u)
        return d

    def new_symetrie(self, nom, obj, droite, u = 1):
        d = Creature(self, obj.classe, nom = nom, method = 'symetrie', args = (obj, droite), u = u)
        return d
    
    def new_projective(self, nom, obj, liste1, liste2, u = 1):
        d = Creature(self, obj.classe, nom = nom, method = 'projective', args =(obj, liste1, liste2), u = u)
        return d

    def newPoint_coord(self, nom, coord, u = 1):#crée un point libre avec les coordonnées suivantes
        p = Creature(self, 'Point', nom=nom, method="coord", args=[coord], u = u)
        return p

    def newPoint_objets(self, nom, methode, objet1, objet2, numero, u = 1):#crée l'intersection de deux objets qui n'est pas dans inters
        p = Creature(self, 'Point', nom = nom, method = methode, args = [objet1, objet2, numero], u = u)
        return p

    def newDroite(self, nom, args, method, u = 1):
        d = Creature(self, 'Droite', nom=nom, method=method, args=args, u = u)
        return d

    def newCA(self, nom, liste, u = 1):
        a = Creature(self, 'Courbe', nom = nom, method = 'interpol', deg = '', args = liste, u = u)
        return a

    def newCAtan(self, nom, d1, d2, point, point2, point3, u = 1):
        a = Creature(self, 'Courbe', nom = nom, method = 'CAtan2', args = [d1, d2, point, point2, point3], u = u)
        return a

    def newProjectionOrtho(self, nom, args, u = 1):
        d = self.newPerp(0, args, u = 0)
        p = Creature(self, 'Point', nom = nom, method = 'inter', args = (args[0], d), u = u)
        return p

    def newCentreInscrit(self, nom, p1, p2, p3, u = 1):
        p = Creature(self, 'Point', nom = nom, method = 'centreInscrit', args = (p1, p2, p3), u = u)
        return p

    def newPerp(self, nom, args, u = 1):
        '''args -> Droite, Point; retourne la perpendiculaire'''
        p1 = Creature(self, 'Point', 0, method = 'inf', args = (args[0],), u = 0)
        p2 = Creature(self, 'Point', 0, method = 'ortho', args = (p1,), u = 0)
        d = Creature(self, 'Droite', nom = nom, method = 'inter', args = (args[1], p2), u = u)
        return d
    
    def newMilieu(self, nom, args, u = 1):
        return Creature(self, 'Point', nom = nom, method = 'milieu', args = (args[0], args[1]), u = u)
    
    def newCentre(self, nom, args, u = 1):
        d1 = Creature(self, 'Droite', 0, method = 'tangente', args = (args[0], self.U), u = 0)
        d2 = Creature(self, 'Droite', 0, method = 'tangente', args = (args[0], self.V), u = 0)
        return Creature(self, 'Point', nom = nom, method = 'inter', args = (d1, d2), u = u, complexe = True)
    
    def newMedia(self, nom, args, u = 1):
        d1 = Creature(self, 'Droite', 0, method = 'inter', args = (args[0], args[1]), u = 0)
        p1 = Creature(self, 'Point', 0, method = 'inf', args = (d1,), u = 0)
        p2 = Creature(self, 'Point', 0, method = 'ortho', args = (p1,), u = 0)
        p3= Creature(self, 'Point', 0, method = 'milieu', args = (args[0], args[1]), u = 0)
        d = Creature(self, 'Droite', nom = nom, method = 'inter', args = (p2, p3), u = u)
        return d


    def newPsurCA(self, nom, args, u = 1):
        if args[0].classe == 'Droite':
            return Creature(self, 'Point', nom = nom, method = 'ProjOrtho', args = [args[0], (args[1][0], args[1][1], 1)], u = u)
        return Creature(self, 'Point', nom = nom, method = 'PsurCA', args = [args[0], args[0].deg, args[1], self.main], u = u)
        
    
    def newPara(self, nom, args, u = 1):
        p1 = Creature(self, 'Point', nom = self.nouveau_nom(0), method = 'inf', args = (args[0],), u = 0)
        d = Creature(self, 'Droite', nom = nom, method = 'inter', args = (args[1], p1), u = u)
        return d
