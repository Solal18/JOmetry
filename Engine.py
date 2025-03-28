import numpy
from math import floor, sqrt, exp, cos, sin, pi, atan2
import time
from random import random
import socket
from groebner import Polynome, resoudre_systeme
from time import perf_counter_ns as perf
import threading
from dataclasses import dataclass

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
    if x is None:
        return '!R!'
    raise ValueError

def crea_id(x, sens = 0, plan = None):
    if isinstance(x, (tuple, list)):
        return [crea_id(e, sens, plan) for e in x]
    if isinstance(x, dict):
        return {crea_id(c, sens, plan):crea_id(v, sens, plan) for c, v in x.items()}
    if isinstance(x, Creature) and sens == 1:
        return x.ide
    if isinstance(x, ID) and sens == -1:
        return plan.objets[x]
    return x

def val(x, objets = None):
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
            case 'R': return None
    return ValueError

def get_set(s):
    for e in s: return e

plan_default = 0

def dist(a, b):
    if a[0].imag==a[1].imag== b[0].imag==b[1].imag ==0:
        return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    return float("inf")

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
    x,y,z = m1
    k,l,m = m2
    a,b,c = k
    d,e,f = l
    g,h,i = m
    return [a*x+b*y+c*z, d*x+e*y+f*z, g*x+h*y+i*z]

def translation(p, v):
    a, b, c = p
    x1, y1, z1 = norm(v[0])
    x2, y2, z2 = norm(v[1])
    return (a + (x2-x1)*c, b + (y2-y1)*c, c)

def translater(args, v):
    nouv_args = []
    for i in args:
        if i[0] == 'Point':
            nouv_args.append(('Point', translation(i[1], v)))
        elif i[0] == 'Droite':
            nouv_args.append(i)
            #à modifier
        else:
            nouv_args.append(i)
    return nouv_args

def homothetie(p, c, rapport):
    x, y, z = c
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

def circonscrit(a,b,c):
    return inter(media(a,b), media(b,c))

def orthocentre(a,b,c):
    return inter(perp(inter(b,c),a), perp(inter(a,c),b))

def inscrit(a,b,c):
    return centreInscrit(a,b,c)

def gravite(a,b,c):
    return inter(inter(a, milieu(b,c)), inter(b, milieu(a,c)))
    
def fermat(a,b,c):
    A = inter(inter(a, rotation(b,c, pi/3)), inter(b, rotation(c,a, pi/3)))
    B = inter(inter(a, rotation(b,c, -pi/3)), inter(b, rotation(c,a, -pi/3)))
    x1,y1,z1 = inter(b,c)
    x2,y2,z2 = inter(c,a)
    x3,y3,z3 = inter(a,b)
    if ((A[0]*x1+A[1]*y1+A[2]*z1)*(a[0]*x1+a[1]*y1+a[2]*z1) <= 0 and
        (A[0]*x2+A[1]*y2+A[2]*z2)*(b[0]*x2+b[1]*y2+b[2]*z2) <= 0 and 
        (A[0]*x3+A[1]*y3+A[2]*z3)*(c[0]*x3+c[1]*y3+c[2]*z3) <= 0):
        return A
    return B

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
    nouv_args = []
    if not isinstance(B, tuple):
        B = B.coords()
    a, b, c = B
    if b != 0:
        for i in args:
            if i[0] == 'Point':
                d1 = Creature(plan=plan_default, classe="Droite", method='coord', args = B, u= 0)
                x, y, z = Creature(plan=plan_default, classe="Droite", method = 'translation', args = (d1, ((0, -c/b, 1), (0, 0,1))), u = 0).coords()
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
    p,q,r,s = liste
    a,b,c = p
    d,e,f = q
    g,h,i = r
    x,y,z = multi_matrix(s, inverse_m([[a,d,g], [b,e,h], [c,f,i]]))
    return [[a*x, d*y, g*z], [b*x, e*y, h*z], [c*x, f*y, i*z]]

def projective(A, liste1, liste2):
    return multi_matrix(A, multi_matrix2(transfo_p(liste2), inverse_m(transfo_p(liste1))))

def inversion(p, centre, r, UV = None):
    if isinstance(r, tuple):
        r = CAtan2(2, inter(UV[0].coord, centre), inter(UV[1].coord, centre), r, UV[0].coord, UV[1].coord)
    if centre == p:
        return (1, 1, 0)
    d = inter(p, centre)
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
        nouv_args.extend([('Point', c)]*(deg-dico[UV[0]] - dico[UV[1]]))
        nouv_args.extend([('Point', UV[0])]*(deg-dico[UV[0]] - dico[c]))
        nouv_args.extend([('Point', UV[1])]*(deg-dico[UV[1]] - dico[c]))
        return 'Courbe', 'interpol', floor(sqrt(2*lignes([tuple(i[1]) for i in nouv_args])+9/4)-3/2), nouv_args
    print("Sos il se passe un truc sus dans inversion")
    return classe, method,deg, nouv_args

transformation = {'translation' : translater, 'rotation' : rotater, 'homothetie' : homotheter, 'symetrie' : symetrer, 'projective' : projective, 'inversion' : inverser}

dico_binom = {(0, 0): 1}
def binom(n, k):
    '''calcule en programmation dynamique k parmi n'''
    if (n, k) in dico_binom:
        return dico_binom[(n, k)]
    if n < k: return 0
    if k == 0: return 1
    return binom(n-1, k-1) + binom(n-1, k)

permut2=[[2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 1, 0], [0, 1, 1], [1, 0, 1]]

def norm(coord):
    '''renvoie les coordonnées normalisés (x/Z, y/Z) de (x,y,z)'''
    if coord[2]==0:
        return coord
    return (coord[0]/coord[2], coord[1]/coord[2], 1)


def determinant(M):
    '''Determinant de la matrice M avec methode du pivot de Gauss
    Utilise numpy avec une méthode peu précise pour éviter
    des calculs trop longs si len(M) > 36'''
    if len(M) > 36:
        sign, det_log = numpy.linalg.slogdet(M)
        return sign * exp(det_log)
    M = [row[:] for row in M]
    n, sign, previous_diagonal = len(M), 1, 1
    for i in range(n-1):
        if M[i][i] == 0:
            # Echange cette ligne avec une autre
            # ayant un premier element different de 0
            for j in range(i+1, n):
                if M[j][i] != 0:
                    M[i], M[j], sign = M[j], M[i], -sign
                    break
            else:
                # Tous les M[*][i] valent zero ==> det(M) = 0
                return 0
        for j in range(i+1, n):
            for k in range(i+1, n):
                M[j][k] = M[j][k] * M[i][i] - M[j][i] * M[i][k]
                M[j][k] /= previous_diagonal
        previous_diagonal = M[i][i]
    return sign * M[-1][-1]
    
    
def permutations(n):
    liste=[]
    for i in range(0, n+1):
        for j in range(0, n+1-i):
            liste.append([n-i-j, j, i])
    return liste

class Arbre:

    def __init__(self, args, objet):
        self.descendants = set()
        self.valeur = objet
        self.parents = set()
        for i in args:
            if isinstance(i, Creature):
                self.parents.add(i)
                i.arbre.descendants.add(self)

    def descente(self, a = None, n = 0):
        if a is None: a = set()
        a.add((self, n))
        for i in self.descendants - a:
            a |= i.descente(a, n+1)
        return a
        
    def supprimer(self):
        for i in self.parents:
            i.arbre.descendants.remove(self)

################################################################################
###                        classe Relation                                   ###
################################################################################   

@dataclass
class Relation:
    parent: set
    enfants: set
    deg: int = 2

    def __str__(self):
        return f'{[i.nom for i in self.parent]} : {[i.nom for i in self.enfants]}'

    def __hash__(self):
        return id(self)
################################################################################
###                        classe Créature                                   ###
################################################################################   

    
class Creature:

    def __init__(self, plan, classe, nom = '', method = '', args = None, deg = 1, color = "red", vis = 1, u = 0, complexe = False, ide = None):
        self.plan = plan
        if args is None: args = []
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
                deg = floor(sqrt(2*lignes(args)+9/4)-3/2)
            else:
                deg = floor(sqrt(2*lignes(args)+9/4)-3/2)
                args = args[:(deg**2+3*deg)//2]
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
        self.relation_parent = set()
        self.relation_enfant = set()
        self.complexe = complexe
        self.tkinter = [None, None] #[cercle, texte] pour les points
        if ide is None:
            self.ide = plan.nouv_ide()
        elif ide not in plan.objets:
            self.ide = ide
        else: raise ValueError('Ide déjà utilisé')
        plan.objets[self.ide] = self
        plan.noms.append(nom)
        plan.modifs = (True, True)
        listes = {'Point' : plan.points, 'Droite' : plan.droites, 'Courbe' : plan.CAs}
        if classe in listes:
            listes[classe][nom] = (self)
        if self.nom != '':
            self.arbre = Arbre(args, self)
        if self.plan.main is not None and plan.main.editeur_objets is not None:
            plan.main.editeur_objets.ajouter(self)
        else:
            pass
        self.dessin()
        if nom not in {'U', 'V', 'Inf'}:
            self.coords()
        self.relation()
        try:
            for j in [self]+list(self.args):
                try:
                    print(j.nom)
                    for i in j.relation_parent:
                        print(i)
                    for i in j.relation_enfant:
                        print(i)
                except:
                    pass
        except:
            for i in self.relation_parent:
                print(i)
            for i in self.relation_enfant:
                print(i)
        print(f'nouveau {self.classe} {nom} avec méthode {method}, arguments {args}')        

    def __str__(self):
        return f'Creature classe : {self.classe}|{self.classe_actuelle} nom : {self.nom}'
    
    __repr__ = __str__

    def __hash__(self):
        return id(self)
    
    def bougeable(self):
        return (self.ide > 2 and (self.arbre.parents == set() or self.method in ('PsurCA', 'ProjOrtho')))
    
    @property
    def infos_user(self):
        return {'nom':self.nom, 'col':self.color, 'vis':self.vis}

    def copain(self):
        return {i for i in list(self.relation_parent)[0].enfants}
    
    def supprimer(self, canvas = None):
        '''fonction recursive pour supprimer des elements
        un peu bizarre pour selectionner un element d'un ensemble,
        mais le plus rapide, j'ai vérifié'''
        if self.plan.main is not None:
            if self.plan.main.editeur_objets:
                self.plan.main.editeur_objets.supprimer_element(self)
        while self.arbre.descendants:
            e = get_set(self.arbre.descendants)
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
      
    def relation(self):
        '''relation_parent : un objet défini une relation'''
        '''relation_enfant : ensemble des relations auxquelles j'appartiens'''
        '''au début j'ai 0 relation'''
        method = self.method
        args = self.args
        deg = self.deg
        if method in {"inter","interpol", "cubic", "harmonique", "milieu"}:
            s = args[0].relation_enfant
            for i in args[1:]:
                s &= i.relation_enfant
            done = False
            for i in list(s):
                if i.deg == deg:
                    if method == "inter":
                        i.parent.add(self)
                        self.relation_parent |= {i}
                    else:
                        i.enfants.add(self)
                        self.relation_enfant |= {i}
                        if i.parent != set():
                            self.relation_parent = {Relation(parent = {self}, enfants = {get_set(i.parent)}, deg = deg)}
                    done = True
            if not done:
                if method in {'inter', 'interpol'}:
                    a = Relation(parent = {self}, enfants = {i for i in args}, deg = deg)
                    self.relation_parent |= {a}
                    for i in args:
                        if isinstance(i, Creature):
                            i.relation_enfant |= {a}
                            if i.classe == 'Point' and self.classe in {'Droite', 'Courbe'}:
                                get_set(i.relation_parent).enfants.add(self)
                            if i.classe == 'Droite' and self.classe == 'Point':
                                get_set(i.relation_parent).enfants.add(self)
                else:
                    b = Relation(parent = set(), enfants = {i for i in args}, deg = deg)
                    self.relation_enfant |= {b}
                    for i in args:
                        i.relation_enfant |= {b}
            if method in {'inter', 'interpol', 'cubic'}:
                for i in args:
                    if isinstance(i, Creature):
                        self.relation_enfant |= {get_set(i.relation_parent)}
        elif method == 'inter2':
            self.relation_parent = {Relation(parent={self}, enfants={args[0], args[1]}, deg=deg)}
            self.relation_enfant = {Relation(parent=set(), enfants=set(), deg=0)}
            get_set(args[0].relation_parent).enfants.add(self)
            get_set(args[1].relation_parent).enfants.add(self)
        elif method in {'tangente', 'tangente2'}:
            get_set(args[1].relation_parent).enfants.add(self)
            self.relation_enfant |=  {get_set(args[1].relation_parent)}
        elif method in {"ProjOrtho", "PsurCA"}:
            get_set(args[0].relation_parent).enfants.add(self)
            self.relation_enfant |= {get_set(args[0].relation_parent)}
            self.relation_parent = {Relation(parent={self}, enfants={args[0]}, deg=deg)}
        if self.classe in {'Point', 'Droite'} and self.relation_parent == set():
            self.relation_parent = {Relation(parent={self}, enfants=set(), deg=deg)}

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
            if self.classe == "Point" and method =="inter2":
                args += [self.args[0].copain(), self.args[1].copain()]
                self.coord= inter2(*args)
            elif self.classe_actuelle == 'Courbe':
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
        w,h = x2-x1, y2-y1
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
        
        
        if self.classe_actuelle == 'Courbe' and self.deg_actu > 1:
            coords = coords.change_variables32()(1)
            if self.deg_actu == 2:
                p1, p2, g2 = coords.parametrisation(self.args_actu[1])
                coo = [(p1(i)/g2(i), p2(i)/g2(i)) for i in range(-50, 50)]
                pol = lambda x: focaliser((p1(x)/g2(x), p2(x)/g2(x)))
                racines = sorted(g2.resoudre()[0])
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

        if self.classe_actuelle == 'Droite' or (self.classe_actuelle == 'Courbe' and self.deg_actu == 1):
            if self == self.plan.inf: return
            if self.method == 'segment':
                A, B = self.args_actu
                z = can.create_line(focaliser(norm(A))[:2], focaliser(norm(B))[:2], width=self.plan.bold, fill=self.color, tag = self.ide)
            else:
                if isinstance(coords, Polynome):
                    coords = coords.change_variables32()(1)
                    coords = (coords.coefs[1][0], coords.coefs[0][1], coords.coefs[0][0])
                nor = norm(coords)
                if abs(nor[0]) <= abs(nor[1]): #pour les droites horizontales
                    z = can.create_line(focaliser((x1, (-1-x1*nor[0])/nor[1])),focaliser((w, (-1-w*nor[0])/nor[1])), width=self.plan.bold, fill=self.color, tag = self.ide)
                else:
                    z = can.create_line(focaliser(((-1-nor[1]*y1)/nor[0],y1)), focaliser(((-1 - h*nor[1])/nor[0], h)), width=self.plan.bold, fill=self.color, tag = self.ide)
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

        if self.classe_actuelle == 'Angle':
            #C'est peut etre pas tres beau comme ça mais ça marche
            a, b, c = [norm(x.coords()) for x in self.args[:3]]
            v1, v2 = (a[0]-b[0], a[1]-b[1]), (c[0]-b[0], c[1]-b[1])
            a1, a2 = atan2( *v1)*180/pi, atan2( *v2)*180/pi
            b = focaliser(b)
            z = can.create_arc(b[0]-15, b[1]-15, b[0]+15, b[1]+15, fill = self.color, tag = self.ide, start = a1 - 90, extent = (a2 - a1)%360)
            a, b, c = focaliser(a), b, focaliser(c)
            a, b, c = [(x[0], x[1], 1) for x in (a,b,c)]
            biss = bissectrice(a, b, c)
            per = perp(biss, b)
            cer = cercle(2, b, (b[0], b[1]+30, 1), self.plan.U.coords(), self.plan.V.coords())
            p1, p2 = inter2(biss, cer, -1)
            if sum([x*y for x,y in zip(per,p1)])*sum([x*y for x,y in zip(per,focaliser(a))]) >= 0:
                p = p1
            else: p = p2
            x, y = norm(p)[:2]
            k = can.create_text(x, y, text = f'{self.nom}={round(coords, 1)}°', font = "Helvetica4", tag = self.ide)
            self.tkinter[0] = z
            self.tkinter[1] = k
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
    a = birapport(inf(inter(A, B)), inf(inter(B, C)), U, V)
    return (atan2(a.imag, a.real)*180/(2*pi))%180

def bissectrice(a, b, c):
    return inter(b, centreInscrit(a, b, c))
    
def birapport(A, B, C, D):
    return determinant([[0,0,1], list(A), list(C)])*determinant([[0,0,1], list(B), list(D)])/(determinant([[0,0,1], list(A), list(D)])*determinant([[0,0,1], list(B), list(C)]))

def harmonique(A, B, C):
    liste = [A, B, (14,11,1), (3,4,1)]
    liste2 =[(-1, 0, 1), (1, 0,1), (14,11,1), (3,4,1)]
    x,y,z = norm(projective(C, liste, liste2))
    return projective((1/x, 0, 1), liste2, liste)
    
def inter2(courbe1, courbe2, numero, copains1 = None, copains2 = None, z = 1):
    '''Calcul des points à l'intersection de courbe1 et courbe2, non compris dans
    copains1 & copains2. Différents cas : si une des courbes est une droite,
    substitue dans l'autre polynôme et résouds. Sinon, applique l'algorithme 
    de buchberger (beaucoup plus long).'''
    if copains1 is None: copains1 = set()
    if copains2 is None: copains2 = set()
    coordonnees = (0,0,0)
    racines = []
    droite = None
    if isinstance(courbe1, (tuple, list)):
        a, b, c = courbe1
        if a != 0:
            droite, courbe = Polynome([-c/a, -b/a]), courbe2.change_variables32()(z)
        else:
            y = -c/b
            courbe = courbe2.change_variables32()(z).change_variables()
            racines = [(x, y, 1) for x in courbe(y).resoudre()[0]]
    if isinstance(courbe2, (tuple, list)):
        a, b, c = courbe2
        if a != 0:
            droite, courbe = Polynome([-c/a, -b/a]), courbe1.change_variables32()(z)
        else:
            y = -c/b
            courbe = courbe1.change_variables32()(z).change_variables()
            racines = [(x, y, 1) for x in courbe(y).resoudre()[0]]
    if droite is None and racines == []:
        courbe1 = courbe1.change_variables32()(z)
        courbe2 = courbe2.change_variables32()(z)
        racines = resoudre_systeme(courbe1, courbe2)
    elif racines == []:
        P = courbe(droite)
        for y in P.resoudre()[0]:
            racines.append((droite(y), y, 1))
    root2 = {}
    for i in copains1 & copains2:
        if i.coord[0].imag == i.coord[1].imag == 0:
            if racines != []:
                root2[racines.index(min(racines, key=lambda x : dist(norm(x), norm(i.coords()))))] = i
                racines[racines.index(min(racines, key=lambda x : dist(norm(x), norm(i.coords()))))] = (0,0,0)
    if numero == -1:
        return racines + list(root2.values())
    if numero < len(racines):
        return racines[numero]
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
    A=(norm(A)[0], norm(A)[1], 1)
    B=(norm(B)[0], norm(B)[1], 1)
    C=(norm(C)[0], norm(C)[1], 1)
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
    '''calcule les coordonées de la bissectrice des droites a et b
    numéro peut valoir 1 ou -1, et permet de calculer la bissectrice
    interieure ou exterieure'''
    xa, ya, za = a
    xb, yb, zb = b
    return (xa*sqrt(xb**2+yb**2) - numero*xb*sqrt(xa**2+ya**2),
            ya*sqrt(xb**2+yb**2) - numero*yb*sqrt(xa**2+ya**2),
            za*sqrt(xb**2+yb**2) - numero*zb*sqrt(xa**2+ya**2))

def media(A, B):
    '''retourne la médiatrice de A et B'''
    d, p = inter(A, B), milieu(A, B)
    return perp(d, p)

def tangente(C, p):
    '''Calcule la tangente en p à la courbe C, 
    et p n'est pas obligé de se trouver sur C'''
    if isinstance(C, (tuple, list)):
        return C
    a, b, c = p
    polynomex = C.change_variables()
    polynomey = C
    polynomez = C.change_variables32()
    coef1 = polynomex(b).derivee()(a)(c)
    coef2 = polynomey(a).derivee()(b)(c)
    coef3 = polynomez.derivee()(c)(a)(b)
    coords_droite = (coef1, coef2, - (coef1*a + coef2*b + coef3*(c - 1)))
    return coords_droite

def tangente2(C, p, numero):
    '''Calcule les tangentes à la courbe C passant par p'''
    if isinstance(C, (tuple, list)):
        return C
    a, b, c = p
    coef1 = C.derivee().change_variables32()(1)
    coef2 = C.change_variables().derivee().change_variables().change_variables32()(1)
    Poly2 = coef1*a+coef2*b+((-1)*coef1*Polynome((0,1))+(-1)*(coef2.change_variables()*Polynome((0,1))).change_variables())*c
    return tangente(C, inter2(Poly2, C,numero))

def cubic(deg, *args):
    '''Interpolation'''
    deg=3
    detConi = []
    args = [tuple(i) for i in args]
    permut = permutations(deg)
    print("ezaj")
    for i in args[2:]:
        detConibis=[]
        a, b, c = i
        for j in permut:
            detConibis.append(a**j[0]*b**j[1]*c**j[2])
        detConi.append(detConibis)
    for i in range(2):
        detConibis=[]
        a,b,c=args[2+i]
        x,y,z=args[0+i]
        if c==0:
            for j in permut:
                if j[2]==0 and j[1]==0:
                    detConibis.append(0)
                elif j[2]==0:
                    detConibis.append(-100*z*j[1]*a**j[0]*b**(j[1]-1)*c**j[2])
                elif j[1]==0:
                    detConibis.append(100*y*j[2]*c**(j[2]-1)*b**j[1]*a**j[0])
                else:
                    detConibis.append(100*y*j[2]*c**(j[2]-1)*b**j[1]*a**j[0]-100*z*j[1]*a**j[0]*b**(j[1]-1)*c**j[2])
        else:
            for j in permut:
                if j[0]==0 and j[1]==0:
                    detConibis.append(0)
                elif j[0]==0:
                    detConibis.append(10**10*(-x*j[1]*a**j[0]*b**(j[1]-1)*c**j[2]))
                elif j[1]==0:
                    detConibis.append(10**10*(y*j[0]*a**(j[0]-1)*b**j[1]*c**j[2]))
                else:
                    detConibis.append(10**10*(y*j[0]*a**(j[0]-1)*b**j[1]*c**j[2]-x*j[1]*a**j[0]*b**(j[1]-1)*c**j[2]))
        detConi.append(detConibis)
    print("eozake")
    print(len(detConi))
    print(detConi)
    print(detConi[-5])
    print(detConi[-3])
    print(detConi[-2])
    if deg <=7:
        a = deg + 3
    else:
        a = 2*deg - 5
    print(detConi)
    return en_poly(detConi, deg)

def interpol(deg, *args):#INTERpolation
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

def PsurCA(C, coo):
    x, y = coo[:2]
    print(coo)
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
    return (x, y, 1)

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

    def action(self, cat, *args, x = 0, **kwargs):
        '''Creature, Supprimer, Modif'''
        print(f'action : de type {cat}, args : {args}, kwargs : {kwargs}')
        if cat == 'Creature':
            r = Creature(self, *args, **kwargs)
            ca = ('Supprimer', r)
            if self.main is not None and 'fantome' in self.main.liste_derniers_clics:
                self.main.liste_derniers_clics[self.main.liste_derniers_clics.index('fantome')] = c
                if c.classe == 'Point':
                    self.main.canvas.itemconfigure(c.tkinter[0], fill = 'orange')  
        if cat == 'Supprimer':
            c = args[0]
            ca = ('Creature', c.classe, c.nom, c.method, c.args, c.deg, c.color, c.vis, c.u, c.complexe, int(c.ide))
            if self.main is None:
                r = c.supprimer()
            else:
                r = c.supprimer(self.main.canvas)
        if cat == 'Modif':
            ca = ('Modif', args[0], args[0].infos_user)
            r = args[0].set_param(**kwargs)
        if cat == 'Move':
            ca = ('Move', args[0], crea_id(args[0].args), 1)
            r = self.move(*args)
        if cat == 'Undo':
            act = crea_id(self.ctrl_z.pop(), -1, self)
            self.ctrl_y.append(self.action( *act, x = 1))
            return
        if cat == 'Redo':
            act = crea_id(self.ctrl_y.pop(), -1, self)
            self.ctrl_z.append(self.action( *act, x = 1))
            return
        if x:
            return ca
        else:
            self.ctrl_y = []
            self.ctrl_z.append(crea_id(ca, 1))
            return r 
    
    def envoi(self, cat, *args, **kwargs):
        self.serveur.send(('JOmetry ' + txt([cat, args, kwargs])).encode('utf-8'))
    
    def connecter_serveur(self, adresse, port, mdp, fich = 1):
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        t = perf()
        connecte = 0
        while perf() - t < 10000000000:
            try:
                client.connect((adresse, port))
                connecte = 1
                break
            except ConnectionRefusedError:
                print("echec durant l'etablissement de la connexion")
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
            return print('mauvais mot de passe')
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
        cat, args, kwargs = a
        self.action(cat, *args, **kwargs)
    
    def nouveau_nom(self, u = 1, classe = 'Point'):
        lettre, chiffre = 0, 0
        nom = 'A'
        dep = {'Point':65, 'Droite':97, 'Courbe':97, 'Angle':945}[classe]
        while nom in self.noms:
            lettre += 1
            chiffre += lettre//26
            lettre = lettre%26
            nom = ('' if u else '_') + chr(dep + lettre) + (str(chiffre) if chiffre else '')
        return nom
    
    def nouv_ide(self):
        ide = 0
        while 1:
            if ide not in self.objets: return ID(ide)
            ide += 1

    def fichier(self):
        return txt(((self.nom, self.notes, self.offset_x, self.offset_y),
                    [[o.ide, o.classe, o.nom, o.method, o.args, o.deg, o.color, o.vis, o.u, o.complexe] for o in self.objets.values()]))
    
    def ouvrir(self, texte):
        while self.objets:
            list(self.objets.values())[0].supprimer()
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
    
    def move(self, point, coords, dessin = 0):
        objets = set()
        if point.bougeable():
            point.args = coords
            for i in sorted(list(point.arbre.descente()), key=lambda x: x[1]):
                i[0].valeur.coords(1)
                objets.add(i[0].valeur)
            self.modifs = (True, True)
        if dessin:
            for obj in objets:
                obj.dessin(1)
        return objets
