import numpy
from math import floor, sqrt, exp, cos, sin, pi
import time
from sympy import groebner, Rational, Poly
from sympy.abc import x, y

def xrint(*args):
    #print(*args)
    return 

plan_default = 0

def dist(a, b):
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def scale_m(k, m1):
    for i in range(len(m1[0])):
        m1[0][1] *= k
    return m1

'''def inverse_m(m):
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

def transfo_p(liste):
    '''envoie liste sur [1,0,0] et les autres'''
    p,q,r,s= liste
    a,b,c = p
    d,e,f = q
    g,h,i = r
    x,y,z = multi_matrix(s,inverse_m([[a,d,g],[b,e,h],[c,f,i]]))
    return [[a*x, d*y, g*z], [b*x, e*y, h*z], [c*x, f*y, i*z]]

def transfo_proj(A, liste1, liste2):
    return multi_matrix(A,multi_matrix2(transfo_p(liste2),inverse_m(transfo_p(liste1))))

def translater(A, v):
    a, b, c = A
    tx, ty, tz = v
    return (a + tx*c, b + ty*c, tz*c)

def rotater(A, B, theta):
    theta = theta/180*pi
    if type(B) is not tuple:
        B = B.coords()
    a, b, c = B
    k = [[cos(theta), sin(theta), 0], [sin(-theta), cos(-theta), 0], [0, 0, 1]]
    return translater(multi_matrix(translater(A, (-a/c, -b/c, 1/c)), k), (a/c, b/c, c))

def homotheter(A, B, rapport):
    if type(B) is not tuple:
        B = B.coords()
    a, b, c = B
    k = [[rapport, 0, 0], [0, rapport, 0], [0, 0, 1]]
    return translater(multi_matrix(translater(A, (-a/c, -b/c, 1/c)), k), (a/c, b/c, c))

#Tu crées de nouvelles droites à chaque calcul !
def symetrer(A, B):
    if type(B) is not tuple:
        B = B.coords()
    a,b, c = B
    if b!=0:
        d1 = Droite(plan=plan_default, method='coord', args = B, u= 0)
        x,y, z = Droite(plan=plan_default, method = 'translation', args = (d1, (0, c/b, 1)), u = 0).coords()
        k = [[(y**2-x**2)/(x**2+y**2) , -2*y*x/(x**2+y**2), 0], [-2*x*y/(x**2+y**2),(x**2-y**2)/(x**2+y**2), 0], [0, 0, 1]]
        return translater(multi_matrix(translater(A, (0, c/b, 1)), k), (0, -c/b, 1))
    elif a!=0:
        d1 = Droite(plan=plan_default, method='coord', args = B, u= 0)
        x,y, z = Droite(plan=plan_default, method = 'translation', args = (d1, (c/a,0, 1)), u = 0).coords()
        k = [[(y**2-x**2)/(x**2+y**2) , -2*y*x/(x**2+y**2), 0], [-2*x*y/(x**2+y**2),(x**2-y**2)/(x**2+y**2), 0], [0, 0, 1]]
        return translater(multi_matrix(translater(A, (c/a, 0, 1)), k), (-c/a, 0, 1))    

transformation = {'translation' : translater, 'rotation' : rotater, 'homothetie' : homotheter, 'symetrie' : symetrer, 'projective' : transfo_proj}

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

def resoudre(polynome):
    roots=[]
    for i in numpy.roots(polynome):
        if numpy.imag(i)==0:
            roots.append(numpy.real(i))
    return roots

def norm(coord):#renvoie les coordonnées normalisés (x/Z, y/Z) de (x,y,z)
    return (coord[0]/coord[2], coord[1]/coord[2])

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

class Polynome:
    
    def __init__(self, mat):
        self.coefs = []
        while mat != [] and mat[-1] == 0: mat.pop()
        for coef in mat:
            if hasattr(coef, '__getitem__'):
                self.coefs.append(Polynome(coef))
            else:
                self.coefs.append(coef)
    
    def __getitem__(self, ind):
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
        if not isinstance(other, Polynome):
            other = Polynome((other,))
        i, a, b, l = 0, 1, 1, []
        while not (a == 0 and b == 0):
            a, b = self[i], other[i]
            l.append(a + b)
            i += 1
        return Polynome(l)
        
    def __sub__(self, other):
        return self + (-1)*other
    
    def __rsub__(self, other):
        return (-1)*self + other
    
    def __mul__(self, other : int):
        return Polynome([coef*other for coef in self])

    def __call__(self, arg):
        if arg == float('inf'):
            return float('inf')*self.coef_domin()
        if arg == -float('inf'):
            if self.deg() % 2 == 0:
                return float('inf')*self.coef_domin()
            return -float('inf')*self.coef_domin() 
        return sum(arg**e*coef for e, coef in enumerate(self))
    
    def coefficients(self):
        return self.coefs
    
    def coef_domin(self):
        return self[self.deg()]
    
    def deg(self):
        return max(e + self[e].deg() if isinstance(self[e], Polynome) else e for e, coef in enumerate(self.coefs))
    
    def change_variables(self):
        for e in range(self.deg() + 1):
            if not isinstance(self[e], Polynome):
                self[e] = Polynome((self[e],))
        return Polynome([Polynome([coef[e] for coef in self]) for e in range(max(poly.deg() for poly in self))])
            
    def derivee(self):
        return Polynome([e*self[e] for e in range(1, self.deg() + 1)])
    
    def resoudre(self):
        if self.deg() == 1:
            return [-self[0]/self[1]], []
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
            if inter[0] == -float('inf'):
                deb = inter[1] - 1
            elif inter[1] == float('inf'):
                deb = inter[0] + 1
            else:
                for m in max_derivee:
                    if inter[0] < m < inter[1]:
                        break
                deb = m
            x = deb
            for i in range(100):
                x = x - (self(x) / derivee(x))
                if not inter[0] < x < inter[1]:
                    print('erreur de Newton')
                    x = deb
                    break
            solutions.append(x)
        return solutions, maximas
    
    def expr_rationals(self, variables, join = 1):
        liste = []
        for e in range(self.deg() + 1):
            if isinstance(self[e], Polynome):
                txt = self[e].expr_rationals(variables[1:], 0)
                liste.append('+'.join(map(lambda x: (f'{variables[0]}**{e}*' if e != 0 else '') + x, txt)))
            else:
                liste.append((f'{variables[0]}**{e}*' if e != 0 else '') + str(Rational(self[e])))
        return '+'.join(liste) if join else liste
    
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
                i.arbre.descendants.add(objet)

    def descente(self, objet, a = set(), n = 0):
        a.add((objet, n))
        for i in objet.descendants - a:
            a |= Arbre.descente(self, i, a, n+1)
        return a
    
    def supprimer(self):
        for i in self.parents:
            i.arbre.descendants.remove(self.valeur)
    
class Creature:

    def __init__(self, plan, nom = '', method = '', args = [], deg = '', color = 'green', vis = 1, u = 0, complexe=True):
        self.plan = plan
        self.nom = nom
        self.coord = None
        self.method = method
        self.args = args
        self.deg = deg
        self.color = color
        self.complexe = complexe
        self.vis = vis
        self.u = u
        self.tkinter = [None, None] #[cercle, texte] pour les points
        plan.objets[nom] = self
        plan.noms.append(nom)
        plan.modifs = (True, True)
        if self.nom != "":
            self.arbre = Arbre(args, self)
        if plan.main.editeur_objets:
            plan.main.editeur_objets.ajouter(self)
        if nom not in ('U', 'V', 'Inf'):
            plan.contre_action(self.supprimer, (self.plan.main.canvas,))
        self.dessin()
        xrint(f'nouveau {self.classe} {nom} avec méthode {method}, arguments {args} et couleur {color}')
        

    def __str__(self):
        return self.nom

    def __hash__(self):
        return id(self)

    def supprimer(self, canvas):
        '''fonction recursive pour supprimer des elements
        un peu bizarre pour selectionner un element d'un ensemble,
        mais le plus rapide, j'ai vérifié'''
        if self.classe == 'Point':
            self.plan.contre_action(Point, (self.plan, self.nom, self.method, self.args, None, self.color, self.u, self.vis))
        if self.classe == 'Droite':
            self.plan.contre_action(Droite, (self.plan, self.nom, self.method, self.args, None, self.color, self.u, self.vis))
        if self.classe == 'Courbe':
            self.plan.contre_action(CA, (self.plan, self.nom, self.method, self.args, '', self.color, self.u, self.vis))
        if self.plan.main.editeur_objets:
            self.plan.main.editeur_objets.supprimer_element(self)
        while self.arbre.descendants:
            for e in self.arbre.descendants:
                break
            e.supprimer(canvas)
        self.arbre.supprimer()
        for i in self.tkinter:
            if i is not None:
                canvas.delete(i)
                self.plan.tkinter_object.pop(i)
        for dic in (self.plan.points, self.plan.droites, self.plan.CAs, self.plan.objets):
            if self.nom in dic:
                del dic[self.nom]
        self.plan.noms.remove(self.nom)
        del self
                

    def coords(self, calcul = 0):
        if self.coord is None and not calcul:
            method = self.method
            objet = self
            transformations = []
            while method in transformation and not isinstance(objet, Point):
                parent = objet.args[0]
                transformations.append((method, objet.args[1:]))
                method = parent.method
                objet = parent
            args = [(i.coords(), 1) if isinstance(i, Creature) else (i, 0) for i in objet.args]
            while transformations:
                method_tr, args_tr = transformations.pop()
                args = [(transformation[method_tr](i[0], *args_tr), 1) if i[1] else i for i in args]
            args = [i[0] for i in args]
            if method == 'cercle':
                self.coord = self.cercle(self.deg, *args)
            elif isinstance(self, CA):
                self.coord = self.inter(self.deg, *args)
            elif isinstance(self, Droite) and method == 'inter':
                self.coord = self.inter(*args)
            else:
                self.coord = getattr(self.__class__, method)(self, *args)
        return self.coord

    def set_coords(self):
        if isinstance(self, CA):
            self.coord = getattr(CA, self.method)(self.deg, *self.args)
        else:
            self.coord = getattr(self.__class__, self.method)(self, *self.args)
            
    def set_param(self, nom, couleur, vis):
        self.plan.contre_action(self.set_param, (self.nom, self.color, self.vis))
        self.plan.noms.remove(self.nom)
        for dic in (self.plan.points, self.plan.droites, self.plan.CAs, self.plan.objets):
            if self.nom in dic:
                del dic[self.nom]
                dic[nom] = self
        self.plan.noms.append(nom)
        edit = self.plan.main.editeur_objets
        if edit is not None:
            for item in edit.tableau.get_children():
                ligne = edit.tableau.item(item)['values']
                if ligne and ligne[0] == self.nom:
                    ligne[0] = nom
                    ligne[4] = couleur
                    ligne[5] = ['non', 'oui'][vis]
                    edit.tableau.item(item, values = ligne)
        self.nom = nom
        self.color = couleur
        self.vis = vis
        self.dessin()
        
    
    def dessin(self, calcul = 1):
        
        can = self.plan.main.canvas
        h, w = can.winfo_height(), can.winfo_width()
        for i in self.tkinter:
            can.delete(i)
        self.tkinter=[None, None]
        
        if not (self.u and self.vis): return

        def focaliser(coordN): #renvoie le focalisé du point (qui est gentil) coordN par foc
            return (self.plan.offset_x[0]*coordN[0] + self.plan.offset_x[1], self.plan.offset_y[0]*coordN[1]+self.plan.offset_y[1])
        
        
        coords = self.coords() if calcul or self.coord is None else self.coord
         
        if isinstance(self, CA) and self.deg !=1:
            xrint("Calcul des points.")
            zzzz=time.time()
            self.plan.CAst[self.nom]=[]
            polynomex = coords.change_variables()
            polynomey = coords
            i = 0

            #while i<h:
            #    for j in range(len(polynomex)):
            #        polynome2x[j] = eval(str(polynomex[j]).replace("y", str(i)))
            #    roots = resoudre(polynome2x)
            #    l_x = []
            #    for x in roots:
            #        if 0 <= x and w >= x:
            #            c = [x, i]
            #            l_x.append((x, i))
            #    self.plan.CAst[self.nom].append(l_x)
            #    i += 1
            i = 0

            while i<w:
                #for j in range(len(polynomey)):
                    #polynome2y[j] = eval(str(polynomey[j]).replace("x", str(i)))
                polynome2y = polynomey(i)
                if i == 10: print(polynome2y)
                roots = polynome2y.resoudre()[0]
                #roots = resoudre(polynome2y)
                l_y = []
                for y in roots:
                    if 0 <= y <= h:
                        c = [i, y]
                        l_y.append((i, y))
                self.plan.CAst[self.nom].append(l_y)
                i += 1
            xrint(f'Fin calcul des points. Temps estimé : {time.time()-zzzz}')
            xrint("Début affichage des points")
            zzzz = time.time()
            points = self.plan.CAst[self.nom]
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
                        z=can.create_line(p[0], p[1], a_p[0], a_p[1], width = self.plan.boldP, fill = self.color)
                        self.tkinter.append(z)
                        self.plan.tkinter_object[z]=self
            for objet in self.tkinter:
                if objet is not None: can.tag_lower(objet, 'limite2')
            xrint(f'Fin affichage des points. Temps estimé : {time.time()-zzzz}.')

        if isinstance(self, Droite) or (isinstance(self, CA) and self.deg==1):
            print(coords)
            if self == self.plan.inf: return
            nor = norm(coords)
            if abs(nor[0]) <= abs(nor[1]): #pour les droites horizontales
                z = can.create_line(focaliser((0, (-1/nor[1]))),focaliser((w, (-1-w*nor[0])/nor[1])), width=self.plan.bold, fill=self.color)
            else:
                print(focaliser((-1/nor[0],0)), focaliser(((-1 - h*nor[1])/nor[0], h)))
                z = can.create_line(focaliser((-1/nor[0],0)), focaliser(((-1 - h*nor[1])/nor[0], h)), width=self.plan.bold, fill=self.color)
            self.tkinter[0] = z
            self.plan.tkinter_object[z] = self
            can.tag_lower(z, 'limite1')

        if isinstance(self, Point):
            a = coords
            if a[0].imag == 0 and a[1].imag == 0 and a[2]!=0:
                a= (a[0]/a[2], a[1]/a[2],1)
                c=focaliser([a[0], a[1]])
                k = can.create_text(c[0], c[1], text = '•', font = "Helvetica " + str(self.plan.boldP*8), fill = self.color)
                z=can.create_text(c[0] + self.plan.boldP*8, c[1], text = self.nom, font = "Helvetica " + str(self.plan.boldP*6))
                self.tkinter[1]=z
                self.tkinter[0]=k
                self.plan.tkinter_object[k]=self
                self.plan.tkinter_object[z]=self
                can.tag_raise(k, 'limite1')
                can.tag_raise(z, 'limite1')


class Point(Creature):
    """Classe Point"""
    classe = 'Point'
    
    def __init__(self, plan, nom="", method="", args=[], objet=None, color="green", u = 0, vis = 1, complexe=True):
        if nom in (0, 1):
            nom = plan.nouveau_nom(nom, "point")
        super().__init__(plan, nom=nom, args = args, color=color, method=method, deg=1, u = u, vis = vis, complexe=complexe)
        if method == "coord" and objet is not None and type(objet) is not Point:
            self.objet = objet
        plan.points[self.nom] = self
        

    def __eq__(self, other):
        """Définition de l'égalité de deux points"""
        if type(other) != type(self):
            return False
        xA, yA, zA = self.coords()
        xB, yB, zB = other.coords()
        return yA*zB-yB*zA == zA*xB-zB*xA == xA*yB-xB*yA == 0
    
    def __hash__(self):
        return id(self)

    def coord(self, c, d=0,e=0):
        """Définition d'un point par ses coordonnées"""
        if isinstance(c, tuple):
            x, y, z = c
            if x==y==z==0:
                raise ValueError("Point (0,0,0) impossible")
            return (x, y, z)
        return (c,d,e)
    
    def inter(self, A, B):
        """Définition d'un point par deux droites"""
        xA, yA, zA = A
        xB, yB, zB = B
        return (yA*zB-yB*zA,
                zA*xB-zB*xA,
                xA*yB-xB*yA)
    
    def translation(self, A, v):
        return translater(A, v)
    
    def rotation(self, A, B, theta):
        return rotater(A, B, theta)

    def homothetie(self, A, B, rapport):
        return homotheter(A, B, rapport)
    
    def symetrie(self, A, B):
        return symetrer(A, B)

    def harmonique(self, A, B, C):
        liste = [A, B, (14,11,1), (3,4,1)]
        liste2 =[(-1, 0, 1), (1, 0,1), (14,11,1), (3,4,1)]
        x,y = norm(transfo_proj(C, liste, liste2))
        return transfo_proj((1/x, 0, 1), liste2, liste)

    def projective(self, A, liste1, liste2):
        return transfo_proj(A, liste1, liste2)

    def inter2(self, courbe1, courbe2, numero):
        coooords=(0,0,0)
        rooot=[]
        if isinstance(courbe1, tuple):
            a, b, c = courbe1
            courbe1 = Polynome(((c, b,), (a,)))
        if isinstance(courbe2, tuple):
            a, b, c = courbe2
            courbe2 = Polynome(((c, b,), (a,)))
        if True:
            poly1, poly2 = courbe1.expr_rationals(('x', 'y')), courbe2.expr_rationals(('x', 'y'))
            print(poly1, poly2)
            b = groebner([poly1, poly2], x, y)
            print(type(b[0]), type(b[1]),f'pol : {b[0]}', f'pol2 : {b[1]}')
            c = Poly(b[1]).all_coeffs()
            root = resoudre(c)
            for r in root:
                k = str(b[0]).replace("y","("+ str(r)+")")
                autre_roots = resoudre(Poly(k).all_coeffs())
                for ax in autre_roots:
                    rooot.append((ax,r))
        else:
            if courbe2.deg==1:
                droite, courbe=courbe2, courbe1
            else:
                droite, courbe=courbe1, courbe2
            coord=droite
            coord2=courbe
            k2=-coord[2]/coord[1]
            k3=-coord[0]/coord[1]
            poly=[0]*(courbe.deg+1)
            if coord[1] == 0:
                #polynomey = find_eq_courbe(courbe.coords(), courbe.deg, "y")
                poly = coords(k2)
                #for j in range(len(polynomey)):
                #    poly[j] = eval(str(polynomey[j]).replace("x", str(k2)))
            else:
                permut = permutations(courbe.deg)
                for i in range(len(permut)):
                    k = (-1)**permut[i][1]/(coord[1]**permut[i][1])
                    for j in range(permut[i][1]+1):
                        poly[courbe.deg-permut[i][0]-j] += k* coord2[i]*coord[0]**j*coord[2]**(permut[i][1]-j) * binom(permut[i][1], j)
            roots = poly.resoudre()
            if coord[1]==0:
                for r in roots:
                    rooot.append((k2, r))
            else:
                for r in roots:
                    rooot.append((r, k2 +r*k3))
        if numero <len(rooot):
            coooords= (rooot[numero][0], rooot[numero][1],1)
        return coooords
    
    def ortho(self, A):
        """Point orthogonal de A"""
        xA, yA, zA = A
        if zA != 0:
            raise ValueError("Orthogonal d'un point non à l'infini")
        return (-yA, xA, 0)
    
    def inf(self, a):
        """Point à l'infini d'une droite"""
        xa, ya, za = a
        return (-ya, xa, 0)
    
    def milieu(self, A, B):
        """Définition du milieu de deux points"""
        xA, yA, zA = A
        xB, yB, zB = B
        return (xA*zB+xB*zA, yA*zB+yB*zA, 2*zA*zB)

    def centreInscrit(self, A, B, C):
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
    
class Droite(Creature):
    """Classe Droite"""
    classe = 'Droite'

    def __init__(self, plan, nom="", method="", args=[], objet=None, color="grey",u=0, vis = 1):
        if nom in (0, 1):
            nom = plan.nouveau_nom(nom)
        super().__init__(plan, nom=nom, args = args, color=color, method=method, deg=1,u=u, vis = vis)
        if method == "coord" and objet is not None and type(objet) is not Droite:
            self.objet = objet
        plan.droites[self.nom] = self
        

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        """Définition de l'égalité de deux droites"""
        if type(other) != type(self): return False
        xa, ya, za = self.coords()
        xb, yb, zb = other.coords()
        return ya*zb-yb*za == za*xb-zb *xa == xa*yb-xb*ya == 0
    
    def coord(self, c,d=0, e=0):
        """Définition d'une droite par ses coordonnées"""
        if isinstance(c, tuple):
            x, y, z = c
            if x == y == z == 0:
                raise ValueError("Droite (0,0,0) impossible")
            return (x, y, z)
        return (c,d,e)
    
    def inter(self, A, B):
        """Définition d'une droite par deux points"""
        xA, yA, zA = A
        xB, yB, zB = B
        return (yA*zB-yB*zA,
                zA*xB-zB*xA,
                xA*yB-xB*yA)
    
    def biss(self, a, b, numero = 1): #numéro vaut 1 ou -1
        xa, ya, za = a
        xb, yb, zb = b
        return (
                xa*sqrt(xb**2+yb**2)-numero*xb*sqrt(xa**2+ya**2),
                ya*sqrt(xb**2+yb**2)-numero*yb*sqrt(xa**2+ya**2),
                za*sqrt(xb**2+yb**2)-numero*zb*sqrt(xa**2+ya**2)
            )
    
    def tangente(self, C, p):
        """C -> CA
        a -> complexe
        b -> complexe
        Construit la tangente à C en le point (a,b)
        """
        a, b = p[:2]
        coords = C
        polynomex = coords.change_variables()
        polynomey = coords
        coef1 = polynomex(b).derivee()(a)
        coef2 = polynomey(a).derivee()(b)
        coords_droite = (coef1, coef2, -coef1*a -coef2*b)
        return coords_droite
    
    def dual(self, p):
        return p.coord
    
    
class CA(Creature):
    """"Classe Courbe_alg"""
    classe = 'Courbe'

    def __init__(self, plan, nom="", method="", args=[], deg="",color="green",u=0, vis = 1):
        if nom in (0, 1):
            nom = plan.nouveau_nom(nom)
        if method == "cercle":
            deg = 2
        elif method in transformation:
            deg = args[0].deg 
        elif deg == "":
            deg = floor(sqrt(2*len(args)+9/4)-3/2)
            args = args[:(deg**2+3*deg)//2]
        else:
            args = args[:(deg**2+3*deg)//2]
        super().__init__(plan, nom=nom, args = args, color=color, method=method, deg=deg, u=u, vis = vis)
        plan.CAs[self.nom] = self
        

    def __hash__(self):
        return id(self)

    def inter(self, deg, *args):#INTERpolation
        xrint('Début interpolation')
        xrint(deg)
        zzzz=time.time()
        permut = permutations(deg)
        detConi = []
        xrint('args :', args)
        for i in args:
            a, b, c = i
            detConibis = []
            for j in permut:
                detConibis.append(a**j[0]*b**j[1]*c**j[2])
            xrint(type(detConibis[0]))
            detConi.append(detConibis)
        coords = []
        a = 0
        if deg <=7:
            a = deg + 3
        else:
            a = 2*deg - 5
        for i in range(len(detConi)):
            for j in range(len(detConi[i])):
                detConi[i][j] = detConi[i][j]/(10**a)
        for i in range(len(permut)):
            sousDet = [[detConi[j][k] for k in range(i)]+[detConi[j][k] for k in range(i+1, len(permut))] for j in range(len(detConi))]
            coords.append(((-1)**(i+1) * determinant(sousDet), (permut[i][:2])))
        nouv_coords = [[0]*(deg+1) for i in range(deg + 1)]
        a = 0
        i = 0
        while a == 0 and i < len(coords):
            if coords[i][0] != 0:
                a = coords[i][0]
            i+=1
        if a != 0:
            for j in range(len(coords)):
                c, b = coords[j][1][0], coords[j][1][1]
                nouv_coords[c][b] = numpy.real(coords[j][0] / a)
        print('nouv_coords :', nouv_coords)
        poly = Polynome(nouv_coords)
        print('poly :', poly)
        xrint(f'Fin interpolation. Temps estimé : {time.time()-zzzz}')
        return poly
    
    def cercle(self, deg, *args):
        ''' crée une conique tangente à d1 en U et à d2 en V passant par le point B
        permet notamment de faire un cercle de centre c1 si confondu avec c2'''
        permut = permutations(deg)
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
        coords = []
        for i in range(len(permut)):
            sousDet = [[detConi[j][k] for k in range(i)] + [detConi[j][k] for k in range(i+1, len(permut))] for j in range(len(detConi))]
            coords.append(((-1)**(i+1) * determinant(sousDet), (permut[i][:2])))
        nouv_coords = [[0]*(deg+1) for i in range(deg + 1)]
        a = 0
        i = 0
        while a == 0 and i < len(coords):
            if coords[i][0] != 0:
                a = coords[i][0]
            i+=1
        if a != 0:
            for j in range(len(coords)):
                c, b = coords[j][1][0], coords[j][1][1]
                nouv_coords[c][b] = numpy.real(coords[j][0] / a)
        print('nouv_coords :', nouv_coords)
        poly = Polynome(nouv_coords)
        print('poly :', poly)
        return poly
    
class Plan:

    def __init__(self, main, nom = 'Plan 1', dd = None):
        self.main = main
        self.noms = []
        self.objets = {}
        self.points = {}
        self.tkinter_object = {}
        self.droites = {}
        self.CAs={}
        self.CAst={}
        self.U = Point(self, nom="U", method="coord", args=[(1,1j,0)])
        self.V = Point(self, nom="V", method="coord", args=[(1,-1j,0)])
        self.inf = Droite(self, nom="Inf", method="inter", args=[self.U, self.V])
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


    def nouveau_nom(self, u = 1, type="courbe"):
        lettre, chiffre = 0, 0
        if type=='courbe':
            nom_init="a"
        else:
            nom_init="A"
        nom = nom_init
        while nom in self.noms:
            lettre += 1
            chiffre += lettre//26
            lettre = lettre%26
            nom = ('' if u else '_') + chr(ord(nom_init) + lettre) + (str(chiffre) if chiffre else '')
        return nom

    def contre_action(self, fonc, args):
        xrint(fonc, args)
        if self.annulation:
            self.ctrl_y[-1].append((fonc, args))
        else:
            self.ctrl_z[-1].append((fonc, args))
        self.main.maj_bouton()

    def action_utilisateur(self, act):
        if (self.derniere_action == act and act is not None) or act in ('ctrlz', 'ctrly'): return
        self.ctrl_y = []
        if len(self.ctrl_z) == 0 or self.ctrl_z[-1] != []:
            self.ctrl_z.append([])
        self.main.maj_bouton()
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
        self.main.maj_bouton()
        xrint(self.ctrl_y, self.ctrl_z)
        
    def ctrly(self):
        self.ctrl_z.append([])
        liste = []
        while liste == []:
            liste = self.ctrl_y.pop(-1)
        for fonc, args in liste[::-1]:
            fonc(*args)
        self.main.maj_bouton()

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

    def move(self, point, coords):
        if point.arbre.parents == set():
            point.args=[coords]
            for i in sorted(list(Arbre.descente(self, point.arbre)), key=lambda x: x[1]):
                Creature.set_coords(i[0].valeur)
        self.modifs = (True, True)
        return {i[0].valeur.nom for i in point.arbre.descente(point.arbre)}

    def new_harmonique(self, nom, A,B,C, u = 1):
        d = Point(self, nom = nom, method = 'harmonique', args = (A,B,C), u = u)
        return d

    def new_rotation(self, nom, obj, p, angle, u = 1):
        d = type(obj)(self, nom = nom, method = 'rotation', args = (obj, p, angle), u = u)
        return d
    
    def new_homothetie(self, nom, obj, p, rapport, u = 1):
        d = type(obj)(self, nom = nom, method = 'homothetie', args = (obj, p, rapport), u = u)
        return d

    def new_translation(self, nom, obj, vecteur, u = 1):
        d = type(obj)(self, nom = nom, method = 'translation', args = (obj, vecteur), u = u)
        return d
    
    def new_symetrie(self, nom, obj, droite, u = 1):
        d = type(obj)(self, nom = nom, method = 'symetrie', args = (obj, droite), u = u)
        return d
    
    def new_projective(self, nom, obj, liste1, liste2, u = 1):
        d = type(obj)(self, nom = nom, method = 'projective', args =(obj, liste1, liste2), u = u)
        return d

    def newPoint_coord(self, nom, coord):#crée un point libre avec les coordonnées suivantes
        p = Point(self, nom=nom, method="coord", args=[coord], u = 1)
        return p

    def newPoint_objets(self, nom, methode, objet1, objet2, numero, u = 1):#crée l'intersection de deux objets qui n'est pas dans inters
        p = Point(self, nom = nom, method = methode, args = [objet1, objet2, numero], u = u)
        return p

    def newDroite(self, nom, args, method, u = 1):
        d = Droite(self, nom=nom, method=method, args=args, u = u)
        return d

    def newCA(self, nom, liste, u = 1):
        a = CA(self, nom = nom, method = "inter", args = liste, u = u)
        return a

    def newCAtan(self, nom, d1, d2, point, point2, point3, u = 1):
        a = CA(self, nom=nom, method='cercle', args=[d1, d2, point, point2, point3], u = u)
        return a

    def newProjectionOrtho(self, nom, args, u = 1):
        d = self.newPerp(self.nouveau_nom(0), args, u = 0)
        p = Point(self, nom = nom, method = 'inter', args = (args[0], d), u = u)
        return p

    def newCentreInscrit(self, nom, p1, p2, p3, u = 1):
        p = Point(self, nom = nom, method = 'centreInscrit', args = (p1, p2, p3), u = u)
        return p

    def newPerp(self, nom, args, u = 1):
        p1 = Point(self, nom = self.nouveau_nom(0), method = 'inf', args = (args[0],), u = 0)
        p2 = Point(self, nom = self.nouveau_nom(0), method = 'ortho', args = (p1,), u = 0)
        d = Droite(self, nom = nom, method = 'inter', args = (args[1], p2), u = u)
        return d
    
    def newMilieu(self, nom, args, u = 1):
        return Point(self, nom = nom, method = 'milieu', args = (args[0], args[1]), u = u)
    
    def newCentre(self, nom, args, u = 1):
        d1 = Droite(self, nom = self.nouveau_nom(0), method = 'tangente', args = (args[0], self.U), u = 0)
        d2 = Droite(self, nom = self.nouveau_nom(0), method = 'tangente', args = (args[0], self.V), u = 0)
        return Point(self, nom = nom, method = 'inter', args = (d1, d2), u = u, complexe = False)

    def newMedia(self, nom, args, u = 1):
        d1 = Droite(self, nom = self.nouveau_nom(0), method = 'inter', args = (args[0], args[1]), u = 0)
        p1 = Point(self, nom = self.nouveau_nom(0), method = 'inf', args = (d1,), u = 0)
        p2 = Point(self, nom = self.nouveau_nom(0), method = 'ortho', args = (p1,), u = 0)
        p3= Point(self, nom = self.nouveau_nom(0), method = 'milieu', args = (args[0], args[1]), u = 0)
        d = Droite(self, nom = nom, method = 'inter', args = (p2, p3), u = u)
        return d
    
    def newPara(self, nom, args, u = 1):
        p1 = Point(self, nom = self.nouveau_nom(0), method = 'inf', args = (args[0],), u = 0)
        d = Droite(self, nom = nom, method = 'inter', args = (args[1], p1), u = u)
        return d

    def newCercleInscrit(self, nom, point1, point2, point3):
        p1, p2, p3 = self.objets[point1], self.objets[point2], self.objets[point3]
        a = CA(nom=nom, method="cercle", args=[Point(self, nom=nom, method="centreInscrit", args=[p1, p2, p3]), Point(self, nom=nom, method="inter", args=[Droite(nom=nom, method="inter", args=[p1, p3]), Droite(nom=nom, method="perp", args=[Droite(nom=nom, method="inter", args=[p2, p3]), Point(self, nom=nom, method="centreInscrit", args=[p1, p2, p3])])]), self.objets["U"], self.objets["V"]], deg=2, u = 1)
        return a

    def eq(self, a, b):
        return self.objets[a]==self.objets[b]
    
    def infos(self, x):
        return self.objets[x]
    
    def switchNom(self, nom, canv):
        point = self.points[nom]
        if canv.itemcget(point.tkinter[1], "text")=="":
            canv.itemconfig(point.tkinter[1], text=point.nom)
        else:
            canv.itemconfig(point.tkinter[1], text="")

    def switchPoint(self, nom, canv):
        point = self.points[nom]
        if canv.itemcget(point.tkinter[0], "state")=='hidden':
            canv.itemconfig(point.tkinter[0], state="normal")
            canv.itemconfig(point.tkinter[1], text=point.nom)    
        else:
            canv.itemconfig(point.tkinter[0], state='hidden')
            canv.itemconfig(point.tkinter[1], text="")

    def Hide(self, nom, canv):
        point = self.points[nom]
        canv.itemconfig(point.tkinter[1], text="")

    def rename(self, nom, nom2, canv):
        if nom2 not in list(self.objets.keys()):
            self.points[nom].nom =nom2
            self.points[nom2] = self.points.pop(nom)
            self.objets[nom2] = self.objets.pop(nom)
            canv.itemconfig(self.points[nom2].tkinter[1], text=self.points[nom2].nom)
        self.modifs = (True, True)
