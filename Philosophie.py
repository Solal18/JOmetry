class Signe:#d'aucuns pourraient dire que ça sert à rien mais si, c'est quand même beaucoup plus philosophique d'avoir un truc à part
    def __init__(self, nom):
        self.nom = nom

    def __eq__(self, other):
        return self.nom == other.nom
        
    def __hash__(self):
        return 1

class ExpressionIndeterminee:#savoir manipuler le signe
    def __init__(self, ei1, method = "signe", ei2=None):
        self.method = method#signe, + ou * globalement
        self.ei1 = ei1#terme de droite (EI aussi)
        self.ei2 = ei2#terme de gauche (EI aussi)
        self.irr = True
        if method == "signe":
            self.exp = ei1
            self.ei1 = Signe(ei1)
        elif method == "nb":
            self.exp = str(ei1)
            self.ei1 = ei1
        else:
            self.irr = ei1.irr and ei2.irr and method == "mult"#on est irréductible ssi on est produit de signe
            s1, s2 = str(self.ei1), str(self.ei2)
            if ei1.method != self.method and ei1.method != "signe" and ei1.method != "nb": s1 = ei1.parentheser()
            if ei2.method != self.method and ei2.method != "signe" and ei2.method != "nb": s2 = ei2.parentheser()
            self.exp = s1 + {"add" : " + ", "mult" : " * "}[method] + s2#c'est une str qui est principalement là pour qu'on puisse visualiser (je ne suis pas sûr que ça serve à quelque chose) 
            
    def remplace(self, other):#fait en sorte que self devienne other (ça marche pas classiquement si je ne m'abuse c'est un peu un patch)
        self.method = other.method
        self.ei1 = other.ei1
        self.ei2 = other.ei2
        self.exp = other.exp
        self.irr = other.irr
        
    def evaluer(self, valeures):#renvoie la valeure de l'expression indéterminée self quand on la calcule avec les valeures fournies par le dictionnaire. Par exemple (X+Z)*Y.evaluer({"X" : 1, "Y" : 4, "Z" : -2}) vaut -4. On peut aussi enlever les guillemets
        if self.method == "signe":
            try:
                return valeures[self]
            except KeyError:
                try:
                    return valeures[self.ei1.nom]
                except KeyError:
                    raise ValueError("Il nous faut plus de valeures pour pouvoir calculer cette EI !")
        elif self.method == "add":
            return self.ei1.evaluer(valeures) + self.ei2.evaluer(valeures)
        elif self.method == "mult":
            return self.ei1.evaluer(valeures) * self.ei2.evaluer(valeures)
        elif self.method == "nb":
            return self.ei1
        else:
            raise TypeError("Opération de type inconnu.")
        

    def devp(self):#renvoi l'objet développé
        if self.irr:
            return self
        elif self.method == "add":
            return self.ei1.devp() + self.ei2.devp()
        elif self.method == "mult":
            if self.ei1.method == "add":
                return (self.ei1.ei1 * self.ei2 + self.ei1.ei2 * self.ei2).devp()
            elif self.ei2.method == "add":
                return (self.ei1 * self.ei2.ei1 + self.ei1 * self.ei2.ei2).devp()
            else:
                return (self.ei1.devp() * self.ei2.devp()).devp()
        else:
            raise IndexError
        
    def set_dev(self):#développe l'objet même
        self.remplace(self.devp())
        
    def irrToEns(self):#renvoie l'ensembles des termes du produit d'un irréductible
        if self.method == "signe" or self.method == "nb":
            t = set()
            t.add(self)
            return t
        elif self.method == "mult":
            return self.ei1.irrToEns() | self.ei2.irrToEns()
        else:
            raise AttributeError("Il n'est pas irréductible !")

            
        
    def sommeToEns(self):#renvoie, étant donné une exp qui est une somme d'irréductibles, l'ensemble des termes
        if self.irr == True:
            t = set()
            t.add(self)
            return t
        elif self.method == "add":
            return self.ei1.sommeToEns() | self.ei2.sommeToEns()
        else:
            raise AttributeError("Vous voulez les termes d'un produits !")
    
    def eiToEns(self):#développe une EI et la transforme en Ens
        return self.devp().sommeToEns()

    def __add__(self, other):
        return ExpressionIndeterminee(self, "add", other)

    def __mul__(self, other):
        if isinstance(other, ExpressionIndeterminee):
            return ExpressionIndeterminee(self, "mult", other)
        else:
            return ExpressionIndeterminee(self, "mult", ExpressionIndeterminee(other, method = "nb"))
        
    __rmul__ = __mul__
    
    def __str__(self):
        return self.exp
    
    def __eq__(self, other):
        if self.method == "signe" and other.method == "signe":
            return self.ei1 == other.ei1
        elif self.irr and other.irr:
            return self.irrToEns() == other.irrToEns()
        else:
            return self.eiToEns() == other.eiToEns()
        
    def __hash__(self):
        return 1
    
    def parentheser(self):
        return "(" + self.exp + ")"

"""
class Test:
    def __init__(self, test):
        self.test = test
    
    def remplace(self, obj):
        self.test = obj.test
        
    def __str__(self):
       return self.test

t = Test("salut")
v = Test("bonjour")
t.remplace(v)
print(t,v)

"""

X = ExpressionIndeterminee("X")
Y = ExpressionIndeterminee("Y")
Z = ExpressionIndeterminee("Z")
t = 2*(X+Y)*(Y+Z)*(Z+X)
print(t.evaluer({X : 5, Y : 3, "Z" : 5.5}))
print(2*X == X*2)
