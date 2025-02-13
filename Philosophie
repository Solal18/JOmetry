class expressionIndeterminee:#savoir manipuler le signe
    def __init__(self, ei1, method = "signe", ei2=None):
        self.method = method
        self.ei1 = ei1
        self.ei2 = ei2
        if method == "signe":
            self.exp = ei1
            self.irr = True
            self.ei1 = Signe(ei1)
        else:
            self.irr = ei1.irr and ei2.irr and method == "mult"
            s1, s2 = str(self.ei1), str(self.ei2)
            if ei1.method != self.method and ei1.method != "signe": s1 = ei1.parentheser()
            if ei2.method != self.method and ei2.method != "signe": s2 = ei2.parentheser()
            self.exp = s1 + {"add" : " + ", "mult" : " * "}[method] + s2

    def devp(self):
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

    def __add__(self, other):
        return expressionIndeterminee(self, "add", other)

    def __mul__(self, other):
        return expressionIndeterminee(self, "mult", other)
    
    def __str__(self):
        return self.exp
    
    def __eq__(self, other):
        t1 = self.devp()
        t2 = other.devp()
        if t1.method != t2.method:
            return False
        else:
            return (self.ei1 == other.ei1 and self.ei2 == other.ei2) or (self.ei1 == other.ei2 and self.ei2 == other.ei1)

    def parentheser(self):
        return "(" + self.exp + ")"

X = expressionIndeterminee("X")
Y = expressionIndeterminee("Y")
Z = expressionIndeterminee("Z")
t = (X+Y+Z)*(X+Y)*Z*X*(X+X+Y+Y+Z)
print(str(t))
print(str(t.devp()))
