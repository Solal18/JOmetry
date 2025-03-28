import tkinter as tk
import tkinter.ttk as ttk
from tkinter import colorchooser as tk_cc
from PIL import Image, ImageTk
import os.path as op
import weakref
from threading import Thread
from serveur import Serveur
from random import randrange
from numpy import float64
import re
import Engine as Geo
        
def polonaise_inverse(formule, dictionnaire):
    ordre = {'+':1, '-':1, '*':2, '/':2, 'd':3, 'angle':3}
    
    def mettre_operateur(operateurs, sortie):
        op = operateurs.pop()
        if op == 'd' or op == 'angle':
            num_args = 2 if op == 'd' else 3
            args = [sortie.pop() for _ in range(num_args)][::-1]
            sortie.extend(args + [op])
        else:
            droite, gauche = sortie.pop(), sortie.pop()
            sortie.append(gauche)
            sortie.append(droite)
            sortie.append(op)

    formule = re.sub(r'\s+', '', formule)
    elems = re.findall(r'd\([^)]*\)|angle\([^)]*\)|[A-Za-z_][A-Za-z_0-9]*|\d+\.?\d*|[+\-*/()]', formule)
    sortie = []
    operateurs = []

    for elem in elems:
        if elem in dictionnaire:
            sortie.append(dictionnaire[elem])
        elif re.match(r'\d+\.?\d*', elem):
            sortie.append(float(elem))
        elif elem in {'+', '-', '*', '/'}:
            while (operateurs and operateurs[-1] != '(' and
                   ordre[operateurs[-1]] >= ordre[elem]):
                mettre_operateur(operateurs, sortie)
            operateurs.append(elem)
        elif elem == '(':
            operateurs.append(elem)
        elif elem == ')':
            while operateurs and operateurs[-1] != '(':
                mettre_operateur(operateurs, sortie)
            if not operateurs or operateurs[-1] != '(':
                raise ValueError("Parenthèse incorrecte.")
            operateurs.pop()
        elif elem.startswith('d(') or elem.startswith('angle('):
            deb = elem.find('(') + 1
            fin = elem.rfind(')')
            if fin == -1:
                raise ValueError("Parenthèse incorrecte.")
            args = elem[deb:fin].split(',')
            if (elem.startswith('d(') and len(args) != 2) or (elem.startswith('angle(') and len(args) != 3):
                raise ValueError("Nombre d'arguments incorrect.")
            for arg in args:
                if arg not in dictionnaire:
                    raise ValueError(f"Clé inconnue : {arg}")
                sortie.append(dictionnaire[arg])
            operateurs.append('d' if elem.startswith('d(') else 'angle')
        else:
            raise ValueError(f"Objet non reconnu : {elem}")

    while operateurs:
        mettre_operateur(operateurs, sortie)

    return sortie


def traduction():
    f = open(f'{op.dirname(__file__)}\\traduction.txt', encoding = 'utf-8')
    texte = f.read().split('\n\n')
    f.close()
    langues = []
    dicos = []
    for p in texte:
        lignes = p.split('\n')
        langues.append(lignes[0])
        lignes = [l[2:] for l in lignes[1:]]
        dic = {}
        for l in lignes:
            guill = [i for i,c in enumerate(l) if c == '"']
            doubp = [i for i,c in enumerate(l) if c == ':' and not guill[0]<i<guill[1]][0]
            cle, val = l[:doubp], l[doubp+1:]
            if not (cle[0] == cle[-1] == '"'):
                cle = int(cle)
            else: cle = cle[1:-1]
            if not (val[0] == val[-1] == '"'):
                val = int(val)
            else: val = val[1:-1]
            dic[cle] = val
        dicos.append(dic)
    langue = langues[0]
    def trad(lang, mot):
        if lang == 'Français': return mot
        if lang not in langues or mot not in dicos[0]:
            print(f'{mot} non traduit en {lang}')
            return mot
        if dicos[0][mot] not in dicos[langues.index(lang)]:
            print(f'{mot} non traduit en {lang}')
            return mot
        return dicos[langues.index(lang)][dicos[0][mot]]
    return langue, langues, trad

try:
    langue, langues, trad = traduction()
except Exception as e:
    print('Impossible de charger les traductions')
    print(f'Erreur rencontrée : {e}')
    langue = 'Français'
    langues = ['Français']
    trad = lambda l, x: x

params = {'BoldP':3, 'BoldC':3, 'Style':'default', 'Langue':langue, 'ColTooltip':'gray', 'ColP':'green', 'ColC':'green', 'TempsTooltip':300}
try:
    f = open(f'{op.dirname(__file__)}\\parametres.txt', encoding = 'utf-8')
    charges = Geo.val(f.read())
    params = params|charges 
    f.close()
except Exception as e:
    print('Impossible de charger les parametres')

langue_def = langue        


class Trad(tk.StringVar):
    variables = set()
    variables_weak = weakref.WeakSet()
    
    def __init__(self, mot, langue = None, noteb = None, weak = 0):
        super().__init__()
        if weak:
            self.variables_weak.add(self)
        else:
            self.variables.add(self)
        self._lang = None
        self.noteb = noteb
        self.mot = mot
        if langue is None: langue = params['Langue']
        self.langue = langue
    
    def __hash__(self):
        return id(self)
    
    @property
    def langue(self):
        return self._lang
    
    @langue.setter
    def langue(self, lang):
        self._lang = lang
        if self.noteb is not None:
            if isinstance(self.noteb[0], ttk.Notebook):
                self.noteb[0].tab(self.noteb[1], text = trad(lang, self.mot))
            elif isinstance(self.noteb[0], ttk.Treeview):
                if isinstance(self.noteb[1], int):
                    self.noteb[0].heading(self.noteb[1], text = trad(lang, self.mot))
                elif isinstance(self.noteb[1], tuple):
                    ligne = self.noteb[0].item(self.noteb[1][0])['values']
                    ligne[self.noteb[1][1]] = trad(lang, self.mot)
                    self.noteb[0].item(self.noteb[1][0], values = ligne)
        else:
            self.set(trad(lang, self.mot))
    
    @classmethod
    def set_lang(cls, lang):
        for variable in cls.variables|set(cls.variables_weak):
            variable.langue = lang



class Scrollable_Frame(ttk.Frame):
    
    def __init__(self, parent, orientation = 'vertical', **kwargs):
        super().__init__(parent)
        self.grid(**kwargs)
        self.canvas = tk.Canvas(self, bd = 0)
        self.frame = ttk.Frame(self.canvas)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        if orientation == 'vertical':
            self.scrollbar = ttk.Scrollbar(self, command = self.canvas.yview, orient = orientation)
        else:
            self.scrollbar = ttk.Scrollbar(self, command = self.canvas.xview, orient = orientation)
            
        
        if orientation == 'vertical':
            self.canvas.configure(yscrollcommand = self.scrollbar.set)
            self.scrollbar.grid(row = 0, column = 1, sticky = 'nsew')
        else:
            self.canvas.configure(xscrollcommand = self.scrollbar.set)
            self.scrollbar.grid(row = 1, column = 0, sticky = 'nsew')
        self.canvas.create_window(0, 0, anchor = 'nw', window = self.frame, tags = ('frame',))
        self.canvas.bind('<Configure>', self.configure_frame)
        self.frame.bind('<Configure>', self.maj_canvas)
        self.maj_canvas()
        self.canvas.grid(row = 0, column = 0, sticky = 'nsew')

    def maj_canvas(self, ev = None):
        if self.canvas.winfo_height() > self.frame.winfo_height():
            self.canvas.configure(height = self.frame.winfo_height())
        if self.canvas.winfo_width() > self.frame.winfo_width():
            self.canvas.configure(width = self.frame.winfo_width())
        self.canvas.configure(scrollregion = self.canvas.bbox('all'))

    def configure_frame(self, ev):
        self.canvas.itemconfig('frame', width = ev.width)
        #self.canvas.itemconfig('frame', height = ev.height)


class AideFenetre:
    def __init__(self, fenetre, doc):
        self.fen = tk.Toplevel(fenetre)
        self.fen.title('Aide')
        self.entree = ttk.Entry(self.fen, width=30)
        self.entree.grid(row = 0, column = 0)
        self.entree.bind('<KeyRelease>', self.mettre_a_jour_resultats)
        self.possibles = tk.StringVar()
        self.liste = tk.Listbox(self.fen, listvariable = self.possibles, width = 30)
        self.liste.grid(row = 1, column = 0)
        self.liste.bind('<<ListboxSelect>>', self.clic_liste)
        self.texte = tk.Text(self.fen, width=50, height=10, padx = 5)
        self.texte.grid(row = 0, column = 1, rowspan = 2)
        self.texte.config(state = 'disabled')
        self.doc = doc
        self.resultats = None
        self.mettre_a_jour_resultats()
        self.texte.config(state = 'normal')
        self.texte.insert("end", "Bienvenue dans le menu d'aide.\nLes différents onglets d'aide sont disponible dans la partie gauche de la fenêtre.")
        self.texte.config(state = 'disabled')
    
    
    def recherche(self, texte):
        resultats = []
        for i in self.doc:
            if any([(texte in j) for j in i[0]]) and len(resultats) < 10:
                resultats.append(i[1])
        return resultats

    def texte_aide(self, texte:str):
        texte, liens = texte+"\n"+self.doc[[i[1] for i in self.doc].index(texte)][2], {}
        texte = texte.replace("\\n", "\n")
        while '[' in texte:
            idx1, idx2, idx3 = texte.index('['), texte.index('|'), texte.index(']')
            liens[(idx1, idx2 - 1)] = texte[idx2 + 1:idx3]
            texte = texte[:idx1]+texte[idx1+1:idx2]+texte[idx3+1:]
        return texte, liens

    def mettre_a_jour_resultats(self, event=None):
        texte_recherche = self.entree.get()
        self.resultats = self.recherche(texte_recherche.lower())
        self.possibles.set(' '.join([x.replace(' ', '\ ') for x in self.resultats]))

    def clic_liste(self, event = None, texte = ''):
        if self.resultats is None: return
        if texte == '': texte = self.resultats[self.liste.curselection()[0]]
        nouveau_texte, liens = self.texte_aide(texte)
        titre, nouveau_texte = self.mettre_en_forme_texte(nouveau_texte)
        self.texte.config(state = 'normal')
        self.texte.delete('1.0', 'end')
        self.texte.insert('end', titre+'\n')
        self.texte.insert('end', nouveau_texte)
        self.texte.config(state = 'disabled')
        self.texte.tag_add('titre', '1.0+0c', f'1.0+{len(titre)}c')
        self.texte.tag_config('titre', underline=True, justify='center')
        for (id1, id2), mot in liens.items():
            if mot not in (i[1] for i in self.doc):
                continue
            i1, i2 = f'1.0+{id1}c', f'1.0+{id2}c'
            self.texte.tag_add(mot, i1, i2)
            self.texte.tag_config(mot, foreground = 'blue', underline = True)
            self.texte.tag_bind(mot, '<Button-1>', lambda ev, mot = mot: self.clic_liste(texte = mot))
    
    def mettre_en_forme_texte(self, texte):
        ltexte = texte.split('\n')
        titre, texte = ltexte[0], ltexte[1:]
        nouveau_texte = ''
        for paragraphe in texte:
            paragraphe = paragraphe.split(' ')
            longueur = 0
            for mot in paragraphe:
                if longueur + len(mot) >= 50:
                    longueur = len(mot)
                    nouveau_texte = nouveau_texte[:-1] + '\n' + mot + ' '
                else:
                    longueur += len(mot) + 1
                    nouveau_texte += mot + ' '
            nouveau_texte = nouveau_texte[:-1] + '\n'
        return titre, nouveau_texte



    
 

class ColorChooser(ttk.Frame):
    
    def __init__(self, parent, fenetre, textvariable):
        super().__init__(parent)
        self.fenetre = fenetre
        self.textvariable = textvariable
        self.entree = tk.Entry(self, textvariable = textvariable)
        self.columnconfigure(0, weight = 1)
        self.dernier_correct = textvariable.get()
        self.entree.grid(row = 0, column = 0, sticky = 'nsew')
        self.img = tk.BitmapImage(data = \
          '#define colorpicker_width 10\n\#define colorpicker_height 10\n\
           static unsigned char colorpicker_bits[] = {\
             0xc0, 0x01, 0xe0, 0x03, 0xe0, 0x03, 0xd8, 0x03, 0xb0, 0x01,\
             0x78, 0x00, 0x5c, 0x00, 0x0e, 0x00, 0x07, 0x00, 0x03, 0x00 };')
        self.bouton = ttk.Button(self, image = self.img, command = self.tk_couleur)
        self.bouton.grid(row = 0, column = 1, sticky = 'nsew')
        self.entree.bind('<KeyRelease>', self.verifier)
        print('ccccccccccccc :',self.dernier_correct)
        
    def tk_couleur(self):
        couleur = tk_cc.askcolor(parent = self)
        if couleur[1] is not None:
            self.textvariable.set(couleur[1])
            
    def get(self):
        self.verifier()
        return self.dernier_correct
    
    def verifier(self, ev = None):
        try: self.fenetre.winfo_rgb(self.textvariable.get())
        except tk.TclError:
            self.entree['bg'] = 'red'
            return
        self.entree['bg'] = 'white'
        self.dernier_correct = self.textvariable.get()
        print('ccccccccccccc :',self.dernier_correct)
        return 

class EditeurObjets:
    
    def __init__(self, fenetre, main, separe):
        self.fenetre = fenetre
        self.separe = separe
        if separe:
            self.grande_frame = tk.Toplevel()
            self.frame = self.grande_frame
            self.grande_frame.protocol('WM_DELETE_WINDOW', self.fermer_fenetre)
        else:
            self.grande_frame = ttk.Frame(main.panedwindow)
            main.panedwindow.insert('end', self.grande_frame)
            self.frame = ttk.Frame(self.grande_frame)
            self.grande_frame.columnconfigure(0, weight = 1)
            self.frame.grid(row = 0, column = 0, sticky = 'nsew')
            tk.Button(self.grande_frame, text = 'fermer', command = self.fermer_fenetre, bg = '#ddd').grid(row = 1, column = 0)
            fenetre.bind('<Return>', self.clic_entree)
        for i in range(3):
            self.frame.columnconfigure(i, weight = 1)
        self.imgs = (tk.BitmapImage(data = \
          '#define colorpicker_width 10\n\#define colorpicker_height 10\n\
           static unsigned char colorpicker_bits[] = {\
             0xc0, 0x01, 0xe0, 0x03, 0xe0, 0x03, 0xd8, 0x03, 0xb0, 0x01,\
             0x78, 0x00, 0x5c, 0x00, 0x0e, 0x00, 0x07, 0x00, 0x03, 0x00 };'),
                     ImageTk.PhotoImage(file = f'{op.dirname(__file__)}\images\poubelle2.png'))
        self.main = main
        self.frame_t = ttk.Frame(self.frame)
        self.frame_t.grid(row = 0, column = 0, columnspan = 4, sticky = 'nsew')
        self.frame_t.columnconfigure(0, weight = 1)
        self.tableau = ttk.Treeview(self.frame_t, columns = ('nom', 'type', 'fonction', 'args', 'couleur', 'vis'), selectmode = 'browse')
        self.tableau.grid(row = 0, column = 0, sticky = 'nsew')
        bar = ttk.Scrollbar(self.frame_t, orient = 'vertical', command = self.tableau.yview)
        bar.grid(row = 0, column = 1, sticky = 'ns')
        self.tableau['yscrollcommand'] = bar.set
        self.tableau.column('#0', width = 0, stretch = False)
        for i, t in (('nom', 'objet'), ('type', 'type'), ('fonction', 'définition'), ('args', 'dépend de'), ('couleur', 'couleur'), ('vis', 'affichage')):
            self.tableau.column(i, width = 40)
            self.tableau.heading(i, text = t)
            Trad(t, params['Langue'], (self.tableau, i))
        self.nom_methodes = {c:v for c, v in {'coord' : 'coordonées', 'inter' : 'intersection', 'inter2' : 'intersection', 'ortho' : 'ortho', 'inf' : 'inf', 'milieu' : 'milieu', 'centreInscrit' : 'centre inscrit',
                        'perp' : 'perpendiculaire', 'media' : 'médiatrice', 'biss' : 'bissectrice', 'rotation' : 'rotation', 'transformation' : 'transformation', 'homothetie' : 'homothetie', 'tangente' : 'tangente',
                        'cercle' : 'conique tangente à deux droites', 'segment':'segment', 'interpol' : 'interpolation', 'harmonique' : 'harmonique', 'PsurCA' : 'point sur courbe', 'invers' : 'inversion', 'inversion':'inversion', 'cubic':'Cubique isoptique'}.items()}
        self.var1, self.var2, self.var3 = tk.StringVar(), tk.StringVar(), tk.IntVar()
        self.entree = ttk.Entry(self.frame, width = 8, state = 'disabled', textvariable = self.var1)
        self.couleur = ColorChooser(self.frame, self.fenetre, self.var2)
        self.aff = tk.Checkbutton(self.frame, bg = '#ddd', state = 'disabled', variable = self.var3, textvariable = Trad('affichage'))
        self.suppr = tk.Button(self.frame, bg = '#ddd', state = 'disabled', image = self.imgs[1], command = None)
        self.label = tk.Label(self.frame, textvariable = Trad('Selectionnez un objet\npour modifier ses proprietes'), bg = '#ddd')
        self.entree.grid(row = 1, column = 0, sticky = 'nsew')
        self.couleur.grid(row = 1, column = 1, padx = 4, sticky = 'nsew')
        self.aff.grid(row = 1, column = 2, sticky = 'nsew')
        self.suppr.grid(row = 1, column = 3, sticky = 'nsew')
        self.label.grid(row = 2, column = 0, columnspan = 4)
        self.tableau.bind("<<TreeviewSelect>>", self.clic_ligne)
        self.frame.bind('<Return>', self.clic_entree)
        self.maj()

    def supprimer_element(self, i, b = 0):
        ide = i.ide
        for item in self.tableau.get_children():
            if item and item == str(ide):
                self.tableau.delete(item)
        if self.selectionne == i:
            self.deselectionner()
        if b: self.main.action('Supprimer', i.plan, i)

    def deselectionner(self):
        self.var1.set('')
        self.var2.set('')
        self.var3.set(0)
        for widget in (self.entree, self.couleur.bouton, self.couleur.entree,
                       self.aff, self.suppr):
            widget['state'] = 'disabled'
        self.selectionne = None

    def fermer_fenetre(self):
        self.main.editeur_objets = None
        if self.separe:self.grande_frame.destroy()
        else:
            print('forget ?')
            self.main.panedwindow.forget(self.grande_frame)
            self.frame.destroy()

    def ajouter(self, obj):
        if obj.u:
            l = [obj.nom, obj.classe, self.nom_methodes[obj.method], list(map(str, obj.args)), obj.color, ('non', 'oui')[obj.vis]]
            self.objets[obj.ide] = l
            self.tableau.insert('', 'end', iid = str(obj.ide), values = l)

    def maj(self):
        self.trads = set()
        for item in self.tableau.get_children():
            self.tableau.delete(item)
        self.objets = {}
        for objet in self.main.plans[0].objets.values():
            if objet.u:
                self.objets[objet.ide] = (objet.nom, objet.classe, self.nom_methodes[objet.method],
                                    objet.args, objet.color, ('non', 'oui')[objet.vis])
        for ide, l in self.objets.items():
            self.tableau.insert('', 'end', iid = str(ide),values = l)
            self.trads.add(Trad(l[2], params['Langue'], (self.tableau, (str(ide), 2)), weak = 1))
        self.deselectionner()

    def clic_entree(self, event):
        if self.selectionne is None: return
        nom, couleur, aff = self.var1.get(), self.couleur.get(), self.var3.get()
        if nom in ('U', 'V'):
            self.label['textvariable'] = Trad('Nom déjà utilisé\n')
            return
        print(couleur)
        try: self.fenetre.winfo_rgb(couleur)
        except:
            self.label['textvariable'] = Trad('Couleur invalide\n')
            return
        self.label['textvariable'] = Trad('\n')
        ide = self.selectionne.ide
        self.main.action('Modif', self.selectionne.plan, self.selectionne, nom = nom, col = couleur, vis = aff)
        for item in self.tableau.get_children():
            val = self.tableau.item(item)['values']
            if item and item == str(ide):
                val[0] = nom
                val[4] = couleur
                val[5] = ['non', 'oui'][aff]
                self.tableau.item(item, values = val)

    def clic_ligne(self, event):
        if self.tableau.selection() == tuple(): return
        ide = int(self.tableau.selection()[0])
        self.ligne_select = self.tableau.selection()[0]
        ligne = self.tableau.item(str(ide))['values']
        self.selectionne = self.main.plans[0].objets[ide]
        for widget in (self.entree, self.couleur.bouton, self.couleur.entree,
                       self.aff, self.suppr):
            widget['state'] = 'normal'
        self.suppr['command'] = lambda: self.supprimer_element(self.selectionne, b = 1)
        self.var1.set(ligne[0])
        self.var2.set(ligne[4])
        self.var3.set(['non', 'oui'].index(ligne[5]))

def dist_point(a, b):
    a, b = a.coords(), b.coords()
    d,e,f, g,h,i = *a, *b
    if f*i == 0:
        p.append(float('inf'))
    else:
        d,e, g,h = d/f,e/f, g/i,h/i
        p.append(Geo.dist((d,e), (g,h)))
    

def calcul(pile, U, V):
    p = []
    for i in pile:
        if i not in ('+', '-', '*', '/', 'd', 'angle'):
            p.append(i)
        elif i == '+':
            a, b = p.pop(), p.pop()
            p.append(a+b)
        elif i == '*':
            a, b = p.pop(), p.pop()
            p.append(a*b)
        elif i == '/':
            a, b = p.pop(), p.pop()
            p.append(b/a)
        elif i == '-':
            a, b = p.pop(), p.pop()
            p.append(b-a)
        elif i == 'angle':
            a, b, c = [p.pop().coords() for _ in range(3)]
            p.append(Geo.angle(a, b, c, U, V))
        elif i == 'd':
            a, b = p.pop(), p.pop()
            if a.classe == 'Point':
                p, o = a, b
            else:
                p, o = b, a
            if o.classe == 'Point':
                p.append(dist_point(p, o))
            elif o.classe == 'Droite':
                p.append(dist_point(p, Geo.ProjOrtho(o, p)))
            elif o.classe == 'Courbe':
                p.append(dist_point(p, Geo.PsurCA(o, p)))
    return p[0]
        

class EtudieurObjets:
    
    def __init__(self, fenetre, main, separe):
        self.fenetre = fenetre
        if separe:
            self.grande_frame = tk.Toplevel()
            self.frame = self.grande_frame
            self.grande_frame.protocol('WM_DELETE_WINDOW', self.fermer_fenetre)
        else:
            self.grande_frame = tk.Frame(main.panedwindow, bg = '#ddd')
            main.panedwindow.add(self.grande_frame, weight = 0)
            self.frame = tk.Frame(self.grande_frame, bg = '#ddd')
            self.frame.grid(row = 0, column = 0)
            tk.Button(self.grande_frame, text = 'fermer', command = self.supprimer, bg = '#ddd').grid(row = 1, column = 0)
            fenetre.bind('<Return>', self.clic_entree)
        self.main = main
        self.valeur = tk.StringVar()
        self.listvariable = tk.StringVar()
        self.texte = ttk.Label(self.frame, text = Trad('Entrez une valeur pour essayer de determiner une constante, en autorisant les points suivants à varier'))
        self.entree = ttk.Entry(self.frame, textvariable = self.valeur)
        self.formule = ttk.Label(self.frame, text = '')
        self.bouton = ttk.Button(self.frame, text = 'Etude', command = self.etude)
        self.liste = tk.Listbox(self.frame, listvariable = self.listvariable, selectmode = 'multiple')
        self.texte.grid(row = 0, column = 0, columnspan = 2)
        self.entree.grid(row = 1, column = 0)
        self.formule.grid(row = 2, column = 0)
        self.bouton.grid(row = 3, column = 0, columnspan = 2)
        self.liste.grid(row = 1, column = 1, rowspan = 2)
        l = []
        for p in main.plans[0].points.values():
            if p.bougeable():
                l.append(p.nom)
        self.listvariable.set(' '.join(l))
                
        
    def etude(self):
        formule = self.valeur.get().strip(' ')
        bouge = [self.main.plans[0].points[self.liste.get(0, 'end')[nombre]] for nombre in self.liste.curselection()]
        if formule in self.main.plans[0].points:
            point = self.main.plans[0].points[formule]
            positions = [point.args for p in bouge]
            l = []
            for i in range(20):
                for p in bouge:
                    x, y = randrange(0, 500), randrange(0, 500)
                    p.plan.move(p, (x, y, 1))
                z = tuple([float(t) if isinstance(t, float64) else t for t in point.coords()])
                l.append(z)
            l = [(p[0]/p[2], p[1]/p[2], 1)for p in l]
            n = 2
            while n <= 5:
                deg = n*(n+3)//2
                points = l[:deg]
                eq = Geo.interpol(n, *points)
                bon = True
                for i in l:
                    if eq(i[0])(i[1]) >= 1e-10:
                        bon = False
                if bon:
                    break
                n += 1
            self.main.action('Creature', self.main.plans[0], 'Courbe', nom = 1, method = 'interpol', args = l[:deg], u = 1)
            for p, pos in zip(bouge, positions):
                if isinstance(pos[0], Geo.Creature):
                    pos = pos[1]
                p.plan.move(p, pos)
            return
        if formule.count('=') != 1 or formule.count('(') != formule.count(')'): return
        formule = formule .replace('=', '-')
        dic = {}
        for obj in self.main.plans[0].objets.values():
            dic[obj.nom] = obj
        pile = polonaise_inverse(formule, dic)
        U, V = self.main.plans[0].U, self.main.plans[0].V
        
        
            
        
        
    def supprimer(self):
        self.grande_frame.grid_forget()
        self.grande_frame.forget()
        self.main.editeur_objets = None

    def fermer_fenetre(self):
        self.main.editeur_objets = None
        self.grande_frame.destroy()
        
class LanceServeur:
    def __init__(self, main):
        self.main = main
        self.frame = tk.Toplevel()
        self.frame.protocol('WM_DELETE_WINDOW', self.fermer_fenetre)
        ttk.Label(self.frame, textvariable = Trad('Lancer un serveur')).grid(row = 0, column = 0, columnspan = 3, sticky = 'nsew', pady = 3, padx = 3)
        ttk.Label(self.frame, textvariable = Trad('Mot de passe')).grid(row = 1, column = 0, columnspan = 1, sticky = 'nsew', pady = 3, padx = 3)
        self.entree = ttk.Entry(self.frame)
        self.entree.grid(row = 1, column = 1, columnspan = 2, sticky = 'nsew', pady = 3, padx = 3)
        ttk.Button(self.frame, textvariable = Trad('Annuler'), command = self.fermer_fenetre).grid(row = 2, column = 0, columnspan = 2, sticky = 'nsew', pady = 3, padx = 3)
        ttk.Button(self.frame, textvariable = Trad('OK'), command = self.lancer).grid(row = 2, column = 2, columnspan = 1, sticky = 'nsew', pady = 3, padx = 3)
        
        
    def fermer_fenetre(self):
        self.main.lanceurserveur = None
        self.frame.destroy()
    
    def lancer(self):
        mdp = self.entree.get()
        t = Thread(target = Serveur, args = (mdp, self.main.plans[0], self.main.plans[0].fichier()))
        t.start()
        self.fermer_fenetre()
        
        
class ConnectServeur:
    def __init__(self, main):
        self.main = main
        self.frame = tk.Toplevel()
        self.frame.protocol('WM_DELETE_WINDOW', self.fermer_fenetre)
        self.vars = []
        for i, text in enumerate((Trad('adresse IP'), Trad('Port'), Trad('Mot de passe'))):
            v = tk.StringVar()
            ttk.Entry(self.frame, textvariable = v).grid(row = i + 1, column = 0, sticky = 'nsew', pady = 3, padx = 3)
            ttk.Label(self.frame, textvariable = text).grid(row = i + 1, column = 1, sticky = 'nsew', pady = 3, padx = 3)
            self.vars.append(v)
        ttk.Button(self.frame, text = 'Annuler', command = self.fermer_fenetre).grid(row = 4, column = 0, sticky = 'nsew', pady = 3, padx = 3)
        ttk.Button(self.frame, text = 'OK', command = self.connect).grid(row = 4, column = 1, sticky = 'nsew', pady = 3, padx = 3)
        
    def fermer_fenetre(self):
        self.main.connecteur_serv = None
        self.frame.destroy()
    
    def connect(self):
        ip, port, mdp = [v.get() for v in self.vars]
        self.main.plans[0].connecter_serveur(ip, int(port), mdp)
        self.fermer_fenetre()
        
        
class Parametres:
    
    def __init__(self, fenetre, main, trad, style, params):
        self.fenetre = fenetre
        self.classeTrad = trad
        self.style = style
        self.main = main
        self.params = params
        self.toplevel = tk.Toplevel()
        self.toplevel.columnconfigure(0, weight = 1)
        self.toplevel.rowconfigure(0, weight = 1)
        self.frame = Scrollable_Frame(self.toplevel, row = 0, column = 0, sticky = 'nsew').frame
        for colonne in range(2):
            self.frame.columnconfigure(colonne, weight = 1)
        self.toplevel.protocol('WM_DELETE_WINDOW', self.fermer_fenetre)
        plan = main.plans[0]
        self.p = [('nombre', Trad('Taille des points'), 'BoldP', 3, (0, 20)),
                  ('nombre', Trad('Epaisseur des lignes'), 'BoldC', 3, (0, 20)),
                  ('choix', Trad("Style de l'interface"), 'Style', 'default', style.theme_names()),
                  ('choix', Trad('Langue'), 'Langue', main.langue_def, main.langues),
                  ('couleur', Trad('Couleur des tooltips'), 'ColTooltip', 'gray'),
                  ('nombre', Trad('Delai des tooltips (ms)'), 'TempsTooltip', 300, (0, 5000)),
                  ('couleur', Trad('Couleur des points'), 'ColP', 'green'),
                  ('couleur', Trad('Couleur des courbes'), 'ColC', 'green'),
                  ('texte', Trad('Nom du plan'), plan.nom, 'Plan 1'),
                  ]
        self.widgets = []
        self.valeurs = []
        ttk.Label(self.frame, text = 'Paramètres').grid(row = 0, column = 0, columnspan = 4)
        for i, e in enumerate(self.p):
            if e[0] == 'nombre':
                v = tk.IntVar()
                w = tk.Spinbox(self.frame, textvariable = v, from_ = e[4][0], to = e[4][1])
            if e[0] == 'choix':
                v = tk.StringVar()
                w = ttk.Combobox(self.frame, state = 'readonly', textvariable = v, values = e[4])
            if e[0] == 'texte':
                v = tk.StringVar()
                w = ttk.Entry(self.frame, textvariable = v)
            if e[0] == 'couleur':
                v = tk.StringVar()
                w = ColorChooser(self.frame, self.fenetre, textvariable = v)
            w.grid(row = i+1, column = 0, sticky = 'nsew')
            self.valeurs.append(v)
            self.widgets.append(w)
            ttk.Label(self.frame, textvariable = e[1]).grid(row = i+1, column = 1, sticky = 'nsew')
        f = ttk.Frame(self.frame, padx = 3, pady = 3)
        for colonne in range(3):
            f.columnconfigure(colonne, weight = 1)
        f.grid(row = i+2, column = 0, columnspan = 2)
        ttk.Button(f, textvariable = Trad('Reinitialiser'), command = lambda: self.assigner_valeurs([e[3] for e in self.p])).grid(row = 0, column = 0, sticky = 'nsew')
        ttk.Button(f, textvariable = Trad('   Annuler   '), command = self.fermer_fenetre).grid(row = 0, column = 1, sticky = 'nsew')
        ttk.Button(f, textvariable = Trad('     OK     '), command = self.changer_param).grid(row = 0, column = 2, sticky = 'nsew')
        self.assigner_valeurs([params[e[2]] for e in self.p[:-1]]+[plan.nom])
            
    def fermer_fenetre(self):
        self.main.parametres = None
        self.toplevel.destroy()
    
    def maj(self):
        pass
    
    def assigner_valeurs(self, liste):
        for v, w in zip(liste, self.valeurs):
            w.set(v)
    
    def changer_param(self):
        l = [w.get() if e[0] == 'couleur' else v.get() for v, w, e in zip(self.valeurs, self.widgets, self.p)]
        for i, e in enumerate(self.p[:-1]):
            self.params[e[2]] = l[i]
        plan = self.main.plans[0]
        self.main.menub.configure(text = f'{l[-1]}')
        plan.boldP, plan.boldC, plan.nom = l[0], l[1], l[-1]
        self.style.theme_use(l[2])
        self.fermer_fenetre()
        self.classeTrad.set_lang(self.params['Langue'])
        self.main.langue = self.params['Langue']
        
                
class Notes:
    
    def __init__(self, fenetre, main, separe = 1):
        self.fenetre = fenetre
        if separe:
            self.grande_frame = tk.Toplevel()
            self.frame = self.grande_frame
            self.grande_frame.protocol('WM_DELETE_WINDOW', self.fermer_fenetre)
        else :
            self.grande_frame = tk.Frame(fenetre, bg = '#ddd')
            self.grande_frame.grid(row = 1, column = 1, sticky = 'ns')
            self.frame = tk.Frame(self.grande_frame, bg = '#ddd')
            self.frame.grid(row = 0, column = 0)
            ttk.Button(self.grande_frame, textvariable = Trad('fermer'), command = self.supprimer, bg = '#ddd').grid(row = 1, column = 0)
            fenetre.bind('<Return>', self.clic_entree)
        self.texte = tk.Text(self.frame)
        self.texte.grid(row = 0, column = 0, sticky = 'nsew')
    
    def fermer_fenetre(self):
        pass
    
    def maj(self):
        pass
    
