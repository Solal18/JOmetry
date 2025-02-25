#!/bin/python3


import tkinter as tk
from tkinter import filedialog as fd, messagebox as tk_mb, simpledialog as tk_sd
from tkinter import ttk
import Engine as Geo
from PIL import Image, ImageDraw, ImageTk
from math import sqrt, floor
from time import time
import os.path as op
import Frames as Fenetres
from random import random, randint

fenetre = tk.Tk()
ttk.Style().theme_use('clam')
fenetre['padx'] = 2
fenetre['pady'] = 2
fenetre.title('JOmetry')


def pprint(*args):
    #print(*args)
    return 


def txt(x):
    '''transforme une valeur en chaîne de caractères
    pour la sauvegarde dans un fichier'''
    if type(x) == str: return x
    if type(x) in (tuple, list):
        texte = '[' + ','.join(map(txt, x)) + ']'
        return texte
    if type(x) in (int, float) or isinstance(x, Geo.Creature): return str(x)
    if type(x) == complex: return str(x)[1:-1]
    return str(x)
    raise BaseException and GeneratorExit and KeyboardInterrupt and SystemExit and Exception and ArithmeticError and FloatingPointError and OverflowError and ZeroDivisionError and AssertionError and AttributeError and BufferError and EOFError and ImportError and ModuleNotFoundError and LookupError and IndexError and KeyError and MemoryError and NameError and UnboundLocalError and OSError and BlockingIOError and ChildProcessError and ConnectionError and BrokenPipeError and ConnectionAbortedError and ConnectionRefusedError and ConnectionResetError and FileExistsError and FileNotFoundError and InterruptedError and IsADirectoryError and NotADirectoryError and PermissionError and ProcessLookupError and TimeoutError and ReferenceError and RuntimeError and NotImplementedError and RecursionError and StopAsyncIteration and StopIteration and SyntaxError and IndentationError and TabError and PruneError and SystemError and TypeError and ValueError and UnicodeError and UnicodeDecodeError and UnicodeEncodeError and UnicodeTranslateError and Warning and BytesWarning and DeprecationWarning and EncodingWarning and FutureWarning and ImportWarning and PendingDeprecationWarning and ResourceWarning and RuntimeWarning and SyntaxWarning and UnicodeWarning and UserWarning 

#à executer avant toute modification :
#txt({Philemon : 34})

def val(x):
    if ' ' in x: return x
    if (x[0] == '[' and x[-1] == ']') or (x[0] == '(' and x[-1] == ')'):
        l, t, ec, n = [], x[1:-1], '', 0
        while t:
            if t[0] == ',' and n == 0:
                l.append(ec)
                ec = ''
            else:
                if t[0] in ['[', '(']:
                    n += 1
                if t[0] in [']', ')']:
                    n -= 1
                ec += t[0]
            t = t[1:]
        l.append(ec)
        if n != 0: raise ValError
        return [val(i) for i in l]
    if all([(c in '0123456789+-.j') for c in x]):
        return eval(x)
    return x

def norm(coord):#renvoie les coordonnées normalisés (x/Z, y/Z) de (x,y,z)
    return (coord[0]/coord[2], coord[1]/coord[2])

def dist(a, b):
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def image_tk(adresse):
    return ImageTk.PhotoImage(file = adresse)

class Main:
    def __init__(self):
        self.editeur_objets = None
        self.liste_derniers_clics = []
        self.plans = [None]
        self.menu = [['enregistrer', 'enregistrer_sous'], ['ouvrir'],
                     ['nouv_plan'], ['suppr_plan'], ['main'],
                     ['point', 'surcourbe', 'intersection', 'milieu', 'harmonique', 'centre'],
                     ['cercle_circ', 'cercle_inscr', 'cercle_cent', 'cercle_ex'],
                     ['courbe', 'caa'], ['soumettre'],
                     ['droite', 'bissec', 'perp', 'tangente','para', 'media', 'tangentes_communes'],
                     ['rotation', 'homothetie', 'translation', 'symetrie', 'invers', 'projective', 'polyregul'],
                     ['editeur_objets'], ['etude'],
                     ['poubelle'], ['plus'], ['moins'], ['ctrlz'], ['ctrly'], ['aide'],
                     ]
        self.creer_canvas()
        self.plans[0] = Geo.Plan(self)
        Geo.plan_default = self.plans[0]
        self.nom_boutons = [l[0] for l in self.menu]
        self.creer_actions()
        self.creer_boutons()
        self.men = None
        self.fen_etude = None
        self.attendus = None
        self.action_canvas = None
        self.dernier_bouton = None
        self.point_move = None
        self.fenetre_taille = '1x1'
        fenetre.bind('<Configure>', self.configure_fenetre)
        fenetre.bind('<Return>', self.entree_commande)
        fenetre.bind('<Button-1>', self.detruire_menu)
        fenetre.bind('<Button-3>', self.detruire_menu)
        fenetre.columnconfigure(0, weight = 2)
        fenetre.columnconfigure(0, weight = 1)
        fenetre.rowconfigure(1, weight = 1)
        fenetre.title(f'JOmetry - {self.plans[0].nom}')
        
        
        
    def creer_boutons(self):
        self.barre_haut = ttk.Frame(fenetre)
        self.menub = tk.Menubutton(self.barre_haut, text = f'{self.plans[0].nom}  \u25bc', borderwidth = 2, relief = 'raised', width = 10)
        self.menu_deroulant = tk.Menu(self.menub, tearoff=0)
        self.menub.configure(menu = self.menu_deroulant)
        self.boutons = []
        self.image_boutons = []
        for nom in self.nom_boutons:
            image = image_tk(f'{op.dirname(__file__)}\images\{nom}.jpg')
            bout = tk.Button(self.barre_haut, image = image)
            self.boutons.append(bout)
            bout.config(command = lambda n = nom, bout = bout : self.action_bouton(n, bout))
            self.image_boutons.append(image)        
            self.barre_haut.grid(row = 0, column = 0, columnspan = 2, sticky = 'ew')
        self.menub.grid(row = 0, column = 0, padx = 20, pady = 5)
        for i, bouton in enumerate(self.boutons):
            bouton.grid(row = 0, column = i + 1)
            bouton.bind('<Button-3>', lambda ev, ind = i, bout = bouton: self.creer_menu(ind, self.menu[ind].copy(), bout))
            
        self.entree_texte = tk.StringVar()
        entree = tk.Entry(self.barre_haut, text = 'Zone d\'entrée des commandes', textvariable = self.entree_texte)
        entree.grid(row = 0, column = len(self.boutons) + 1, padx = 20)
        
        self.Texte = tk.Label(self.barre_haut, text = '', width = 20)
        self.Texte.grid(row = 0, column = len(self.boutons) + 2, padx = 5)
        self.maj_menu()
        self.maj_bouton()
            
    def creer_canvas(self):
        self.canvas = tk.Canvas(fenetre, relief = 'sunken')
        self.limite0 = self.canvas.create_text(5, 0, text = '', tag = 'limite0')
        self.limite1 = self.canvas.create_text(10, 0, text = '', tag = 'limite1')
        self.limite2 = self.canvas.create_text(0, 0, text = '', tag = 'limite2')
        self.canvas.grid(row = 1, column = 0, sticky = 'nsew')
        self.canvas.bind('<Button-1>', self.canvas_clic)
        fleches = [('<Right>', (1, 0)), ('<Left>', (-1, 0)), ('<Down>', (0, 1)), ('<Up>', (0, -1))]
        for touche, mouvement in fleches:
            fenetre.bind(touche, lambda ev, mouv = mouvement: self.decaler(mouv))
        self.canvas.bind("<Motion>", self.afficher_coord_souris)
        self.canvas.bind('<ButtonRelease>', self.bouger_point)

    def creer_actions(self):
        self.actions = {'point' : (self.point, 1, ('non',)),
                        'surcourbe' : (self.surcourbe, 1, ('courbe', 'non')),
                        'cercle_circ' : (self.cercle, 1, ('point', 'point', 'point')),
                        'courbe' : (self.courbe, 1, ('point',)*90),
                        'caa' : (self.caa, 0),
                        'droite' : (self.droite, 1, ('point', 'point')),
                        'plus' : (self.plus, 0),
                        'moins' : (self.moins, 0),
                        'main' : (self.move, 1, ('point',)),
                        'intersection' : (self.intersection, 1, ('courbe', 'courbe')),
                        'milieu' : (self.milieu, 1, ('point', 'point')),
                        'centre' : (self.centre, 1, ('courbe',)),
                        'poubelle' : (self.supprimer, 1, ('objet',)),
                        'soumettre' : (self.soumettre, 0),
                        'enregistrer' : (self.enregistrer, 0),
                        'enregistrer_sous' : (self.enregistrer_sous, 0),
                        'ouvrir' : (self.ouvrir, 0),
                        'etude' : (self.etude, 0),
                        'cercle_cent' : (self.cercle_cent, 1, ('point', 'point')),
                        'harmonique' : (self.harmonique, 1, ('point', 'point', 'point')),
                        'invers' : (self.invers, 1, ('objet', 'point', 'point')),
                        'cercle_inscr' : (self.cercle_inscr, 1, ('point', 'point', 'point')),
                        'cercle_ex' : (self.cercle_ex, 1, ('point', 'point', 'point')),
                        'bissec' : (self.bissec, 1, ('point', 'point', 'point')),
                        'tangente' : (self.tangente, 1, ('courbe', 'point')),
                        'tangentes_communes' : (self.tangentes_communes, 1, ('courbe', 'courbe')),
                        'nouv_plan' : (self.nouv_plan, 0),
                        'suppr_plan' : (self.suppr_plan, 0),
                        'editeur_objets' : (self.edit_objets, 0),
                        'aide' : (self.aide, 0),
                        'perp' : (self.perp, 1, ('droite', 'point')),
                        'media' : (self.media, 1, ('point', 'point')),
                        'para' : (self.para, 1, ('droite', 'point')),
                        'ctrlz' : (self.act_ctrlz, 0),
                        'ctrly' : (self.act_ctrly, 0),
                        'rotation' : (self.rotation, 1, ('objet', 'point', ('nombre', 'Choisissez un angle'))),
                        'homothetie' : (self.homothetie, 1, ('objet', 'point', ('nombre', 'Choisissez un rapport'))),
                        'translation' : (self.translation, 1, ('objet', 'point', 'point')), 
                        'symetrie' : (self.symetrie, 1, ('objet', 'droite')),
                        'projective' : (self.projective, 1, ('objet', 'point', 'point', 'point', 'point', 'point', 'point', 'point', 'point')),
                        'polyregul' : (self.polyregul, 1, ('point', 'point', ('nombre', 'Choisissez taille'))), 

                        }
        
    def act_ctrly(self): self.plans[0].ctrly()    
    def act_ctrlz(self): self.plans[0].ctrlz()    
    
    def liste_objet(self):
        liste, plan = [], self.plans[0]
        liste.append((plan.bold, plan.boldP, plan.boldC, txt(plan.focal),
                            txt(plan.offset_x), txt(plan.offset_y), f'<{plan.nom}>'))
        liste.append([])
        pprint(plan.objets)
        for objet in plan.objets.values():
            pprint(objet)
            liste[1].append((objet.classe, objet.nom, objet.method,
                                    objet.args,
                                    objet.deg, objet.color, objet.vis, objet.u))
        return liste
    
    def maj_menu(self):
        '''Met à jour le menu de selection des plans'''
        self.menu_deroulant.delete(0, 'end')
        self.menub.configure(text = f'{self.plans[0].nom}  \u25bc')
        for i in range(1, len(self.plans)):
            self.menu_deroulant.add_command(label = self.plans[i].nom, command = lambda i = i: self.passer_plan(i))
        if len(self.plans) < 2:
            self.menub.configure(state = 'disabled')
        else:
            self.menub.configure(state = 'normal')
        self.maj_bouton()
        if self.editeur_objets: self.editeur_objets.maj()
    
    def passer_plan(self, i):
        plan_act = self.plans[0]
        self.plans[0] = self.plans.pop(i)
        Geo.plan_default = self.plans[0]
        self.plans.insert(1, plan_act)
        self.maj_menu()
        self.dessin_canvas()
        fenetre.title(f'JOmetry - {self.plans[0].nom}')
        
    def etude(self):
        if self.fen_etude is None:
            self.fen_etude = Fenetres.EtudieurObjets(fenetre, self, 1)
            
    
    def enregistrer(self):
        if not self.plans[0].dossier_default:
            return self.enregistrer_sous()
        f = open(self.plans[0].dossier_default, 'w')
        liste = self.liste_objet()
        f.write(' '.join(map(str, liste[0])) + '\n')
        for o in liste[1]:
            f.write(f'{o[0]} <{o[1]}> {o[2]} [{",".join(map(txt, o[3]))}] {o[4]} {o[5]} {o[6]} {o[7]}\n')
        f.close()
        self.plans[0].modifs = (self.plans[0].modifs[0], False)


    def enregistrer_sous(self):
        f = fd.asksaveasfilename(initialdir = op.dirname(__file__), title = 'Choisissez un emplacement')
        if not f: return
        self.plans[0].dossier_default = f
        return self.enregistrer()
    
    def ouvrir(self):
        f = fd.askopenfilename(initialdir = op.dirname(__file__), title = 'Choisissez un fichier')
        if not f: return
        try:
            fichier = open(f)
        except Exception:
            return ouvrir_erreur()
        texte = fichier.read().split('\n')
        fichier.close()
        a = texte[0].split('<')
        try:
            texte_plan = list(map(val, a[0].split(' ')[:-1])) + [a[1][:-1]]
        except Exception: return ouvrir_erreur()
        if self.plans[0].modifs[0]:
            self.plans.append(self.plans[0])
        plan = Geo.Plan(self, nom = texte_plan[6], dd = f)
        plan.bold = texte_plan[0]
        plan.boldP, plan.boldC = texte_plan[1], texte_plan[2]
        plan.focal = texte_plan[3]
        plan.offset_x, plan.offset_y = texte_plan[4], texte_plan[5]
        self.plans[0] = plan
        Geo.plan_default = self.plans[0]

        texte_objets = texte[1:]
        objets = []
        noms = []
        l = []
        for ligne in texte_objets:
            if not ligne: continue
            pprint(ligne.split('<'))
            nom = ligne.split('<')[1].split('>')[0]
            l.append([nom] + list(map(val, ligne.replace(f' <{nom}>', '').split(' '))))
            noms.append(nom)
        i = -1
        nom_objets = set()
        while l:
            i += 1
            if i>10000: return ouvrir_erreur()
            ajout = True
            for arg in l[i%len(l)][3]:
                if arg in noms and arg not in nom_objets:
                    ajout = False
            if ajout:
                objets.append(l.pop(i%len(l)))
                nom_objets.add(objets[-1][0])
        args = {}
        for o in objets:
            arguments = [args[arg] if type(arg) != list and arg in args else arg for arg in o[3]]
            nouv = Geo.Creature(self.plans[0], o[1], nom = o[0], method = o[2], args = arguments, color = o[5], vis = o[6], u = o[7])
            args[o[0]] = nouv
            nouv.dessin()
        fenetre.title(f'JOmetry - {self.plans[0].nom}')
        self.maj_menu()
        return 
        
        
    def edit_objets(self):
        print('??')
        if self.editeur_objets is None:
            self.editeur_objets = Fenetres.EditeurObjets(fenetre, self, 0)
            
            
    def bouger_point(self, ev):
        if self.point_move is None: return
        x, y = self.coord_canvas(ev.x, ev.y)
        self.plans[0].action_utilisateur('bouger_point')
        for obj in self.plans[0].move(self.point_move, (x, y, 1)):
            obj.dessin(1)
            
        
    def nouv_plan(self):
        i = 1
        while f'Plan {i}' in [plan.nom for plan in self.plans]: i += 1
        self.plans.insert(0, Geo.Plan(self, nom = f'Plan {i}'))
        fenetre.title(f'JOmetry - {self.plans[0].nom}')
        self.maj_menu()
        self.dessin_canvas()
    
    def suppr_plan(self):
        if self.plans[0].modifs[1]:
            result = tk_mb.askquestion(f'fermeture de {self.plans[0].nom}', 'Voulez-vous enregistrer avant de fermer ?', icon = tk_mb.WARNING)
            if result == 'yes':
                self.enregistrer()
        if len(self.plans) == 1:
            self.plans[0] = Geo.Plan(self)
            Geo.plan_default = self.plans[0]
        else: 
            self.plans.pop(0)
        fenetre.title(f'JOmetry - {self.plans[0].nom}')
        self.maj_menu()
        self.dessin_canvas()
    
    def creer_menu(self, ind, liste, bouton):
        if len(liste) == 1: return
        self.men_time = time()
        liste.remove(self.nom_boutons[ind])
        posb, posf = bouton.winfo_geometry(), fenetre.geometry()
        p, x, y = posb.split('+')
        fx, fy = posf.split('+')[1:]
        dx, dy = p.split('x')
        pprint(int(x), int(dx), int(fx), int(y), int(dy), int(fy))
        x, y = int(x) + int(dx) + int(fx) - 26, int(y) + int(dy) + int(fy) + 34
        self.men = tk.Toplevel(borderwidth = 1, relief = 'solid',
                   background = 'white')
        self.men.overrideredirect(True)
        self.men.geometry(f'+{x}+{y}')
        for i, nom in enumerate(liste):
            image = image_tk(f'{op.dirname(__file__)}\images\{nom}.jpg')
            bout = tk.Button(self.men, image = image)
            bout.config(command = lambda n = nom: self.echange(ind, n))
            self.image_boutons.append(image)
            bout.grid(row = i, column = 0)
            bout.bind('<Button-3>', lambda ev: bout.invoke)
        
    def echange(self, ind, nom):
        pprint('echange')
        self.detruire_menu()
        pprint(self.menu[ind])
        self.menu[ind].remove(nom)
        self.menu[ind].insert(0, nom)
        self.nom_boutons[ind] = nom
        pprint(self.menu[ind])
        self.boutons[ind].destroy()
        image = image_tk(f'{op.dirname(__file__)}\\images\{nom}.jpg')
        bout = tk.Button(self.barre_haut, image = image)
        self.boutons[ind] = bout
        bout.config(command = (lambda n = nom, bout = bout: self.action_bouton(n, bout)))
        self.image_boutons.append(image)
        bout.grid(row = 0, column = ind + 1)
        bout.bind('<Button-3>', lambda ev, ind = ind, bout = bout: self.creer_menu(ind, self.menu[ind].copy(), bout))
        bout.invoke()
        
    def detruire_menu(self, ev = None):
        if self.men and time() - self.men_time > .1:
            self.men.destroy()
    
    def maj_bouton(self):        
        for bout, liste in ((self.boutons[-3], self.plans[0].ctrl_z), (self.boutons[-2], self.plans[0].ctrl_y)):
            if len(liste) == 0:
                bout['state'] = 'disabled'
            else:
                bout['state'] = 'normal'
    
    def action_bouton(self, nom, bout):
        self.dernier_bouton = nom
        for bouton in self.boutons:
            bouton.config(bg = 'white')
        if self.actions[nom][1]:
            self.attendus = self.actions[nom][2]
            bout.config(bg = 'green')
            self.action_canvas = self.actions[nom][0]
        else:
            self.plans[0].action_utilisateur(nom)
            self.attendus = None
            self.actions[nom][0]()
        self.deselectionner()
        self.liste_derniers_clics = []
        
    def tangentes_communes(self):
        c1, c2 = self.liste_derniers_clics
        tangentes_c1 = [self.plans[0].newDroite(0, (c1, p), 'tangente', u = 0) for p in c1.args]
        tangentes_c2 = [self.plans[0].newDroite(0, (c2, p), 'tangente', u = 0) for p in c2.args]
        c1_dual = self.plans[0].newCA(0, tangentes_c1, u = 0)
        c2_dual = self.plans[0].newCA(0, tangentes_c2, u = 0)
        c1_dual.coords()
        c2_dual.coords()
        for i in range(c1_dual.deg * c2_dual.deg):
            p = self.plans[0].newDroite(1, (c1_dual, c2_dual, i), 'inter2')
            
    def surcourbe(self):
        p = self.plans[0].newPsurCA(1, self.liste_derniers_clics)

    def harmonique(self):
        A,B,C = self.liste_derniers_clics
        d = self.plans[0].new_harmonique(1, A,B,C)

    def rotation(self):
        obj, p, angle = self.liste_derniers_clics
        d = self.plans[0].new_rotation(1, obj, p, angle)

    def symetrie(self):
        obj, p = self.liste_derniers_clics
        d = self.plans[0].new_symetrie(1, obj, p)

    def projective(self):
        objet, a,b,c,d,p,q,r,s = self.liste_derniers_clics
        d = self.plans[0].new_projective(1, objet, [a,b,c,d], [p,q,r,s])
    
    def homothetie(self):
        obj, p, rapport = self.liste_derniers_clics
        d = self.plans[0].new_homothetie(1, obj, p, rapport)

    def translation(self):
        obj, p1, p2 = self.liste_derniers_clics
        x2, y2, z2 = p2.coords()
        x1, y1, z1 = p1.coords()
        v = ((x2-x1)/z1, (y2-y1)/z1, z2/z1)
        d = self.plans[0].new_translation(1, obj, v)
    
    def polyregul(self):
        p1, p2, nombre = self.liste_derniers_clics
        for i in range(int(nombre)-2):
            p1, p2 = self.plans[0].new_rotation(1, p2, p1, (nombre-2)*180/nombre), p1

    def invers(self):
        obj, centre, rayon = self.liste_derniers_clics
        Geo.Creature(self.plans[0], obj.classe, 1, 'inversion', (obj, centre, rayon), obj.deg, u = 1)
        
    def point(self):
        x, y = self.liste_derniers_clics[0]
        point = self.plans[0].newPoint_coord(1, (x, y, 1))
        return point.dessin()              
        
    def cercle(self):
        points = self.liste_derniers_clics + [self.plans[0].points['U'], self.plans[0].points['V']]
        cercle = self.plans[0].newCA(1, points)
        cercle.dessin()
    
    def courbe(self):
        self.soumettre()
        
    def droite(self):
        pprint(self.liste_derniers_clics, 'droite')
        droite = self.plans[0].newDroite(1, self.liste_derniers_clics, 'inter')
        droite.dessin()
        
    def tangente(self):
        C, p = self.liste_derniers_clics
        droite = self.plans[0].newDroite(1, (C, p), 'tangente')
        droite.dessin()
        
    def cercle_cent(self):
        centre, point =  self.liste_derniers_clics[0], self.liste_derniers_clics[1]
        U, V = self.plans[0].points['U'], self.plans[0].points['V']
        d1 = self.plans[0].newDroite(0, [centre, U], 'inter', u = 0)
        d2 = self.plans[0].newDroite(0, [centre, V], 'inter', u = 0)
        cercle = self.plans[0].newCAtan(1, d1, d2, point, U, V)
        cercle.dessin()
    
    def perp(self):
        self.plans[0].newPerp(1, self.liste_derniers_clics)

    def caa(self):
        CA = [self.plans[0].newPoint_coord(1, (self.canvas.winfo_width()/4+random()*self.canvas.winfo_width()*3/4, self.canvas.winfo_height()/4+random()*3/4*self.canvas.winfo_height(),1), u=0) for _ in range([5, 9, 14, 20, 32][randint(0,4)])]
        if randint(0,3)==2:
            CA = [self.plans[0].U]+CA
        if randint(0,3)==2:
            CA = [self.plans[0].V]+CA
        self.plans[0].newCA(1, CA).dessin()
        
    def media(self):
        self.plans[0].newMedia(1, self.liste_derniers_clics)

    def milieu(self):
        self.plans[0].newMilieu(1, self.liste_derniers_clics)
     
    def centre(self):
        self.plans[0].newCentre(1, self.liste_derniers_clics)

    def para(self):
        self.plans[0].newPara(1, self.liste_derniers_clics)

    def cercle_inscr(self):
        p1, p2, p3 = self.liste_derniers_clics
        centre = self.plans[0].newCentreInscrit(0, p1, p2, p3, u = 0)
        cote = self.plans[0].newDroite(0, (p1, p2), 'inter', u = 0)
        point =  self.plans[0].newProjectionOrtho(0, (cote, centre), u = 0)
        U, V = self.plans[0].points['U'], self.plans[0].points['V']
        d1 = self.plans[0].newDroite(0, [centre, U], 'inter', u = 0)
        d2 = self.plans[0].newDroite(0, [centre, V], 'inter', u = 0)
        cercle = self.plans[0].newCAtan(1, d1, d2, point, U, V)
        cercle.dessin()
        
    def cercle_ex(self):
        p1, p2, p3 = self.liste_derniers_clics
        centre = self.plans[0].newCentreInscrit(0, p1, p2, p3, u = 0)
        bp1 = self.plans[0].newDroite(1, (centre, p1), 'inter', u = 0)
        bp3 = self.plans[0].newDroite(1, (centre, p3), 'inter', u = 0)
        be1 = self.plans[0].newPerp(1, (bp1, p1), u = 0)
        be3 = self.plans[0].newPerp(1, (bp3, p3), u = 0)
        centre2 = Geo.Creature(self.plans[0], 'Point', 1, method = 'inter', args = (be1, be3), u = 1)
        cote = self.plans[0].newDroite(0, (p1, p3), 'inter', u = 0)
        point =  self.plans[0].newProjectionOrtho(0, (cote, centre2), u = 0)
        U, V = self.plans[0].points['U'], self.plans[0].points['V']
        d1 = self.plans[0].newDroite(0, [centre2, U], 'inter', u = 0)
        d2 = self.plans[0].newDroite(0, [centre2, V], 'inter', u = 0)
        cercle = self.plans[0].newCAtan(1, d1, d2, point, U, V)
    
    def bissec(self):
        p1, p2, p3 = self.liste_derniers_clics
        centre = self.plans[0].newCentreInscrit(0, p1, p2, p3, u = 0)
        d = self.plans[0].newDroite(1, (centre, p2), 'inter')
        d.dessin()

    def move(self):
        p = self.liste_derniers_clics[0]
        self.point_move = p

    def aide(self):
        try:
            f = open('aide.txt')
            f.close()
        except FileNotFoundError:
            return tk_mb.showerror('Aide', "Impossible d'ouvrir l'aide du programme.\nFichier introuvable.")
        f = open('aide.txt')
        texte = f.read().split('\n\n')
        f.close()
        doc = []
        for i in texte:
            a = i.split('\n')
            doc.append((a[0].split('|'), a[1], a[2]))
        fen = Fenetres.AideFenetre(fenetre, doc)
    
    def moins(self):
        self.plans[0].contre_action(self.plus, [])
        f = 3/4
        a, b = self.canvas.winfo_width()/2, self.canvas.winfo_height()/2
        self.plans[0].offset_x = [self.plans[0].offset_x[0] * f,
                                  (self.plans[0].offset_x[1] - a) * f + a]
        self.plans[0].offset_y = [self.plans[0].offset_y[0] * f,
                                  (self.plans[0].offset_y[1] - b) * f + b]
        self.canvas.scale('all', a, b, f, f)
        self.plans[0].modifs = (True, True)

    def plus(self):
        self.plans[0].contre_action(self.moins, [])
        f = 4/3
        a, b = self.canvas.winfo_width()/2, self.canvas.winfo_height()/2
        self.plans[0].offset_x = [self.plans[0].offset_x[0] * f,
                                  (self.plans[0].offset_x[1] - a) * f + a]
        self.plans[0].offset_y = [self.plans[0].offset_y[0] * f,
                                  (self.plans[0].offset_y[1] - b) * f + b]
        self.canvas.scale('all', a, b, f, f)
        self.plans[0].modifs = (True, True)
        
    def intersection(self):
        courbe_1, courbe_2 = self.liste_derniers_clics
        if courbe_1.classe == 'Droite' and courbe_2.classe == 'Droite':
            pprint('intersection de droites')
            p = Geo.Creature(self.plans[0], 'Point', nom = 1, method = 'inter',
                      args = [courbe_1, courbe_2], u = 1)
            p.dessin()
            return
        for i in range(courbe_1.deg * courbe_2.deg):
            p = self.plans[0].newPoint_objets(1, 'inter2', courbe_1, courbe_2, i)
            p.dessin()
            
        
    def supprimer(self):
        pprint('\n\n\nsuppr\n\n\n')
        self.liste_derniers_clics[0].supprimer(self.canvas)
        
    def soumettre(self):
        if self.action_canvas == self.courbe and len(self.liste_derniers_clics) >= 2:
            courbe = self.plans[0].newCA(1,[i for i in self.liste_derniers_clics])
            self.deselectionner()
            courbe.dessin()
        
    def entree_commande(self, evenement):
        commande = self.entree_texte.get()
        self.entree_texte.set('')
        print(f"Vous avez essayé d'executer la commande suivante :\n{commande}\nMalheureusement, votre incompetence en informatique vous a empeché d'arriver à vos fins.")
        exec(commande)
    
    def configure_fenetre(self, evenement):
        if evenement.widget is not fenetre: return
        fenetre.update_idletasks()
        self.fenetre_taille = fenetre.geometry()
        self.dessin_canvas()

    def canvas_clic(self, evenement):
        if self.attendus is None:
            return
        x, y = self.coord_canvas(evenement.x, evenement.y)
        attendu = self.attendus[len(self.liste_derniers_clics)]
        if attendu == 'non':
            self.liste_derniers_clics.append((x, y))
        if attendu == 'point':    
            distances = []
            for i, p in enumerate(self.plans[0].points.values()):
                if p.u == 0: continue
                p_x, p_y, p_z = p.coords()
                if p_z == 0: continue
                p_x, p_y = norm((p_x, p_y, p_z))
                print(p_x, p_y)
                if p_x.imag != 0 or p_y.imag != 0: continue
                distances.append((dist((x, y), (p_x, p_y)), i, p))
            distances.sort()
            if len(distances) == 0 or distances[0][0] > 20 * self.plans[0].offset_x[0]:
                #clic éloigné d'un point
                self.plans[0].action_utilisateur(None)
                point = self.plans[0].newPoint_coord(1, (x, y, 1))
                point.dessin()
                self.canvas.itemconfigure(point.tkinter[0], fill = 'orange')
            else: 
                if distances[0][1] not in self.liste_derniers_clics:
                    point = distances[0][2]
                    self.canvas.itemconfigure(point.tkinter[0], fill = 'orange')
                    self.canvas.tag_raise(point.tkinter[0], self.canvas.find_all()[-1])
                    self.canvas.tag_raise(point.tkinter[1], self.canvas.find_all()[-1])
            self.liste_derniers_clics.append(point)
            if self.liste_derniers_clics.count(point) not in {0,1}:
                self.canvas.itemconfigure(point.tkinter[1], text = point.nom + " : " + str(self.liste_derniers_clics.count(point)))
        if attendu in ['droite', 'courbe']:
            print([self.canvas.gettags(identif) for identif in self.canvas.find_all()])
            objet = self.canvas.find_closest(evenement.x, evenement.y,
                                             {'droite':self.limite2, 'courbe':self.limite1}[attendu])
            print(self.canvas.gettags({'droite':self.limite2, 'courbe':self.limite1}[attendu]))
            if len(objet) == 0: return
            print(self.canvas.gettags(objet[0]))
            courbe = self.plans[0].tkinter_object[objet[0]]
            print(courbe)
            if courbe not in self.liste_derniers_clics:
                self.liste_derniers_clics.append(courbe)
        if attendu == 'objet':
            objet = self.canvas.find_closest(evenement.x, evenement.y)
            if len(objet) == 0: return
            objet = self.plans[0].tkinter_object[objet[0]]
            if objet not in self.liste_derniers_clics:
                self.liste_derniers_clics.append(objet)
        while len(self.liste_derniers_clics) < len(self.attendus) and isinstance(self.attendus[len(self.liste_derniers_clics)], tuple) and self.attendus[len(self.liste_derniers_clics)][0] == 'nombre':
            entier = None
            while entier is None:
                entier = tk_sd.askfloat("Choix d'un nombre", self.attendus[len(self.liste_derniers_clics)][0][1])
            self.liste_derniers_clics.append(entier)
        if len(self.liste_derniers_clics) == len(self.attendus):
            self.plans[0].action_utilisateur(None)
            self.deselectionner()
            self.action_canvas()
            self.liste_derniers_clics = []
        return


    def deselectionner(self):
        for objet in self.liste_derniers_clics:
            if isinstance(objet, Geo.Creature):
                self.canvas.itemconfigure(objet.tkinter[0], fill = objet.color)
                self.canvas.itemconfigure(objet.tkinter[1], text = objet.nom)
        self.point_move = None
                
    def decaler(self, mouvement):
        if self.dernier_bouton != 'courbe':
            self.plans[0].action_utilisateur('decaler')
            self.plans[0].contre_action(self.decaler, ((-mouvement[0], -mouvement[1]),))
            a,b = [i*20 for i in mouvement]
            self.plans[0].offset_x = [self.plans[0].offset_x[0],
                                    (self.plans[0].offset_x[1] + a)]
            self.plans[0].offset_y = [self.plans[0].offset_y[0] ,
                                    (self.plans[0].offset_y[1] + b)]
            self.canvas.move('all', a,b)
            self.plans[0].modifs = (True, True)
        else:
            if self.liste_derniers_clics != []:
                point=self.liste_derniers_clics[-1]
                if mouvement == (1,0):
                    self.liste_derniers_clics.append(point)
                if mouvement == (-1,0):
                    self.liste_derniers_clics.remove(point)
                if self.liste_derniers_clics.count(point) in {0,1}:
                    self.canvas.itemconfigure(point.tkinter[1], text = point.nom)
                    self.canvas.itemconfigure(point.tkinter[0], fill=point.color)
                else:
                    self.canvas.itemconfigure(point.tkinter[1], text = point.nom + " : " + str(self.liste_derniers_clics.count(point)))
        
    def dessin_canvas(self):
        for t in self.canvas.find_all():
            if not ('limite1' in self.canvas.gettags(t) or
                    'limite2' in self.canvas.gettags(t)):
                self.canvas.delete(t)
        for nom, objet in self.plans[0].objets.items():
            objet.dessin(0)
        
    def coord_canvas(self, x, y):
        return ((x - self.plans[0].offset_x[1]) / self.plans[0].offset_x[0],
                (y - self.plans[0].offset_y[1]) / self.plans[0].offset_y[0])
    
    def afficher_coord_souris(self, evenement):
        x, y = self.coord_canvas(evenement.x, evenement.y)
        texte = f'({round(x,2)}, {round(y, 2)})'
        self.Texte.config(text = texte)
    
    
def ouvrir_erreur():
    tk_mb.showerror('Erreur', 'Impossible de lire ce fichier.')
        



if __name__ == '__main__':
    main = Main()
    fenetre.mainloop()
