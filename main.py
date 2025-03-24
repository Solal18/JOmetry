#!/bin/python3


print('Chargement  ', end = '')

import threading
from time import sleep
chargement = True
def animate():
    i = 0
    l = ('|', '/', '-', '\\')
    while chargement:
        c = l[i]
        i = (i+1)%4
        print('\b' + c, end = '', flush = 1)
        sleep(0.1)
    print('\rC\'est bon !')
t = threading.Thread(target=animate)
t.start()
import weakref
import tkinter as tk
from tkinter import filedialog as fd, messagebox as tk_mb, simpledialog as tk_sd
from tkinter import ttk
import Engine as Geo
from PIL import Image, ImageDraw, ImageTk
from math import sqrt, pi
import os.path as op
import Frames as Fenetres
from time import time
from Engine import txt, val
from random import random, randint

fenetre = tk.Tk()
style = ttk.Style()
style.theme_use('default')
fenetre['padx'] = 2
fenetre['pady'] = 2
fenetre.title('JOmetry')

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
        if lang not in langues or mot not in dicos[0]:
            return mot
        if dicos[0][mot] not in dicos[langues.index(lang)]:
            return mot
        return dicos[langues.index(lang)][dicos[0][mot]]
    return langue, langues, trad

try:
    langue, langues, trad = traduction()
except Exception as e:
    print('Impossible de charger les traductions')
    langue = 'Français'
    langues = ['Français']
    trad = lambda x: x

params = {'BoldP':3, 'BoldC':3, 'Style':'default', 'Langue':langue, 'ColTooltip':'gray', 'ColP':'green', 'ColC':'green', 'TempsTooltip':300}
try:
    f = open(f'{op.dirname(__file__)}\\parametres.txt', encoding = 'utf-8')
    charges = val(f.read())
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
            self.noteb[0].tab(self.noteb[1], text = trad(lang, self.mot))
        else:
            self.set(trad(lang, self.mot))
    
    @classmethod
    def set_lang(cls, lang):
        for variable in cls.variables|set(cls.variables_weak):
            variable.langue = lang
        



def norm(coord):#renvoie les coordonnées normalisés (x/Z, y/Z) de (x,y,z)
    return (coord[0]/coord[2], coord[1]/coord[2])

def dist(a, b):
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def image_tk(adresse):
    return ImageTk.PhotoImage(file = adresse)

class Tooltip:
    
    def __init__(self, bout, texte, main):
        bout.bind('<Enter>', self.deb)
        bout.bind('<Motion>', self.deb)
        bout.bind('<Leave>', self.fin)
        self.bout = bout
        self.top = None
        self.texte = Trad(texte, weak = 1)
        self.timer = None
        self.main = main
        
    def deb(self, ev = None):
        self.fin()
        self.timer = fenetre.after(params['TempsTooltip'], self.aff)
    
    def fin(self, ev = None):
        if self.top is not None: self.top.destroy()
        if self.timer is not None: fenetre.after_cancel(self.timer)
    
    def aff(self):
        self.top = tk.Toplevel(borderwidth = 1, relief = 'solid',
                   background = params['ColTooltip'])
        dx, dy = fenetre.winfo_pointerxy()
        self.top.overrideredirect(True)
        self.top.geometry(f'+{dx+15}+{dy+20}')
        ttk.Label(self.top, textvariable = self.texte).grid(row = 0, column = 0)
        

class Main:
    def __init__(self):
        self.editeur_objets = None
        self.liste_derniers_clics = []
        self.plans = [None]
        self.delai_tooltip = 300
        self.onglets = ['Ctrl', 'classiques', 'Points', 'Droites', 'Courbes', 'Transformations', 'Frames']
        self.boutons2 = [['enregistrer', 'enregistrer_sous', 'ouvrir', 'nouv_plan', 'suppr_plan', 'parametres'],
                         ['main', 'point', 'droite', 'cercle_circ', 'courbe', 'soumettre', 'angle'],
                         ['point', 'surcourbe', 'intersection', 'milieu', 'harmonique', 'centre', 'angle'], #Il faudra mettre angle dans une autre categorie
                         ['droite', 'segment', 'bissec', 'perp', 'para', 'media', 'tangente', 'tangentes_communes'],
                         ['courbe', 'soumettre', 'caa', 'cercle_circ', 'cercle_inscr', 'cercle_cent', 'cercle_ex', 'tangente', 'tangentes_communes'],
                         ['rotation', 'homothetie', 'translation', 'symetrie', 'invers', 'projective', 'polyregul', 'inv_plan'],
                         ['editeur_objets', 'etude', 'poubelle', 'connect', 'serveur', 'aide'],
                         ]
        self.creer_canvas()
        self.plans[0] = Geo.Plan(self)
        Geo.plan_default = self.plans[0]
        self.creer_actions()
        self.creer_boutons()
        self.men = None
        self.langue = langue
        self.langue_def = langue_def
        self.langues = langues
        self.fen_etude = None
        self.attendus = None
        self.action_canvas = None
        self.dernier_bouton = None
        self.point_move = None
        self.lanceurserveur = None
        self.parametres = None
        self.fenetre_taille = '1x1'
        self.connecteur_serv = None
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
        self.barre_haut.grid(row = 0, column = 0, sticky = 'ew')
        
        self.frameg = ttk.Frame(self.barre_haut)
        self.framed = ttk.Frame(self.barre_haut)
        self.fen_onglets = ttk.Notebook(self.barre_haut)
        self.frameg.grid(row = 0, column = 0)
        self.framed.grid(row = 0, column = 2, pady = 3, padx = 3)
        self.fen_onglets.grid(row = 0, column = 1, padx = 5)
        
        self.menub = ttk.Menubutton(self.frameg, text = f'{self.plans[0].nom}', width = 10)
        self.menu_deroulant = tk.Menu(self.menub, tearoff=0)
        self.menub.configure(menu = self.menu_deroulant)
        self.menub.grid(row = 0, column = 0, pady = 5)
        self.Texte = ttk.Label(self.frameg, text = '', width = 20, anchor = 'n')
        self.Texte.grid(row = 1, column = 0)
        
        def bouton(f, nom):
            image = image_tk(f'{op.dirname(__file__)}\images\{nom}.jpg')
            bout = ttk.Button(f, image = image)
            self.boutons.append(bout)
            bout.config(command = lambda n = nom, bout = bout : self.action_bouton(n, bout))
            self.image_boutons.append(image)
            Tooltip(bout, nom, self)
            return bout
        
        self.boutons = []
        self.image_boutons = []
        for i, (ong, bouts) in enumerate(zip(self.onglets, self.boutons2)):
            f = ttk.Frame(self.fen_onglets)
            for j, nom in enumerate(bouts):
                bout = bouton(f, nom)
                bout.grid(row = 0, column = j)
            self.fen_onglets.add(f, text = ong)
            Trad(ong, langue, (self.fen_onglets, i))
        
        #self.entree_texte = tk.StringVar()
        #entree = ttk.Entry(self.barre_haut, text = 'Zone d\'entrée des commandes', textvariable = self.entree_texte)
        #entree.grid(row = 0, column = 2, padx = 20)
        
        for nom, x, y in (('ctrlz', 0, 0), ('ctrly', 0, 1), ('plus', 1, 0), ('moins', 1, 1)):
            bout = bouton(self.framed, nom)
            bout.grid(row = x, column = y)
            
        
        self.maj_menu()
        self.maj_bouton()
           
    def creer_canvas(self):
        self.panedwindow = ttk.PanedWindow(fenetre, orient = 'horizontal')
        self.canvas = tk.Canvas(self.panedwindow, relief = 'sunken')
        self.panedwindow.add(self.canvas, weight = 1)
        self.limite0 = self.canvas.create_text(-505, 0, text = ' ', tag = 'limite0')
        self.limite1 = self.canvas.create_text(-510, 0, text = ' ', tag = 'limite1')
        self.limite2 = self.canvas.create_text(-500, 0, text = ' ', tag = 'limite2')
        self.panedwindow.grid(row = 1, column = 0, sticky = 'nsew')
        self.canvas.bind('<Button-1>', self.canvas_clic)
        fleches = [('<Right>', (1, 0)), ('<Left>', (-1, 0)), ('<Down>', (0, 1)), ('<Up>', (0, -1))]
        for touche, mouvement in fleches:
            fenetre.bind(touche, lambda ev, mouv = mouvement: self.decaler(mouv))
        self.canvas.bind('<ButtonRelease>', self.bouger_point)

    def creer_actions(self):
        self.actions = {'point' : (self.point, 1, ('non',)),
                        'surcourbe' : (self.surcourbe, 1, ('courbe', 'non')),
                        'cercle_circ' : (self.cercle, 1, ('point', 'point', 'point')),
                        'courbe' : (self.courbe, 1, ('point',)*90),
                        'droite' : (self.droite, 1, ('point', 'point')),
                        'segment' : (self.segment, 1, ('point', 'point')),
                        'plus' : (self.plus, 0),
                        'moins' : (self.moins, 0),
                        'main' : (self.move, 1, ('point',)),
                        'angle' : (self.angle, 1, ('point', 'point', 'point')),
                        'intersection' : (self.intersection, 1, ('courbe', 'courbe')),
                        'milieu' : (self.milieu, 1, ('point', 'point')),
                        'centre' : (self.centre, 1, ('courbe',)),
                        'poubelle' : (self.supprimer, 1, ('objet',)),
                        'soumettre' : (self.soumettre, 0),
                        'enregistrer' : (self.enregistrer, 0),
                        'enregistrer_sous' : (self.enregistrer_sous, 0),
                        'ouvrir' : (self.ouvrir, 0),
                        'etude' : (self.etude, 0),
                        'serveur' : (self.serveur, 0),
                        'caa' : (self.caa, 0),
                        'cercle_cent' : (self.cercle_cent, 1, ('point', 'point')),
                        'harmonique' : (self.harmonique, 1, ('point', 'point', 'point')),
                        'invers' : (self.invers, 1, ('objet', 'point', 'point')),
                        'inv_plan' : (self.inv_plan, 1, ('point', 'point')),
                        'cercle_inscr' : (self.cercle_inscr, 1, ('point', 'point', 'point')),
                        'cercle_ex' : (self.cercle_ex, 1, ('point', 'point', 'point')),
                        'bissec' : (self.bissec, 1, ('point', 'point', 'point')),
                        'tangente' : (self.tangente, 1, ('courbe', 'point')),
                        'tangentes_communes' : (self.tangentes_communes, 1, ('courbe', 'courbe')),
                        'nouv_plan' : (self.nouv_plan, 0),
                        'suppr_plan' : (self.suppr_plan, 0),
                        'editeur_objets' : (self.edit_objets, 0),
                        'parametres' : (self.parametres, 0),
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
                        'connect' : (self.connect, 0),
                        }
        
    def act_ctrly(self):
        self.action('Redo', self.plans[0])
    def act_ctrlz(self):
        self.action('Undo', self.plans[0])
    
    def liste_objet(self):
        liste, plan = [], self.plans[0]
        liste.append((plan.bold, plan.boldP, plan.boldC, txt(plan.focal),
                            txt(plan.offset_x), txt(plan.offset_y), f'<{plan.nom}>'))
        liste.append([])
        for objet in plan.objets.values():
            liste[1].append((objet.classe, objet.nom, objet.method,
                                    objet.args,
                                    objet.deg, objet.color, objet.vis, objet.u))
        return liste
    
    def maj_menu(self):
        '''Met à jour le menu de selection des plans'''
        self.menu_deroulant.delete(0, 'end')
        self.menub.configure(text = f'{self.plans[0].nom}')
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
        f.write(self.plans[0].fichier())
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
        texte = fichier.read()
        fichier.close()
        self.nouv_plan()
        self.plans[0].ouvrir(texte)
        return 
        
        
    def edit_objets(self):
        print('??')
        if self.editeur_objets is None:
            self.editeur_objets = Fenetres.EditeurObjets(fenetre, self, 0)
            
    def bouger_point(self, ev):
        if self.point_move is None: return
        x, y = self.coord_canvas(ev.x, ev.y)
        if isinstance(self.point_move, tuple):
            mov1, mov2 = x - self.point_move[0], y -self.point_move[1]
            self.decaler((mov1/20, mov2/20))
        else:
            self.plans[0].action_utilisateur('bouger_point')
            l = self.action('Move', self.point_move.plan, self.point_move, (x, y, 1))
            for obj in l:
                obj.dessin(1)
            
            
    def nouv_plan(self):
        i = 1
        while f'Plan {i}' in [plan.nom for plan in self.plans]: i += 1
        self.plans.insert(0, Geo.Plan(self, nom = f'Plan {i}'))
        Geo.plan_default = self.plans[0]
        fenetre.title(f'JOmetry - {self.plans[0].nom}')
        self.maj_menu()
        self.dessin_canvas()
    
    def suppr_plan(self):
        result = "wesh"
        if self.plans[0].modifs[1]:
            result = tk_mb.askyesnocancel(f'fermeture de {self.plans[0].nom}', 'Voulez-vous enregistrer avant de fermer ?', icon = tk_mb.WARNING)
            if result == 'yes':
                self.enregistrer()
        if result is not None:
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
        x, y = int(x) + int(fx) + 10, int(y) + int(dy) + int(fy) + 34
        self.men = tk.Toplevel(borderwidth = 1, relief = 'solid',
                   background = 'white')
        self.men.overrideredirect(True)
        self.men.geometry(f'+{x}+{y}')
        for i, nom in enumerate(liste):
            image = image_tk(f'{op.dirname(__file__)}\images\{nom}.jpg')
            bout = ttk.Button(self.men, image = image)
            bout.config(command = lambda n = nom: self.echange(ind, n))
            self.image_boutons.append(image)
            bout.grid(row = i, column = 0)
            bout.bind('<Button-3>', lambda ev: bout.invoke)
    
    def serveur(self):
        if self.lanceurserveur is not None or self.plans[0].serveur is not None: return
        self.lanceurserveur = Fenetres.LanceServeur(self)
        
    def parametres(self):
        if self.parametres is not None: return
        self.parametres = Fenetres.Parametres(fenetre, self, Trad, style, params)        
    
    def connect(self):
        if self.plans[0].serveur is not None: return
        if self.connecteur_serv is not None: return
        self.connecteur_serv = Fenetres.ConnectServeur(self)
        
    def echange(self, ind, nom):
        self.detruire_menu()
        self.menu[ind].remove(nom)
        self.menu[ind].insert(0, nom)
        self.nom_boutons[ind] = nom
        self.boutons[ind].destroy()
        image = image_tk(f'{op.dirname(__file__)}\\images\{nom}.jpg')
        bout = ttk.Button(self.barre_haut, image = image)
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
        for bout, liste in ((self.boutons[-4], self.plans[0].ctrl_z), (self.boutons[-3], self.plans[0].ctrl_y)):
            if len(liste) == 0:
                bout['state'] = 'disabled'
            else:
                bout['state'] = 'normal'
    
    def action_bouton(self, nom, bout):
        self.dernier_bouton = nom
        for bouton in self.boutons:
            s = style.theme_use()
            bout.state(('!pressed',))
        if self.actions[nom][1]:
            self.attendus = self.actions[nom][2]
            s = style.theme_use()
            bout.state(('pressed',))
            self.action_canvas = self.actions[nom][0]
        else:
            self.plans[0].action_utilisateur(nom)
            self.attendus = None
            self.actions[nom][0]()
        self.deselectionner()
        self.liste_derniers_clics = []
        self.afficher_attendu()
        
    def tangentes_communes(self):
        c1, c2 = self.liste_derniers_clics
        tangentes_c1 = [self.action('Creature', self.plans[0], 'Droite', nom = 0, method = 'tangente', args = (c1, p), u = 0) for p in c1.args]
        tangentes_c2 = [self.action('Creature', self.plans[0], 'Droite', nom = 0, method = 'tangente', args = (c2, p), u = 0) for p in c2.args]
        c1_dual = self.action('Creature', self.plans[0], 'Courbe', nom = 0, method = 'interpol', deg = '', args = tangentes_c1, u = 0)
        c2_dual = self.action('Creature', self.plans[0], 'Courbe', nom = 0, method = 'interpol', deg = '', args = tangentes_c2, u = 0)
        c1_dual.coords()
        c2_dual.coords()
        for i in range(c1_dual.deg * c2_dual.deg):
            self.action('Creature', self.plans[0], 'Droite', nom = 1, method = 'inter2', args = (c1_dual, c2_dual, i), u = 1)
            
    def angle(self):
        A, B, C = self.liste_derniers_clics
        return self.action('Creature', self.plans[0], 'Angle', nom = 1, method = 'angle', args = [A, B, C, self.plans[0].U, self.plans[0].V], u = 1)
        
    def surcourbe(self):
        obj, pos = self.liste_derniers_clics
        if obj.classe == 'Droite':
            return self.action('Creature', self.plans[0], 'Point', nom = 1, method = 'ProjOrtho', args = [obj, (pos[0], pos[1], 1)], u = 1)
        return self.action('Creature', self.plans[0], 'Point', nom = 1, method = 'PsurCA', args = [obj, obj.deg, pos, self], u = 1)

    def harmonique(self):
        A,B,C = self.liste_derniers_clics
        return self.action('Creature', self.plans[0], 'Point', nom = 1, method = 'harmonique', args = (A, B, C), u = 1)

    def rotation(self):
        obj, p, angle = self.liste_derniers_clics
        return self.action('Creature', self.plans[0], obj.classe, nom = 1, method = 'rotation', args = (obj, p, -angle/180*pi), u = 1)

    def symetrie(self):
        obj, d = self.liste_derniers_clics
        return self.action('Creature', self.plans[0], obj.classe, nom = 1, method = 'symetrie', args = (obj, d), u = 1)

    def projective(self):
        objet, a,b,c,d,p,q,r,s = self.liste_derniers_clics
        return self.action('Creature', self.plans[0], obj.classe, nom = 1, method = 'projective', args = (obj, [a,b,c,d], [p,q,r,s]), u = 1)
    
    def homothetie(self):
        obj, p, rapport = self.liste_derniers_clics
        return self.action('Creature', self.plans[0], obj.classe, nom = 1, method = 'homothetie', args = (obj, p, rapport), u = 1)

    def translation(self):
        obj, p1, p2 = self.liste_derniers_clics
        return self.action('Creature', self.plans[0], obj.classe, nom = 1, method = 'translation', args = (obj, (p1.coords(), p2.coords())), u = 1)
        
    def polyregul(self):
        p1, p2, nombre = self.liste_derniers_clics
        nombre = int(nombre)
        for i in range(nombre - 2):
            p1, p2 = self.action('Creature', self.plans[0], 'Point', nom = 1, method = 'rotation', args = (p2, p1, (nombre-2)/nombre*pi), u = 1), p1

    def invers(self):
        obj, centre, rayon = self.liste_derniers_clics
        if obj.classe == 'Point':
            self.action('Creature', self.plans[0], obj.classe, 1, 'inversion', (obj, centre, rayon, (self.plans[0].U, self.plans[0].V)), obj.deg, u = 1)
        else:
            self.action('Creature', self.plans[0], obj.classe, 1, 'inversion', (obj, centre, rayon), obj.deg, u = 1)
            
    def inv_plan(self):
        centre, rayon = self.liste_derniers_clics
        plan = self.plans[0]
        l = plan.objets.values()
        self.nouv_plan()
        n_plan = self.plans[0]
        for obj in l:
            if obj.nom in ('U', 'V', 'Inf'): continue
            if obj.classe == 'Point':
                self.action('Creature', n_plan, obj.classe, obj.nom + "'", 'inversion', (obj, centre, rayon, (n_plan.U, n_plan.V)), obj.deg, u = 1)
            else:
                self.action('Creature', n_plan, obj.classe, obj.nom + "'", 'inversion', (obj, centre, rayon), obj.deg, u = 1)
            self.action('Creature', n_plan, 'Point', centre.nom + "'", 'rien', (centre, ), 1, u = 1)
            n_plan.offset_x = plan.offset_x
            n_plan.offset_y = plan.offset_y
            n_plan.bold, n_plan.boldP, n_plan.boldC = plan.bold, plan.boldP, plan.boldC
            n_plan.focal = plan.focal
            n_plan.plan_trans = plan
            
    def point(self):
        x, y = self.liste_derniers_clics[0]
        return self.action('Creature', self.plans[0], 'Point', nom = 1, method = 'coord', args = [(x, y, 1)], u = 1)


    def cercle(self):
        points = self.liste_derniers_clics + [self.plans[0].U, self.plans[0].V]
        return self.action('Creature', self.plans[0], 'Courbe', nom = 1, method = 'interpol', args = points, u = 1)
        
    def droite(self):
        A, B = self.liste_derniers_clics
        return self.action('Creature', self.plans[0], 'Droite', nom = 1, method = 'inter', args = (A, B), u = 1)
    
    def segment(self):
        A, B = self.liste_derniers_clics
        return self.action('Creature', self.plans[0], 'Droite', nom = 1, method = 'segment', args = (A, B), u = 1)
        
    def tangente(self):
        C, p = self.liste_derniers_clics
        return self.action('Creature', self.plans[0], 'Droite', nom = 1, method = 'tangente', args = (C, p), u = 1)
        
    def cercle_cent(self):
        centre, point =  self.liste_derniers_clics[0], self.liste_derniers_clics[1]
        U, V = self.plans[0].U, self.plans[0].V
        return self.action('Creature', self.plans[0], 'Courbe', nom = 1, method = 'cercle', args = (centre, point, U, V), u = 1)
        
    def perp(self):
        d, p = self.liste_derniers_clics
        return self.action('Creature', self.plans[0], 'Droite', nom = 1, method = 'perp', args = (d, p), u = 1)

    def caa(self):
        CA = [self.plans[0].newPoint_coord(1, (self.canvas.winfo_width()/4+random()*self.canvas.winfo_width()*3/4, self.canvas.winfo_height()/4+random()*3/4*self.canvas.winfo_height(),1), u=0) for _ in range([5, 9, 14, 20, 32][randint(0,4)])]
        if randint(0,3)==2:
            CA = [self.plans[0].U]+CA
        if randint(0,3)==2:
            CA = [self.plans[0].V]+CA
        self.plans[0].newCA(1, CA).dessin()
        
    def media(self):
        A, B = self.liste_derniers_clics
        return self.action('Creature', self.plans[0], 'Droite', nom = 1, method = 'media', args = (A, B), u = 1)

    def milieu(self):
        A, B = self.liste_derniers_clics
        return self.action('Creature', self.plans[0], 'Point', nom = 1, method = 'milieu', args = (A, B), u = 1)
     
    def centre(self):
        self.plans[0].newCentre(1, self.liste_derniers_clics)

    def para(self):
        d, A = self.liste_derniers_clics
        p = self.action('Creature', self.plans[0], 'Point', nom = 0, method = 'inf', args = (d,), u = 0)
        return self.action('Creature', self.plans[0], 'Droite', nom = 1, method = 'inter', args = (A, p), u = 1)

    def cercle_inscr(self):
        p1, p2, p3 = self.liste_derniers_clics
        plan = self.plans[0]
        centre = self.action('Creature', plan, 'Point', nom = 0, method = 'centreInscrit', args = (p1, p2, p3), u = 0)
        cote = self.action('Creature', plan, 'Droite', nom = 0, method = 'inter', args = (p1, p2), u = 0)
        point =  self.action('Creature', plan, 'Point', nom = 0, method = 'ProjOrtho', args = (cote, centre), u = 0)
        U, V = plan.U, plan.V
        return self.action('Creature', plan, 'Courbe', nom = 1, method = 'cercle', args = (centre, point, U, V), u = 1)
        
    def cercle_ex(self):
        p1, p2, p3 = self.liste_derniers_clics
        plan = self.plans[0]
        centre = self.action('Creature', plan, 'Point', nom = 0, method = 'centreInscrit', args = (p1, p2, p3), u = 0)
        bp1 = self.action('Creature', plan, 'Droite', nom = 0, method = 'inter', args = (centre, p1), u = 0)
        bp3 = self.action('Creature', plan, 'Droite', nom = 0, method = 'inter', args = (centre, p3), u = 0)
        be1 = self.action('Creature', plan, 'Droite', nom = 0, method = 'perp', args = (bp1, p1), u = 0)
        be3 = self.action('Creature', plan, 'Droite', nom = 0, method = 'perp', args = (bp3, p3), u = 0)
        centre2 = self.action('Creature', plan, 'Point', nom = 1, method = 'inter', args = (be1, be3), u = 1)
        cote = self.action('Creature', plan, 'Droite', nom = 0, method = 'inter', args = (p1, p3), u = 0)
        point = self.action('Creature', plan, 'Point', nom = 0, method = 'ProjOrtho', args = (cote, centre2), u = 0)
        U, V = plan.U, plan.V
        return self.action('Creature', plan, 'Courbe', nom = 1, method = 'cercle', args = (centre2, point, U, V), u = 1)
    
    def bissec(self):
        p1, p2, p3 = self.liste_derniers_clics
        centre = self.action('Creature', self.plans[0], 'Point', nom = 0, method = 'centreInscrit', args = (p1, p2, p3), u = 0)
        self.action('Creature', self.plans[0], 'Droite', nom = 1, method = 'inter', args = (centre, p2), u = 1)

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
            return self.action('Creature', self.plans[0], 'Point', nom = 1,
                               method = 'inter', args = [courbe_1, courbe_2], u = 1)
        for i in range(courbe_1.deg * courbe_2.deg):
            self.action('Creature', self.plans[0], 'Point', nom = 1, method = 'inter2', args = [courbe_1, courbe_2, i], u = 1)
        return None
            
    def supprimer(self):
        obj = self.liste_derniers_clics[0]
        self.action('Supprimer', obj.plan, obj)
        
    def soumettre(self):
        if self.action_canvas == self.courbe and len(self.liste_derniers_clics) >= 2:
            courbe = self.action('Creature', self.plans[0], 'Courbe', nom = 1, method = 'interpol', deg = '', args = [i for i in self.liste_derniers_clics], u = 1)
            self.deselectionner()
            courbe.dessin()
            self.action_bouton('courbe', self.boutons[7])
        
    courbe = soumettre
        
    def entree_commande(self, evenement):
        commande = self.entree_texte.get()
        self.entree_texte.set('')
        print(f"Vous avez essayé d'executer la commande suivante :\n{commande}\nMalheureusement, votre incompetence en informatique vous a empeché d'arriver à vos fins.")
        if commande[0] == '#':
            exec(commande[1:])
    
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
                if p_x.imag != 0 or p_y.imag != 0: continue
                distances.append((dist((x, y), (p_x, p_y)), i, p))
            distances.sort()
            if len(distances) == 0 or distances[0][0] > 20 * self.plans[0].offset_x[0]:
                #clic éloigné d'un point
                self.plans[0].action_utilisateur(None)
                if self.dernier_bouton !="main":
                    point = self.action('Creature', self.plans[0], 'Point', nom=1, method='coord', args=[(x, y, 1)], u=1)
                    if self.plans[0].serveur is not None:
                        point = 'fantome'
                    else:
                        self.canvas.itemconfigure(point.tkinter[0], fill = 'orange')
                else:
                    point = (x,y)
            else: 
                if distances[0][1] not in self.liste_derniers_clics:
                    point = distances[0][2]
                    self.canvas.itemconfigure(point.tkinter[0], fill = 'orange')
                    self.canvas.tag_raise(point.tkinter[0], self.canvas.find_all()[-1])
                    self.canvas.tag_raise(point.tkinter[1], self.canvas.find_all()[-1])
            self.liste_derniers_clics.append(point)
            if point != 'fantome' and self.liste_derniers_clics.count(point) not in {0,1}:
                self.canvas.itemconfigure(point.tkinter[1], text = point.nom + " : " + str(self.liste_derniers_clics.count(point)))
        if attendu in ['droite', 'courbe'] and self.plans[0].tkinter_object != {}:
            objet = self.canvas.find_closest(evenement.x, evenement.y,
                                             {'droite':self.limite2, 'courbe':self.limite1}[attendu])
            if len(objet) == 0: return
            courbe = self.plans[0].tkinter_object[objet[0]]
            if courbe not in self.liste_derniers_clics:
                self.liste_derniers_clics.append(courbe)
        if attendu == 'objet' and self.plans[0].tkinter_object != {}:
            objet = self.canvas.find_closest(evenement.x, evenement.y)
            if len(objet) == 0: return
            objet = self.plans[0].tkinter_object[objet[0]]
            if objet not in self.liste_derniers_clics and objet.classe != 'Angle':
                self.liste_derniers_clics.append(objet)
        nombres = True
        while len(self.liste_derniers_clics) < len(self.attendus) and isinstance(self.attendus[len(self.liste_derniers_clics)], tuple) and self.attendus[len(self.liste_derniers_clics)][0] == 'nombre' and nombres:
            entier = tk_sd.askfloat("Choix d'un nombre", self.attendus[len(self.liste_derniers_clics)][0][1])
            if entier is None:
                self.deselectionner()
                nombres = False
            else:
                self.liste_derniers_clics.append(entier)
        if len(self.liste_derniers_clics) == len(self.attendus):
            self.fin_clic_canvas()
        self.afficher_attendu()
        return

    def fin_clic_canvas(self):
        if 'fantome' in self.liste_derniers_clics:
            fenetre.after(50, self.fin_clic_canvas)
            print('on attend pour un fantome')
            return
        self.plans[0].action_utilisateur(None)
        self.deselectionner()
        self.action_canvas()
        self.liste_derniers_clics = []
        return
    
    def deselectionner(self):
        for objet in self.liste_derniers_clics:
            if isinstance(objet, Geo.Creature) and objet.classe != 'Angle':
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
                    if self.liste_derniers_clics.count(point)==0:
                        self.canvas.itemconfigure(point.tkinter[0], fill = point.color)
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
    
    def afficher_attendu(self):
        if self.attendus is None: return self.Texte.config(text = '')
        self.texte_att_aff = Trad(self.attendus[len(self.liste_derniers_clics)], weak = 1)
        self.Texte.config(textvariable = self.texte_att_aff)
        
    def action(self, cat, plan, *args, **kwargs):
        if plan.serveur is None:
            p = plan.action(cat, *args, **kwargs)
            self.maj_bouton()
            return p
        else:
            p = plan.envoi(cat, *args, **kwargs)
            self.maj_bouton()
            return p
    
    def fin(self):
        for i in range(len(self.plans)):
            self.suppr_plan()
        f = open(f'{op.dirname(__file__)}\\parametres.txt', encoding = 'utf-8', mode = 'w')
        f.write(txt(params))
        f.close()
        fenetre.destroy()
    
    
def ouvrir_erreur():
    tk_mb.showerror('Erreur', 'Impossible de lire ce fichier.')
        
chargement = False



if __name__ == '__main__':
    main = Main()
    fenetre.protocol('WM_DELETE_WINDOW', main.fin)
    fenetre.mainloop()
