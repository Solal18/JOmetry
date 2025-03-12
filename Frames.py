import tkinter as tk
from tkinter.ttk import Treeview, Combobox
from tkinter import colorchooser as tk_cc
from PIL import Image, ImageTk
import os.path as op
from threading import Thread
from serveur import Serveur

class Scrollable_Frame(tk.Frame):
    
    def __init__(self, parent, orientation = 'vertical', **kwargs):
        super().__init__(parent)
        self.grid(**kwargs)
        self.canvas = tk.Canvas(self, bd = 0)
        self.frame = tk.Frame(self.canvas)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.scrollbar = tk.Scrollbar(self, command = self.canvas.yview, orient = orientation)
        
        if orientation == 'vertical':
            self.canvas.configure(yscrollcommand = self.scrollbar.set)
            self.scrollbar.grid(row = 0, column = 1, sticky = 'nsew')
        else:
            self.canvas.configure(xscrollcommand = self.scrollbar.set)
            self.scrollbar.grid(row = 1, column = 0, sticky = 'nsew')
        self.canvas.create_window(0, 0, anchor = 'nw', window = self.frame, tags = ('frame',))
        self.canvas.bind('<Configure>', self.configure_frame)
        self.frame.bind('<Configure>', self.maj_canvas)
        
        self.canvas.grid(row = 0, column = 0, sticky = 'nsew')

    def maj_canvas(self, ev):
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

        self.entree = tk.Entry(self.fen, width=30)
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

    
    def recherche(self, texte):
        resultats = []
        for i in self.doc:
            if any([(texte in j) for j in i[0]]) and len(resultats) < 10:
                resultats.append(i[1])
        return resultats

    def texte_aide(self, texte):
        texte, liens = self.doc[[i[1] for i in self.doc].index(texte)][2], {}
        while '[' in texte:
            idx1, idx2, idx3 = texte.index('['), texte.index('|'), texte.index(']')
            liens[(idx1, idx2 - 1)] = texte[idx2 + 1:idx3]
            texte = texte[:idx1]+texte[idx1+1:idx2]+texte[idx3+1:]
        return texte, liens

    def mettre_a_jour_resultats(self, event):
        texte_recherche = self.entree.get()
        self.resultats = self.recherche(texte_recherche)
        self.possibles.set(' '.join([x.replace(' ', '\ ') for x in self.resultats]))

    def clic_liste(self, event = None, texte = ''):
        if self.resultats is None: return
        if texte == '': texte = self.resultats[self.liste.curselection()[0]]
        nouveau_texte, liens = self.texte_aide(texte)
        self.texte.config(state = 'normal')
        self.texte.delete('1.0', 'end')
        self.texte.insert('end', nouveau_texte)
        self.texte.config(state = 'disabled')
        for (id1, id2), mot in liens.items():
            i1, i2 = f'1.0+{id1}c', f'1.0+{id2}c'
            self.texte.tag_add(mot, i1, i2)
            self.texte.tag_config(mot, foreground = 'blue', underline = True)
            self.texte.tag_bind(mot, '<Button-1>', lambda ev, mot = mot: self.clic_liste(texte = mot))
    
 

class EditeurObjets:
    
    def __init__(self, fenetre, main, separe):
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
            tk.Button(self.grande_frame, text = 'fermer', command = self.supprimer, bg = '#ddd').grid(row = 1, column = 0)
            fenetre.bind('<Return>', self.clic_entree)
        self.imgs = (tk.BitmapImage(data = \
          '#define colorpicker_width 10\n\#define colorpicker_height 10\n\
           static unsigned char colorpicker_bits[] = {\
             0xc0, 0x01, 0xe0, 0x03, 0xe0, 0x03, 0xd8, 0x03, 0xb0, 0x01,\
             0x78, 0x00, 0x5c, 0x00, 0x0e, 0x00, 0x07, 0x00, 0x03, 0x00 };'),
                     ImageTk.PhotoImage(file = f'{op.dirname(__file__)}\images\poubelle2.bmp'))
        self.main = main
        self.tableau = Treeview(self.frame, columns = ('nom', 'type', 'fonction', 'args', 'couleur', 'vis'), selectmode = 'browse')
        self.tableau.grid(row = 0, column = 0, columnspan = 4)
        self.tableau.column('#0', width = 0, stretch = False)
        for i, t in (('nom', 'objet'), ('type', 'type'), ('fonction', 'définition'), ('args', 'dépend de'), ('couleur', 'couleur'), ('vis', 'affichage')):
            self.tableau.column(i, width=40)
            self.tableau.heading(i, text = t)
        self.nom_methodes = {'coord' : 'coordonées', 'inter' : 'intersection', 'inter2' : 'intersection', 'ortho' : 'ortho', 'inf' : 'inf', 'milieu' : 'milieu', 'centreInscrit' : 'centre inscrit',
                        'perp' : 'perpendiculaire', 'media' : 'médiatrice', 'biss' : 'bissectrice', 'rotation' : 'rotation', 'transformation' : 'transformation', 'homothetie' : 'homothetie', 'tangente' : 'tangente',
                        'cercle' : 'conique tangente à deux droites', 'segment':'segment', 'interpol' : 'interpolation', 'harmonique' : 'harmonique', 'PsurCA' : 'point sur courbe', 'invers' : 'inversion'}
        self.var1, self.var2, self.var3 = tk.StringVar(), tk.StringVar(), tk.IntVar()
        self.entree = tk.Entry(self.frame, width = 8, state = 'disabled', textvariable = self.var1)
        self.couleur = tk.Frame(self.frame)
        self.couleur_choix = tk.Entry(self.couleur, width = 8, state = 'disabled', textvariable = self.var2)
        self.bouton_col = tk.Button(self.couleur, image = self.imgs[0], state = 'disabled', command = self.tk_couleur)
        self.aff = tk.Checkbutton(self.frame, bg = '#ddd', state = 'disabled', variable = self.var3, text = 'affichage')
        self.suppr = tk.Button(self.frame, bg = '#ddd', state = 'disabled', image = self.imgs[1], command = None)
        self.label = tk.Label(self.frame, text = 'Selectionnez un objet\npour modifier ses proprietes', bg = '#ddd')
        self.entree.grid(row = 1, column = 0)
        self.couleur.grid(row = 1, column = 1, padx = 4)
        self.couleur_choix.grid(row = 0, column = 0)
        self.bouton_col.grid(row = 0, column = 1)
        self.aff.grid(row = 1, column = 2)
        self.suppr.grid(row = 1, column = 3)
        self.label.grid(row = 2, column = 0, columnspan = 4)
        self.tableau.bind("<<TreeviewSelect>>", self.clic_ligne)
        self.frame.bind('<Return>', self.clic_entree)
        self.maj()

    def supprimer_element(self, i, b = 0):
        nom = i.nom
        for item in self.tableau.get_children():
            ligne = self.tableau.item(item)['values']
            if ligne and ligne[0] == nom:
                self.tableau.delete(item)
        if self.selectionne == i:
            self.deselectionner()
        if b: self.main.action('Supprimer', i.plan, i)

    def deselectionner(self):
        self.var1.set('')
        self.var2.set('')
        self.var3.set(0)
        for widget in (self.entree, self.couleur_choix, self.bouton_col,
                       self.aff, self.suppr):
            widget['state'] = 'disabled'
        self.selectionne = None

    def supprimer(self):
        self.grande_frame.grid_forget()
        self.grande_frame.forget()
        self.main.editeur_objets = None

    def fermer_fenetre(self):
        self.main.editeur_objets = None
        self.grande_frame.destroy()

    def ajouter(self, obj):
        if obj.u:
            self.tableau.insert('', 'end', values = [obj.nom, obj.classe, self.nom_methodes[obj.method], list(map(str, obj.args)), obj.color, ('non', 'oui')[obj.vis]])

    def maj(self):
        for item in self.tableau.get_children():
            self.tableau.delete(item)
        self.ides = {}
        self.objets = []
        for objet in self.main.plans[0].objets.values():
            if objet.u:
                self.ides[objet.nom] = objet.ide
                self.objets.append((objet.nom, objet.classe, self.nom_methodes[objet.method],
                                    objet.args, objet.color, ('non', 'oui')[objet.vis]))
        for l in self.objets:
            self.tableau.insert('', 'end', values = l)
        self.deselectionner()

    def clic_entree(self, event):
        self.main.plans[0].action_utilisateur(f'proprietes{self.selectionne}')
        if self.selectionne is None: return
        nom, couleur, aff = self.var1.get(), self.var2.get(), self.var3.get()
        if nom in ('U', 'V'):
            self.label['text'] = 'Nom déjà utilisé\n'
            return
        try: self.fenetre.winfo_rgb(couleur)
        except:
            self.label['text'] = 'Couleur invalide\n'
            return
        self.label['text'] = '\n'
        anc_nom = self.selectionne.nom
        self.main.action('Modif', self.selectionne.plan, self.selectionne, nom = nom, col = couleur, vis = aff)
        for item in self.tableau.get_children():
            ligne = self.tableau.item(item)['values']
            if ligne and ligne[0] == anc_nom:
                ligne[0] = nom
                ligne[4] = couleur
                ligne[5] = ['non', 'oui'][aff]
                self.tableau.item(item, values = ligne)

    def clic_ligne(self, event):
        if self.tableau.selection() == tuple(): return
        ligne = self.tableau.item(self.tableau.selection()[0])['values']
        self.ligne_select = self.tableau.selection()[0]
        ide = self.ides[ligne[0]]
        self.selectionne = self.main.plans[0].objets[ide]
        for widget in (self.entree, self.couleur_choix, self.bouton_col,
                       self.aff, self.suppr):
            widget['state'] = 'normal'
        self.suppr['command'] = lambda: self.supprimer_element(self.selectionne, b = 1)
        self.var1.set(ligne[0])
        self.var2.set(ligne[4])
        self.var3.set(['non', 'oui'].index(ligne[5]))
        
    def tk_couleur(self):
        couleur = tk_cc.askcolor()
        if couleur[1] is not None:
            self.var2.set(couleur[1])


class EtudieurObjets:
    
    def __init__(self, fenetre, main, separe):
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
            tk.Button(self.grande_frame, text = 'fermer', command = self.supprimer, bg = '#ddd').grid(row = 1, column = 0)
            fenetre.bind('<Return>', self.clic_entree)
        self.main = main
        self.valeur = tk.StringVar()
        self.listvariable = tk.StringVar()
        self.texte = tk.Label(self.frame, text = 'Entrez une valeur pour essayer de determiner une constante, en autorisant les points suivants à varier')
        self.entree = tk.Entry(self.frame, textvariable = self.valeur)
        self.formule = tk.Label(self.frame, text = '')
        self.bouton = tk.Button(self.frame, text = 'Etude', command = self.etude)
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
        from numpy import float64
        from random import randrange
        import Engine as Geo
        formule = self.valeur.get()
        if formule in self.main.plans[0].points:
            point = self.main.plans[0].points[formule]
            bouge = [self.main.plans[0].points[self.liste.get(0, 'end')[nombre]] for nombre in self.liste.curselection()]
            positions = [point.args for p in bouge]
            l = []
            print(point, bouge, positions)
            for i in range(20):
                for p in bouge:
                    x, y = randrange(0, 500), randrange(0, 500)
                    p.plan.move(p, (x, y, 1))
                z = tuple([float(t) if isinstance(t, float64) else t for t in point.coords()])
                print(z)
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
            self.main.plans[0].newCA(1, l[:deg])
            for p, pos in zip(bouge, positions):
                if isinstance(pos[0], Geo.Creature):
                    pos = pos[1]
                p.plan.move(p, pos)
            
        
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
        tk.Label(self.frame, text = 'Lancer un serveur').grid(row = 0, column = 0, columnspan = 3, sticky = 'nsew', pady = 3, padx = 3)
        tk.Label(self.frame, text = 'Mot de passe').grid(row = 1, column = 0, columnspan = 1, sticky = 'nsew', pady = 3, padx = 3)
        self.entree = tk.Entry(self.frame)
        self.entree.grid(row = 1, column = 1, columnspan = 2, sticky = 'nsew', pady = 3, padx = 3)
        tk.Button(self.frame, text = 'Annuler', command = self.fermer_fenetre).grid(row = 2, column = 0, columnspan = 2, sticky = 'nsew', pady = 3, padx = 3)
        tk.Button(self.frame, text = 'OK', command = self.lancer).grid(row = 2, column = 2, columnspan = 1, sticky = 'nsew', pady = 3, padx = 3)
        
        
    def fermer_fenetre(self):
        self.main.lanceurserveur = None
        self.frame.destroy()
    
    def lancer(self):
        mdp = self.entree.get()
        t = Thread(target = Serveur, args = (mdp, self.main.plans[0], self.main.plans[0].fichier()))
        t.start()
        self.fermer_fenetre()
        
        
class Parametres:
    
    def __init__(self, fenetre, main, style):
        self.fenetre = fenetre
        self.style = style
        self.main = main
        self.toplevel = tk.Toplevel()
        self.toplevel.columnconfigure(0, weight = 1)
        self.toplevel.rowconfigure(0, weight = 1)
        self.frame = Scrollable_Frame(self.toplevel, row = 0, column = 0, sticky = 'nsew').frame
        for colonne in range(2):
            self.frame.columnconfigure(colonne, weight = 1)
        self.toplevel.protocol('WM_DELETE_WINDOW', self.fermer_fenetre)
        plan = main.plans[0]
        self.p = [('nombre', 'Taille des points', plan.boldP, 3),
                  ('nombre', 'Epaisseur des lignes', plan.boldC, 3),
                  ('choix', "Style de l'interface", style.theme_use(), 'clam', style.theme_names()),
                  ('texte', 'Nom du plan', plan.nom, 'Plan 1'),
                  ('nombre', 'Taille des points', plan.boldP, 3),
                  ('nombre', 'Epaisseur des lignes', plan.boldC, 3),
                  ('choix', "Style de l'interface", style.theme_use(), 'clam', style.theme_names()),
                  ('texte', 'Nom du plan', plan.nom, 'Plan 1'),
                  ('nombre', 'Taille des points', plan.boldP, 3),
                  ('nombre', 'Epaisseur des lignes', plan.boldC, 3),
                  ('choix', "Style de l'interface", style.theme_use(), 'clam', style.theme_names()),
                  ('texte', 'Nom du plan', plan.nom, 'Plan 1')]
        self.widgets = []
        self.valeurs = []
        tk.Label(self.frame, text = 'Paramètres').grid(row = 0, column = 0, columnspan = 4)
        for i, e in enumerate(self.p):
            if e[0] == 'nombre':
                v = tk.IntVar()
                w = tk.Spinbox(self.frame, textvariable = v, from_ = 0)
            if e[0] == 'choix':
                v = tk.StringVar()
                w = Combobox(self.frame, state = 'readonly', textvariable = v, values = e[4])
            if e[0] == 'texte':
                v = tk.StringVar()
                w = tk.Entry(self.frame, textvariable = v)
            w.grid(row = i+1, column = 0, sticky = 'nsew')
            self.valeurs.append(v)
            self.widgets.append(w)
            tk.Label(self.frame, text = e[1]).grid(row = i+1, column = 1, sticky = 'nsew')
        f = tk.Frame(self.frame, padx = 3, pady = 3)
        for colonne in range(3):
            f.columnconfigure(colonne, weight = 1)
        f.grid(row = i+2, column = 0, columnspan = 2)
        tk.Button(f, text = 'Reinitialiser', command = lambda: self.assigner_valeurs([e[3] for e in self.p])).grid(row = 0, column = 0, sticky = 'nsew')
        tk.Button(f, text = '   Annuler   ', command = self.fermer_fenetre).grid(row = 0, column = 1, sticky = 'nsew')
        tk.Button(f, text = '     OK     ', command = self.changer_param).grid(row = 0, column = 2, sticky = 'nsew')
        self.assigner_valeurs([e[2] for e in self.p])
            
    def fermer_fenetre(self):
        self.main.parametres = None
        self.toplevel.destroy()
    
    def maj(self):
        pass
    
    def assigner_valeurs(self, liste):
        for v, w in zip(self.p, self.valeurs):
            w.set(v[2])
    
    def changer_param(self):
        l = [w.get() for w in self.valeurs]
        plan = self.main.plans[0]
        plan.boldP, plan.boldC, plan.nom = l[0], l[1], l[3]
        self.style.theme_use(l[2])
        self.fermer_fenetre()
        self.main.menub.configure(text = f'{l[3]}  \u25bc')
                
class Notes:
    
    def __init__(self, fenetre, main, separe):
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
            tk.Button(self.grande_frame, text = 'fermer', command = self.supprimer, bg = '#ddd').grid(row = 1, column = 0)
            fenetre.bind('<Return>', self.clic_entree)
        
