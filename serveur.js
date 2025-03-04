import Engine as Geo
import socket
import threading


class Serveur:
    
    def __init__(self, mdp, plan = None):
        print(f'Serveur lancé, mot de passe : {mdp}')
        self.mdp = mdp
        self.clients = []
        self.autorises = []
        ip, port = socket.gethostbyname(socket.gethostname()), 61804
        if plan is not None:
            t = threading.Thread(target = plan.connecter_serveur, args = ('localhost', 61802, mdp))
            t.start()
            serveur = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            serveur.bind(('localhost', 61802))
            serveur.listen()
            print('Serveur ecoute')
            client = serveur.accept()
            self.clients.append(client[0])
            self.autorises.append(0)
            t = threading.Thread(target = self.ecoute_client, args = (client, len(self.autorises) - 1))
            t.start()
        self.plan = Geo.Plan()
        Geo.plan_default = self.plan
        print(f'Serveur lancé, adresse {ip}, port {port}, mot de passe : {mdp}')
        serveur.close()
        serveur = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serveur.bind((ip, port))
        serveur.listen(3)
        while True:
            client = serveur.accept()
            self.clients.append(client[0])
            self.autorises.append(0)
            t = threading.Thread(target = self.ecoute_client, args = (client, len(self.autorises) - 1))
            t.start()
    
    def ecoute_client(self, a, i):
        print(f"appareil à l'adresse connecté : {a} !! {i}")
        client, adresse = a
        autorisation, tests = 0, 0
        while True:
            requete = client.recv(2048)
            msg = requete.decode('utf-8')
            if len(msg) < 9 or msg[:8] != 'JOmetry ':
                continue
            msg = msg[8:]
            if msg == 'fin':
                client.send('JOmetry deconnecte'.encode('utf-8'))
                break
            if msg == f'0 {self.mdp}':
                client.send('JOmetry connecte'.encode('utf-8'))
                autorisation = 1
                self.autorises[i] = 1
                print(f"appareil à l'adresse {adresse} autorisé")
                continue
            if autorisation == 0:
                tests += 1
                if tests >= 5:
                    client.send('JOmetry 0 non autorisé'.encode('utf-8'))
                    break
                continue
            print(f'serveur traite le message suivant recu de client {adresse} : {msg}')
            for i, sock in enumerate(self.clients):
                if self.autorises[i]:
                    sock.send(requete)
            self.plan.decode_action(msg)
        client.close()
        
        
        
        
        
        
        
        
if __name__ == '__main__':
    mdp = input("Lancement d'un serveur JOmetry\nMot de passe de connexion : ")
    Serveur(mdp)
    
