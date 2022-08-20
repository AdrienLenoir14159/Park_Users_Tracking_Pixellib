import cv2
import numpy as np
import sys
from numpy.linalg import inv


class Object:
    LISTE_OBJECT = []
    COMPTEUR = 0

    ARBITRAIRE = 14400  # A factor sqrt(14400) = 120 for the radius of the predicted areas.
    ARBITRAIRE_DEUX = 12  # Number of frames , here 2*12 = 24, in which to find the support points

    def __init__(self, num_frame_appa, color, coor_actuelle):
        Object.COMPTEUR += 1
        self.id = Object.COMPTEUR
        Object.LISTE_OBJECT.append(self)
        self.num_frame_appa = num_frame_appa  # A l'initialisation
        # self.num_frame_dispa = 0  # Sera modifie en fin d'analyse video
        self.color = color  # (B,G,R)
        self.coor_suivi = {self.num_frame_appa: coor_actuelle}  # {indice1 : (Xcoor1, Ycoor1), ...}

    @property
    def mostRecentPosId(self):
        coor_suivi_numero_frames = self.coor_suivi.keys()
        return max(coor_suivi_numero_frames)

    @property
    def secondMostRecentPosId(self):
        coor_suivi_numero_frames = list(self.coor_suivi.keys())

        ### Catching the maximum :
        maxi, ind_maxi = coor_suivi_numero_frames[0], 0
        for indice in range(1, len(coor_suivi_numero_frames)):
            if coor_suivi_numero_frames[indice] > maxi:
                maxi = coor_suivi_numero_frames[indice]
                ind_maxi = indice
        ### Done

        coor_suivi_numero_frames.pop(ind_maxi)
        return max(coor_suivi_numero_frames)  # We return here the second greatest element of the list

    @property
    def mostRecentPos(self):
        return self.coor_suivi[self.mostRecentPosId]

    @property
    def secondMostRecentPos(self):
        return self.coor_suivi[self.secondMostRecentPosId]

    @property
    def interstice(self):
        return self.mostRecentPosId - self.secondMostRecentPosId
        # : Mesure du nombre de frames que l'objet saute (ou pas).

    @property # {(Xcenter, Ycenter), RadiusSquare} Predicts the zone of next appearance
    def predicZone(self):
        self.length_coor_suivi = len(self.coor_suivi)
        if self.length_coor_suivi == 1:  # Si premiere frame :
            return [self.coor_suivi[self.num_frame_appa], 2500]
            # Simple copie d'où on est; Rcarre = 2500 est arbitraire (soit R = 50)
            # Format : {(centerPosX, centerPosY), RadiusSquare}
        else:
            # Sinon, pour les frames suivantes à condition que ces deux dernières se suivent :
            inter = self.interstice

            actualPos = self.mostRecentPos
            oldPos = self.secondMostRecentPos

            expectedCenterPos = (actualPos[0] + (1 / inter) * (actualPos[0] - oldPos[0]),
                                 actualPos[1] + (1 / inter) * (actualPos[1] - oldPos[1]))
            return [(int(expectedCenterPos[0]), int(expectedCenterPos[1])),
                    Object.ARBITRAIRE * int(
                        ((actualPos[0] - oldPos[0]) / inter) ** 2 + ((actualPos[1] - oldPos[1]) / inter) ** 2)]
            # Format : {(centerPosX, centerPosY), RadiusSquare}

    @property # Gives the current direction of the Object
    def direction(self):
        if (self.mostRecentPos[0] - self.secondMostRecentPos[0]) != 0:
            direc = (self.mostRecentPos[1] - self.secondMostRecentPos[1]) / (
                        self.mostRecentPos[0] - self.secondMostRecentPos[0])
        else:
            direc = (self.mostRecentPos[1] - self.secondMostRecentPos[1])
        if direc == 0:
            direc = 0.01
        return direc

    # Prints the track on a blank image, in the color of the Object
    def showTraj(self):

        # MAX_SIZE_WINDOW = 0

        image_depart = 255 * np.ones((2160, 4096, 3))
        # image_depart = np.zeros((2160, 4096, 3))
        fichier_trajectoire = "Itineraire_objet_" + str(self.id) + ".png"
        # cv2.imwrite(fichier_trajectoire, image_depart)
        # image_depart = cv2.imread(fichier_trajectoire, 1)
        # Manipulation étrange pour l'avoir à la bonne taille

        for indice in self.coor_suivi.keys():
            image_depart = cv2.circle(
                image_depart,
                self.coor_suivi[indice],
                15,
                self.color,
                thickness=5
                # lineType=8,
                # shift=0
            )

        # Vérification visuelle :
        # cv2.namedWindow("Test_image_blanche_avec_points", MAX_SIZE_WINDOW)     De même que précédemment :
        # cv2.imshow("Test_image_blanche_avec_points", image_depart)

        cv2.imwrite(fichier_trajectoire, image_depart)  # Si le test est concluant

    # Checks if there are any holes left along the track
    def verif(self):
        N = max(self.coor_suivi.keys())
        if 2 * sum(self.coor_suivi.keys()) == N * (N + 1):  # Somme des entiers jusqu'à N = N(N+1)/2
            return True  # Il ne manque rien
        else:
            return False  # Il manque des points -> Necessite une interpolation

    # Returns a list of the indices representing the frames on which the Object isn't recognized
    def holes(self):
        liste = self.coor_suivi.keys()
        fin = max(liste)
        res = []
        i = 0
        j = 0
        while i <= fin and j < len(liste):
            if i == liste[j]:
                i += 1
                j += 1
            if i < liste[j]:
                for k in range(i, liste[j]):
                    res.append(k)
                j += 1
                if j < len(liste):
                    i = liste[j]
        return res

    # Returns an array containing the polynomial coefficients found
    def interpol(cls, listeX, listeY, ordre):  # En pratique, les "listeX" et "listeY" sont des extraits de coor_suivi
        if len(listeX) != len(listeY):
            print("Erreur : Les listes des Xs et des Ys ne sont pas de la même longueur")
            sys.exit(10)
        if ordre == 0 or ordre >= len(listeX):
            print("Probleme d'ordre choisi")
            sys.exit(11)
        else:

            listeX, listeY = cls.tri_rec(listeX,
                                     listeY)  # Tri des x dans l'ordre croissant, avec accord des y en conséquence

            if ordre == 1:  # Donc matrices dim 2
                A = np.array([[listeX[0], 1],
                              [listeX[-1], 1]])
                B = np.array([[listeY[0]],
                              [listeY[-1]]])
            elif ordre == 2:  # Donc matrices dim 3
                A = np.array([[listeX[0] ** 2, listeX[0], 1],
                              [listeX[int(len(listeX) / 2)] ** 2, listeX[int(len(listeX) / 2)], 1],
                              [listeX[-1] ** 2, listeX[-1], 1]])
                B = np.array([[listeY[0]],
                              [listeY[int(len(listeY) / 2)]],
                              [listeY[-1]]])
            elif ordre == 3:  # Et ainsi de suite ...
                A = np.array([[listeX[0] ** 3, listeX[0] ** 2, listeX[0], 1],
                              [listeX[int(len(listeX) / 3)] ** 3, listeX[int(len(listeX) / 3)] ** 2,
                               listeX[int(len(listeX) / 3)], 1],
                              [listeX[int(2 * len(listeX) / 3)] ** 3, listeX[int(2 * len(listeX) / 3)] ** 2,
                               listeX[int(2 * len(listeX) / 3)], 1],
                              [listeX[-1] ** 3, listeX[-1] ** 2, listeX[-1], 1]])
                B = np.array([[listeY[0]],
                              [listeY[int(len(listeY) / 3)]],
                              [listeY[int(2 * len(listeY) / 3)]],
                              [listeY[-1]]])
            elif ordre == 4:  # Et ainsi de suite ...
                A = np.array([[listeX[0] ** 4, listeX[0] ** 3, listeX[0] ** 2, listeX[0], 1],
                              [listeX[int(len(listeX) / 4)] ** 4, listeX[int(len(listeX) / 4)] ** 3,
                               listeX[int(len(listeX) / 4)] ** 2, listeX[int(len(listeX) / 4)], 1],
                              [listeX[int(len(listeX) / 2)] ** 4, listeX[int(len(listeX) / 2)] ** 3,
                               listeX[int(len(listeX) / 2)] ** 2, listeX[int(len(listeX) / 2)], 1],
                              [listeX[int(3 * len(listeX) / 4)] ** 4, listeX[int(3 * len(listeX) / 4)] ** 3,
                               listeX[int(3 * len(listeX) / 4)] ** 2, listeX[int(3 * len(listeX) / 4)], 1],
                              [listeX[-1] ** 4, listeX[-1] ** 3, listeX[-1] ** 2, listeX[-1], 1]])
                B = np.array([[listeY[0]],
                              [listeY[int(len(listeY) / 4)]],
                              [listeY[int(len(listeY) / 2)]],
                              [listeY[int(3 * len(listeY) / 4)]],
                              [listeY[-1]]])
            else:  # Ordre 5 et plus, mais l'on s'arrette ici à cinq
                A = np.array([[listeX[0] ** 5, listeX[0] ** 4, listeX[0] ** 3, listeX[0] ** 2, listeX[0], 1],
                              [listeX[int(len(listeX) / 5)] ** 5, listeX[int(len(listeX) / 5)] ** 4,
                               listeX[int(len(listeX) / 5)] ** 3, listeX[int(len(listeX) / 5)] ** 2,
                               listeX[int(len(listeX) / 5)], 1],
                              [listeX[int(2 * len(listeX) / 5)] ** 5, listeX[int(2 * len(listeX) / 5)] ** 4,
                               listeX[int(2 * len(listeX) / 5)] ** 3, listeX[int(2 * len(listeX) / 5)] ** 2,
                               listeX[int(2 * len(listeX) / 5)], 1],
                              [listeX[int(3 * len(listeX) / 5)] ** 5, listeX[int(3 * len(listeX) / 5)] ** 4,
                               listeX[int(3 * len(listeX) / 5)] ** 3, listeX[int(3 * len(listeX) / 5)] ** 2,
                               listeX[int(3 * len(listeX) / 5)], 1],
                              [listeX[int(4 * len(listeX) / 5)] ** 5, listeX[int(4 * len(listeX) / 5)] ** 4,
                               listeX[int(4 * len(listeX) / 5)] ** 3, listeX[int(4 * len(listeX) / 5)] ** 2,
                               listeX[int(4 * len(listeX) / 5)], 1],
                              [listeX[-1] ** 5, listeX[-1] ** 4, listeX[-1] ** 3, listeX[-1] ** 2, listeX[-1], 1]])
                B = np.array([[listeY[0]],
                              [listeY[int(len(listeY) / 5)]],
                              [listeY[int(2 * len(listeY) / 5)]],
                              [listeY[int(3 * len(listeY) / 5)]],
                              [listeY[int(4 * len(listeY) / 5)]],
                              [listeY[-1]]])

            #############
            ## On a alors A.[Matrice colonne contenant les coefficients polynomiaux recherches] = B
            ## Ou A à la forme suivante : (Pour dim 3, soit ordre 2)
            # [[x0**2, x0, 1],
            #  [x1**2, x1, 1],
            #  [x2**2, x2, 1]]
            # Et B la forme suivante :
            # [[y0],
            #  [y1],
            #  [y2]]
            # Donc on return:
            return inv(A) @ B

    # Color accessor
    def setColor(self, color):
        self.color = color

    # Coordinates dictionary accessor
    def addCoorSuivi(self, frame_id, coorXY):
        self.coor_suivi[frame_id] = coorXY  # coorXY : tuple (x, y)

    @classmethod # To pass the values to the "interpol" function
    def tri_rec(cls, lx, ly):  # Trie la liste x, et ordonne la liste y en accord. Fonctionne avec la fonction "recombine"
        if len(lx) == 1:
            return (lx, ly)
        else:
            return cls.recombine(cls.tri_rec(lx[0:int(len(lx) / 2)], ly[0:int(len(ly) / 2)]),
                             cls.tri_rec(lx[int(len(lx) / 2):len(lx)], ly[int(len(ly) / 2):len(ly)]))

    @classmethod # Works with the "tri_rec" method
    def recombine(cls, t1, t2):
        lx1 = t1[0]
        lx2 = t2[0]
        ly1 = t1[1]
        ly2 = t2[1]

        resx = []
        resy = []
        while len(lx1) > 0 and len(lx2) > 0:
            if lx1[0] > lx2[0]:
                resx.append(lx2.pop(0))
                resy.append(ly2.pop(0))
            else:
                resx.append(lx1.pop(0))
                resy.append(ly1.pop(0))
        if len(lx1) == 0:
            for i in range(len(lx2)):
                resx.append(lx2[i])
                resy.append(ly2[i])
        else:
            for i in range(len(lx1)):
                resx.append(lx1[i])
                resy.append(ly1[i])
        return (resx, resy)
