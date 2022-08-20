from pixellib.instance import instance_segmentation
import random as rd
import cv2
import numpy as np


import objects

# Durées en milisecondes
UN = 1000
TROIS = 3000
CINQ = 5000

# Constantes a regler
PERIM = 31 # nombre de pixels tel qu'un Object plus proche que cela peut-etre considere comme la suite d'une trace.
PERIM_MULT = 1.15
FORWARD_UNIT = 11 # Environs le nombre de pixels avances en une frame

### Segmentation avec extraction et sauvegarde de chaque objet
seg_img = instance_segmentation(infer_speed="average")
seg_img.load_model("mask_rcnn_coco.h5")

capture = cv2.VideoCapture("DJI_0021.MOV")

frame_count = 0

while True:
    ok, frame_UN = capture.read()  # capture.get(cv2.CAP_PROP_POS_MSEC)
    if ok != True:
        print("\nOn sort !\n")
        capture.release()
        cv2.destroyAllWindows()
        break

    frame_count += 1

    cv2.imwrite("image_travail.jpg", frame_UN)

    target_type = seg_img.select_target_classes(person=True)
    masks, rendu = seg_img.segmentImage("image_travail.jpg",
                                        show_bboxes=True,
                                        segment_target_classes=target_type,
                                        output_image_name="DJI_0021_RECOG.JPG"
                                        # extract_segmented_objects = True,
                                        # save_extracted_objects = True
                                        )

    box_coor = masks['rois']  ############## LE DEBUT DES MORCEAUX IMPORTANTS

    frame_DEUX = cv2.imread("DJI_0021_RECOG.JPG")
    cv2.namedWindow("apercu_travail", cv2.WINDOW_NORMAL)
    cv2.imshow("apercu_travail", frame_DEUX)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

    ### DBT attributions
    dico_keeper = {}  # 0 for non-attributed, 1 for attributed
    # To keep track of the attributions, because no two objects can relate to the same detected bbox
    for indice_bbox in range(box_coor.shape[0]):  ### POUR CHAQUE BBOX DE CHAQUE FRAME
        dico_keeper[indice_bbox] = 0

        x = int((box_coor[indice_bbox, 1] + box_coor[indice_bbox, 3]) / 2)
        y = int((box_coor[indice_bbox, 0] + box_coor[indice_bbox, 2]) / 2)
        if len(objects.Object.LISTE_OBJECT) == 0:
            objects.Object(frame_count,
                           (rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255)),
                           (x, y))

        # for obj in objects.Object.LISTE_OBJECT: ### POUR CHAQUE Objet A COMPARER AVEC CHAQUE BBOX, A CHAQUE FRAME
        # # Les normalizers peuvent être différents des objects.Object.interstice

        if dico_keeper[indice_bbox] == 1:  # Cela ne peut plus arriver dans cette version du code...
            pass
        # elif pow(((x - obj.predicZone[0][0]) / normalizer), 2) + pow(((y - obj.predicZone[0][1]) / normalizer), 2) <= \
        # obj.predicZone[1]:
        else:
            #####
            last_poses = [obj.mostRecentPos for obj in objects.Object.LISTE_OBJECT]
            normalizers = [frame_count - obj.mostRecentPosId for obj in objects.Object.LISTE_OBJECT]

            ind_nearest = 0
            dist_nearest = int(pow(PERIM * PERIM_MULT, 2))  ##### POUR LIGNE VERIF CHANGEMENT
            i = 0
            for (x2, y2) in last_poses:
                if normalizers[i] == 0:
                    normalizers[i] = 1
                if int((pow(x - x2, 2) + pow(y - y2, 2)) / pow(normalizers[i], 2)) <= pow(PERIM, 2):
                    dist = int((pow(x - x2, 2) + pow(y - y2, 2)) / pow(normalizers[i], 2))
                    # print(dist, "out of ", dist_nearest) #########
                    if dist < dist_nearest:
                        dist_nearest = dist
                        ind_nearest = i
                    i += 1
                else:
                    pass

            # x2, y2 = objects.Object.LISTE_OBJECT[ind_nearest].mostRecentPos
            if dist_nearest <= pow(PERIM, 2):  ##### LIGNE VERIF CHANGEMENT
                # if len(objects.Object.LISTE_OBJECT[ind_nearest].coor_suivi) >= 2:
                # if (x2-x != 0) and (0.5 <= abs(((y2-y)/(x2-x))/objects.Object.LISTE_OBJECT[ind_nearest].direction) <= 2):
                # objects.Object.LISTE_OBJECT[ind_nearest].addCoorSuivi(frame_count, (x, y))
                # elif (x2-x == 0) and (0.5 <= abs((y2-y)/objects.Object.LISTE_OBJECT[ind_nearest].direction) <= 2):
                objects.Object.LISTE_OBJECT[ind_nearest].addCoorSuivi(frame_count, (x, y))  #####
                # else:
                # objects.Object(frame_count,
                # (rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255)),
                # (x, y))
                # else:
                # objects.Object.LISTE_OBJECT[ind_nearest].addCoorSuivi(frame_count, (x, y))
            else:
                objects.Object(frame_count,
                               (rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255)),
                               (x, y))
            #####
            # obj.addCoorSuivi(frame_count, (x, y))
            dico_keeper[indice_bbox] = 1
        # Pour éviter tout paradoxe
        # Car une fois la bbox détectée sur une frame attribuée à un objet,
        # nul autre objet ne peut simultanément s'y identifier

        if dico_keeper[indice_bbox] == 0:  # Bbox non attribuée car trop loin de tt les object.
            # Ne survient plus non plus dans cette version du code
            objects.Object(frame_count,
                           (rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255)),
                           (x, y))
            dico_keeper[indice_bbox] = 1
    ### FIN attributions

    # if cv2.waitKey(10) & 0xFF == ord('q'):  # Si la touche 'q' est pressée, on quitte
    #     cap.release()
    #     cv2.destroyAllWindows()
    # sys.exit(2) ################### Pas sur, on peut vouloir fermer les fenetres sans perdre les données calculées

    print("Numero de la frame traitée :", frame_count)  ##### JUST FOR NOW
    if objects.Object.LISTE_OBJECT != None:
        print("Nombres d'objets alors détectés :", len(objects.Object.LISTE_OBJECT), "\n")  #####

capture.release()
cv2.destroyAllWindows()

# It is to note that we're only expecting about 50 Objects tops to be discovered
# (Only about 50 MOVING in the video, 70 otherwise)
# So 200+ is definitely too much, and a dozen, too little

###########################

# On va attaquer autour des trous : Si trou à un indice i, on regarde des deux côtés jusqu'à i-k, i+k
# pour trouver nos points d'appuis.
a_rajouter = []
points_dappui = {}
coeffs_poly = np.empty
for obj in objects.Object.LISTE_OBJECT:
    if not obj.verif():
        a_rajouter = obj.holes()
        if len(a_rajouter) >= 3:  # (Or si moins de 3 trous, pas grave dans tout les cas)
            for i in range(len(a_rajouter) - 1):
                if a_rajouter[i + 1] - a_rajouter[i] > 1:  # Si on detecte une bordure
                    points_dappui[a_rajouter[i] + 1] = obj.coor_suivi[a_rajouter[i] + 1]
        # On va ensuite selectionner de 2 à 5 points d'appuis au voisinnage de chaque trou.
        liste_x = []
        liste_y = []
        for i in range(objects.Object.ARBITRAIRE_DEUX,
                       len(a_rajouter) - objects.Object.ARBITRAIRE_DEUX - 1,
                       2 * objects.Object.ARBITRAIRE_DEUX):
            for j in range(objects.Object.ARBITRAIRE_DEUX):
                if points_dappui.get(a_rajouter[i] + j) != None:
                    liste_x.append(points_dappui[a_rajouter[i] + j][0])
                    liste_y.append(points_dappui[a_rajouter[i] + j][1])
                if points_dappui.get(a_rajouter[i] - j) != None:
                    liste_x.append(points_dappui[a_rajouter[i] + j][0])
                    liste_y.append(points_dappui[a_rajouter[i] + j][1])
        # Ici pas besoin que les points soient rangés dans l'ordre croissant
        # des 'x' au sein de "liste_x" et de "liste_y", mais juste que les 'x' et
        # les 'y' soient bien alignées (même indice dans leur liste respective
        # s'ils se correspondent)
        ordre = 0
        if len(points_dappui) >= 6:
            # Car ordre 5 maximal pour le moment dans le code de la fonction interpol
            ordre = 5
        elif len(points_dappui) >= 2:
            ordre = len(points_dappui) - 1
        else:
            pass

        coeffs_poly = obj.interpol(liste_x, liste_y, ordre)

        for frame_manquante in a_rajouter:
            x = obj.coor_suivi[frame_manquante-1][0] + FORWARD_UNIT
            # y = obj.coor_suivi[frame_manquante-1][1] + ...
            y = 0
            for degree in range(ordre+1):
                y += coeffs_poly[degree] * pow(x, ordre - degree)
            obj.addCoorSuivi(frame_manquante, (x, y))

            # EN PRINCIPE A CE STADE TOUT LES OBJETS ONT UNE TRAJECTOIRE SANS TROU

cv2.destroyAllWindows()

for obj in objects.Object.LISTE_OBJECT:
    print(obj.id)
    obj.showTraj()
