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

capture = cv2.VideoCapture("example.MOV")

frame_count = 0

while True:
    ok, frame_UN = capture.read()
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
                                        output_image_name="image_travail_2.jpg"
                                        )

    box_coor = masks['rois']

    frame_DEUX = cv2.imread("image_travail_2.jpg")
    cv2.namedWindow("apercu_travail", cv2.WINDOW_NORMAL)
    cv2.imshow("apercu_travail", frame_DEUX)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

    ### Beginning of the attributions
    dico_keeper = {}  # 0 for non-attributed, 1 for attributed
    # To keep track of the attributions, because no two objects can relate to the same detected bbox
    for indice_bbox in range(box_coor.shape[0]):  ### For every box from each frame
        dico_keeper[indice_bbox] = 0

        x = int((box_coor[indice_bbox, 1] + box_coor[indice_bbox, 3]) / 2)
        y = int((box_coor[indice_bbox, 0] + box_coor[indice_bbox, 2]) / 2)
        if len(objects.Object.LISTE_OBJECT) == 0:
            objects.Object(frame_count,
                           (rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255)),
                           (x, y))

        # # The normalizers can differ from the objects.Object.interstice

        if dico_keeper[indice_bbox] == 1:
            pass
        else:
            #####
            last_poses = [obj.mostRecentPos for obj in objects.Object.LISTE_OBJECT]
            normalizers = [frame_count - obj.mostRecentPosId for obj in objects.Object.LISTE_OBJECT]

            ind_nearest = 0
            dist_nearest = int(pow(PERIM * PERIM_MULT, 2))
            i = 0
            for (x2, y2) in last_poses:
                if normalizers[i] == 0:
                    normalizers[i] = 1
                if int((pow(x - x2, 2) + pow(y - y2, 2)) / pow(normalizers[i], 2)) <= pow(PERIM, 2):
                    dist = int((pow(x - x2, 2) + pow(y - y2, 2)) / pow(normalizers[i], 2))
                    if dist < dist_nearest:
                        dist_nearest = dist
                        ind_nearest = i
                    i += 1
                else:
                    pass
                
            if dist_nearest <= pow(PERIM, 2): # For good measure
                objects.Object.LISTE_OBJECT[ind_nearest].addCoorSuivi(frame_count, (x, y))
            else:
                objects.Object(frame_count,
                               (rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255)),
                               (x, y))
            dico_keeper[indice_bbox] = 1
        # To avoid any paradox
        # Once a bbox from a frame gets attributed, no other Object can identify to it.

        if dico_keeper[indice_bbox] == 0:  # Bbox too far from every other Object
            objects.Object(frame_count,
                           (rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255)),
                           (x, y))
            dico_keeper[indice_bbox] = 1
    ### End of the attributions

    print("Numero de la frame traitée :", frame_count)
    if objects.Object.LISTE_OBJECT != None:
        print("Nombres d'objets alors détectés :", len(objects.Object.LISTE_OBJECT), "\n")

capture.release()
cv2.destroyAllWindows()

###########################

# Now let's fill up the holes : If there's a hole at index i, we check on both sides till indices i-k and i+k, to find our support points.
a_rajouter = []
points_dappui = {}
coeffs_poly = np.empty
for obj in objects.Object.LISTE_OBJECT:
    if not obj.verif():
        a_rajouter = obj.holes()
        if len(a_rajouter) >= 3:  # (If less than three holes, no problem anyways)
            for i in range(len(a_rajouter) - 1):
                if a_rajouter[i + 1] - a_rajouter[i] > 1:  # If we're detecting a border
                    points_dappui[a_rajouter[i] + 1] = obj.coor_suivi[a_rajouter[i] + 1]
        # Let's now consider 2 to 6 support points in the neiborhood of our holes.
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
        # Here no need to arrange the xs and ys, only that the xs and ys stay aligned (same index if corresponding)
        ordre = 0
        if len(points_dappui) >= 6:
            # Because 5th order maximum in the interpol function from the Object class
            ordre = 5
        elif len(points_dappui) >= 2:
            ordre = len(points_dappui) - 1
        else:
            pass

        coeffs_poly = obj.interpol(liste_x, liste_y, ordre)

        for frame_manquante in a_rajouter:
            x = obj.coor_suivi[frame_manquante-1][0] + FORWARD_UNIT
            y = 0
            for degree in range(ordre+1):
                y += coeffs_poly[degree] * pow(x, ordre - degree)
            obj.addCoorSuivi(frame_manquante, (x, y))

            # At this point we should be left with full tracks with no holes

cv2.destroyAllWindows()

for obj in objects.Object.LISTE_OBJECT:
    print(obj.id)
    obj.showTraj()
