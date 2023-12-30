# Importar bibliotecas
import cv2
import os
import numpy as np

def players1(inImage):

    image = inImage.copy()
        
    # Convertir la imagen a HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Definir rangos de colores
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])

    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    lower_red = np.array([0, 31, 255])
    upper_red = np.array([176, 255, 255])

    lower_white = np.array([0, 0, 0])
    upper_white = np.array([0, 0, 255])

    # Crear una máscara
    mask = cv2.inRange(hsv, lower_green, upper_green)
    res = cv2.bitwise_and(image, image, mask=mask)

    # Convertir la imagen a escala de grises
    res_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    # Definir un kernel para realizar operaciones morfológicas en la imagen de umbral
    kernel = np.ones((13, 13), np.uint8)
    thresh = cv2.threshold(res_gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Encontrar contornos en la imagen de umbral
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    prev = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    idx = 0

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # Detectar jugadores
        if (h >= (1.5) * w):
            if (w > 15 and h >= 15):
                idx = idx + 1
                player_img = image[y:y + h, x:x + w]
                player_hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)

                mask1 = cv2.inRange(player_hsv, lower_blue, upper_blue)
                res1 = cv2.bitwise_and(player_img, player_img, mask=mask1)
                res1 = cv2.cvtColor(res1, cv2.COLOR_HSV2BGR)
                res1 = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
                nzCount = cv2.countNonZero(res1)

                mask2 = cv2.inRange(player_hsv, lower_red, upper_red)
                res2 = cv2.bitwise_and(player_img, player_img, mask=mask2)
                res2 = cv2.cvtColor(res2, cv2.COLOR_HSV2BGR)
                res2 = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
                nzCountred = cv2.countNonZero(res2)

                if (nzCount >= 20):
                    cv2.putText(image, 'France', (x - 2, y - 2), font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
                else:
                    pass
                if (nzCountred >= 20):
                    cv2.putText(image, 'Belgium', (x - 2, y - 2), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
                else:
                    pass
        if ((h >= 1 and w >= 1) and (h <= 30 and w <= 30)):
            player_img = image[y:y + h, x:x + w]

            player_hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(player_hsv, lower_white, upper_white)
            res1 = cv2.bitwise_and(player_img, player_img, mask=mask1)
            res1 = cv2.cvtColor(res1, cv2.COLOR_HSV2BGR)
            res1 = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
            nzCount = cv2.countNonZero(res1)

            if (nzCount >= 3):
                cv2.putText(image, 'football', (x - 2, y - 2), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
    return image

def run_players1(inImage):
    image = players1(inImage)

    # Mostrar la imagen con las detecciones y la original
    cv2.imshow('Original', inImage)
    cv2.imshow('Player Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
