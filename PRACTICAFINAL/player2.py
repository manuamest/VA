# Importar bibliotecas
import cv2
import numpy as np
from whitelines import white_pixels_image

#REVISAR UMBRAL Y CONTORNO PELOTA

def players2(inImage):

    image = inImage.copy()
    image2 = inImage.copy()
    # Convertir la imagen a HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Definir el rango de colores para los jugadores
    lower_green = np.array([20, 20, 20])
    upper_green = np.array([60, 255, 255])

    # Crear una máscara
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_not_green = cv2.bitwise_not(mask)

    res = cv2.bitwise_and(image, image, mask=mask)
    #res = cv2.bitwise_and(image, image, mask=mask_not_green)

    # Convertir la imagen a escala de grises
    res_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('MASK', mask)
    # cv2.imshow('RES', res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Definir un kernel para realizar operaciones morfológicas en la imagen de umbral
    kernel = np.ones((13, 13), np.uint8)
    thresh = cv2.threshold(res_gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Encontrar contornos en la imagen de umbral
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # Detectar jugadores
        if h > w and w >= 8 and h > 8 and w < 100 and h < 100:

            # Dibujar un rectángulo alrededor del jugador
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
        else:
             if h >= (1) * w and w < 10 and h < 10:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
        
    return image

def run_players2(inImage):

    image = players2(inImage)

    # Mostrar la imagen con las detecciones
    cv2.imshow("Original", inImage)
    cv2.imshow('Player Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    inImage = cv2.imread("SOLOCAMPO.jpg")
    
    resultado = players2(inImage)

    # Mostrar la imagen con las detecciones
    cv2.imshow("Original", inImage)
    cv2.imshow('Player Detection', resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()