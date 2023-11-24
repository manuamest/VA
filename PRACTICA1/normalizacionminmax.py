import cv2
import numpy as np

def normalizar_imagen(inImage):

    # Convertir la imagen a escala de grises (si es a color)
    imagen_gris = cv2.cvtColor(inImage, cv2.COLOR_BGR2GRAY)

    # Obtener los valores mínimo y máximo de la imagen
    min_valor = np.min(imagen_gris)
    max_valor = np.max(imagen_gris)

    # Normalizar la imagen entre 0 y 255
    imagen_normalizada = ((imagen_gris - min_valor) / (max_valor - min_valor)) * 255

    # Convertir la imagen normalizada de nuevo a tipo entero
    imagen_normalizada = imagen_normalizada.astype(np.uint8)
    
    return imagen_normalizada
