import cv2 as cv
import numpy as np

# Cargar las dos imágenes que deseas mostrar
imagen1 = cv.imread('imgp1/imagen_normalizada.png')
imagen2 = cv.imread('imgp1/adjustIntensity.jpg')

# Verificar que las imágenes se hayan cargado correctamente
if imagen1 is None or imagen2 is None:
    print("No se pudieron cargar ambas imágenes.")
else:
    # Concatenar las dos imágenes horizontalmente
    imagen_concatenada = cv.hconcat([imagen1, imagen2])

    # Mostrar la imagen concatenada en una ventana
    cv.imshow('Imagen Concatenada', imagen_concatenada)
    cv.waitKey(0)
    cv.destroyAllWindows()
