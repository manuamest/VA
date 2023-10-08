import cv2 as cv
import numpy as np

# Carga la imagen en blanco y negro
imagen = cv.imread('imgp1/miguel.jpg', cv.IMREAD_GRAYSCALE)

# Verifica que la imagen se haya cargado correctamente
if imagen is None:
    print("No se pudo cargar la imagen.")
else:
    # Convierte la imagen a punto flotante (tipo float32)
    imagen_normalizada = imagen.astype(np.float32) / 255.0

    # Ahora 'imagen_normalizada' contiene la imagen normalizada con valores en [0, 1]

    # Guarda la imagen normalizada en punto flotante
    cv.imwrite('imgp1/imagen_normalizada.png', (imagen_normalizada * 255).astype(np.uint8))

    # Puedes mostrar la imagen normalizada si lo deseas
    cv.imshow('Imagen Normalizada', imagen_normalizada)
    cv.waitKey(0)
    cv.destroyAllWindows()
