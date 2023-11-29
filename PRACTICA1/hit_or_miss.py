import cv2
import numpy as np
from morph import erode

def hit_or_miss(inImage, objSE, bgSE, center=None):
    # Comprobamos si los elementos estructurantes son incoherentes
    if np.any(np.logical_and(objSE, bgSE)):
        return "Error: elementos estructurantes incoherentes"

    # Si no se proporciona un centro, lo calculamos
    if center is None:
        center = (objSE.shape[0] // 2, objSE.shape[1] // 2)

    # Aplicamos la erosión a la imagen de entrada con el elemento estructurante del objeto
    erosionObj = erode(inImage, objSE, center)
    # Aplicamos la erosión al complemento de la imagen de entrada con el elemento estructurante del fondo
    erosionBg = erode((1 - inImage), bgSE, center)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # Intersección de las dos erosiones
    outImage = cv2.bitwise_and(erosionObj, erosionBg)

    return outImage

def run_hitormiss(inImage):

    #_, binary_image = cv2.threshold(inImage, 0.5, 1, cv2.THRESH_BINARY)

    objSE =  np.array([[0, 1, 0], 
                       [1, 1, 1],
                       [0, 1, 0]], dtype=np.uint8)

    bgSE = np.array([[1, 0, 1],
                     [0, 0, 0],
                     [1, 0, 1]], dtype=np.uint8)

    #inImage = np.array([[0, 0, 0, 0, 0, 0, 0],
    #                            [0, 0, 0, 1, 1, 0, 0],
    #                            [0, 0, 1, 1, 1, 1, 0],
    #                            [0, 0, 1, 1, 1, 1, 0],
    #                            [0, 0, 0, 1, 1, 0, 0],
    #                            [0, 0, 0, 1, 0, 0, 0],
    #                            [0, 0, 0, 0, 0, 0, 0]])

    _, binary_image = cv2.threshold(inImage, 0.5, 1, cv2.THRESH_BINARY)

    # Llamamos a la función hit_or_miss
    output_image = hit_or_miss(binary_image, objSE, bgSE)

    #print(inImage)
    #print(output_image)

    # Guardamos la imagen de salida
    cv2.imwrite('resultados/hitormiss.png', (output_image * 255).astype(np.float32))

    # Mostrar la imagen original y la filtrada
    cv2.imshow("Original Image", inImage)
    cv2.imshow("HitOrMiss Image", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
