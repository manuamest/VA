import numpy as np
import cv2

def extend(inImage, arriba, abajo, derecha, izquierda):
    # Duplicar en vertical y en horizontal
    duplicated_vertical = np.tile(inImage, (1 + arriba + abajo, 1))
    duplicated_image = np.tile(duplicated_vertical, (1, 1 + izquierda + derecha))

    return duplicated_image


def erode(inImage, SE, center=None):

    # Calculo centro
    if center is None:
        center = [SE.shape[0]//2, SE.shape[1]//2]

    # Crear una imagen de salida del mismo tamaño que la imagen de entrada
    outImage = np.zeros_like(inImage)

    # Recorrer cada píxel de la imagen
    for i in range(center[0], inImage.shape[0]-center[0]):
        for j in range(center[1], inImage.shape[1]-center[1]):
            # Aplicar el elemento estructurante al píxel y su vecindario
            neighborhood = inImage[i-center[0]:i+(SE.shape[0]-center[0]), j-center[1]:j+(SE.shape[1]-center[1])]
            # Si todos los píxeles bajo el elemento estructurante son 1, el píxel se mantiene
            if (neighborhood[SE==1] == 1).all():
                outImage[i, j] = 1

    return outImage

def dilate(inImage, SE, center=None):
    
    # Calculo centro
    if center is None:
        center = [SE.shape[0]//2, SE.shape[1]//2]

    # Crear una imagen de salida del mismo tamaño que la imagen de entrada
    outImage = np.zeros_like(inImage)

    # Parametros para la extension
    arriba = max(0, center[0] - 1)
    abajo = max(0, SE.shape[0] - center[0] - 1)
    izquierda = max(0, center[1] - 1)
    derecha = max(0, SE.shape[1] - center[1] - 1)

    # Extender la imagen
    extendedImage = extend(inImage, arriba, abajo, derecha, izquierda)


    # Recorrer cada píxel de la imagen
    for i in range(center[0], inImage.shape[0]-center[0]):
        for j in range(center[1], inImage.shape[1]-center[1]):
            # Aplicar el elemento estructurante al píxel y su vecindario
            neighborhood = inImage[i-center[0]:i+(SE.shape[0]-center[0]), j-center[1]:j+(SE.shape[1]-center[1])]
            # Si cualquier píxel bajo el elemento estructurante es 1, el píxel se mantiene
            if (neighborhood[SE==1] == 1).any():
                outImage[i, j] = 1


    return outImage


def opening(inImage, SE, center=None):
    return dilate(erode(inImage, SE, center), SE, center)

def closing(inImage, SE, center=None):
    return erode(dilate(inImage, SE, center), SE, center)

# Ejemplo de uso
def run_morph(inImage, op):

    SE = np.ones((1, 1), dtype=np.uint8)

    # Aplicar el operador morfologico
    if op == 'Erode':
        output_image = erode(inImage, SE)
    elif op == 'Dilate':
        output_image = dilate(inImage, SE)
    elif op == 'Opening':
        output_image = opening(inImage, SE)
    elif op == 'Closing':
        output_image = closing(inImage, SE)

    cv2.imwrite('resultados/' + op + '.jpg', (output_image * 255).astype(np.float32))

    # Mostrar la imagen original y la filtrada
    cv2.imshow("Original Image", inImage)
    cv2.imshow( op + " Image", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
