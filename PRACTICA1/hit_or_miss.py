import cv2
import numpy as np

def dilate(inImage, SE, center=None):
    # Si el centro no se proporciona, se calcula
    if center is None:
        center = [SE.shape[0]//2, SE.shape[1]//2]

    # Crear una imagen de salida del mismo tamaño que la imagen de entrada
    outImage = np.zeros_like(inImage)

    #extend con np.pad

    # Recorrer cada píxel de la imagen
    for i in range(center[0], inImage.shape[0]-center[0]):
        for j in range(center[1], inImage.shape[1]-center[1]):
            # Aplicar el elemento estructurante al píxel y su vecindario
            neighborhood = inImage[i-center[0]:i+(SE.shape[0]-center[0]), j-center[1]:j+(SE.shape[1]-center[1])]
            # Si cualquier píxel bajo el elemento estructurante es 1, el píxel se mantiene
            if (neighborhood[SE==1] == 1).any():
                outImage[i, j] = 1


    return outImage


def erode(inImage, SE, center=None):
    # Si el centro no se proporciona, se calcula
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
    erosionBg = erode(np.abs(1 - inImage), bgSE, center)

    # Intersección de las dos erosiones
    print(erosionBg.dtype)
    print(erosionObj.dtype)
    outImage = cv2.bitwise_and(erosionObj, erosionBg)

    return outImage

def run_hitormiss(inImage):

    #_, binary_image = cv2.threshold(inImage, 0.5, 1, cv2.THRESH_BINARY)

    objSE =  np.array([[0, 0, 0], 
                    [1, 1, 0],
                    [0, 1, 0]], dtype=np.uint8)

    bgSE = np.array([[0, 1, 1],
                    [0, 0, 1],
                    [0, 0, 0]], dtype=np.uint8)

    inImage = np.array([[0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 1, 0, 0],
                                [0, 0, 1, 1, 1, 1, 0],
                                [0, 0, 1, 1, 1, 1, 0],
                                [0, 0, 0, 1, 1, 0, 0],
                                [0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0]])

    # Llamamos a la función hit_or_miss
    output_image = hit_or_miss(inImage, objSE, bgSE)

    print(inImageDiaposHoM)
    print(output_image)

    # Guardamos la imagen de salida
    #cv2.imwrite('resultados/hitormiss.png', (output_image * 255).astype(np.float32))

    # Mostrar la imagen original y la filtrada
    #cv2.imshow("Original Image", inImage)
    #cv2.imshow("HitOrMiss Image", output_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
