import cv2
import numpy as np

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
        center = (objSE.shape[0] // 2 + 1, objSE.shape[1] // 2 + 1)

    # Aplicamos la erosión a la imagen de entrada con el elemento estructurante del objeto
    erosionObj = erode(inImage, objSE, center)

    # Aplicamos la erosión al complemento de la imagen de entrada con el elemento estructurante del fondo
    erosionBg = erode(1 - inImage, bgSE, center)

    # La operación hit-or-miss es la intersección de las dos erosiones
    outImage = cv2.bitwise_and(erosionObj, erosionBg)

    # erosionObj && erosionBg

    return outImage

# Cargamos la imagen de entrada
image = cv2.imread('imgp1/HITORMISSORIGINAL.png', cv2.IMREAD_GRAYSCALE)/ 255
_, binary_image = cv2.threshold(image, 0.5, 1, cv2.THRESH_BINARY)

# Definimos los elementos estructurantes para el objeto y el fondo
#objSE = np.ones((3,3), dtype=np.uint8)
#bgSE = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]], dtype=np.uint8)


# Definimos los elementos estructurantes para el objeto y el fondo
#objSE = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
#bgSE = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]], dtype=np.uint8)
# Crear una matriz de 11x11 con el centro como 1 y el resto como 0s

# Crear una matriz de 11x11 con el centro y sus vecinos como 0 y el resto como 1s
objSE = np.ones((11, 11), dtype=np.uint8)
objSE[4:7, 4:7] = 0

# Crear una matriz de 11x11 con el centro y sus vecinos como 1 y el resto como 0s
bgSE = np.zeros((11, 11), dtype=np.uint8)
bgSE[4:7, 4:7] = 1

kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

#binary_image = (binary_image * 255).astype(np.uint8)
#output_image = cv2.morphologyEx(binary_image, cv2.MORPH_HITMISS, kernel)

objSE = np.array([[0,0,0,0,0,0,0],
                 [0,0,0,1,1,0,0],
                 [0,0,1,1,1,1,0],
                 [0,0,1,1,1,1,0],
                 [0,0,0,1,1,0,0],
                 [0,0,0,1,0,0,0],
                 [0,0,0,0,0,0,0]], dtype=np.uint8)

bgSE =  np.array([[1,1,1,1,1,1,1],
                 [1,1,1,0,0,1,1],
                 [1,1,0,0,0,0,1],
                 [1,1,0,0,0,0,1],
                 [1,1,1,0,0,1,1],
                 [1,1,1,0,1,1,1],
                 [1,1,1,1,1,1,1]], dtype=np.uint8)

bgSE =  np.array([[0, 0, 0], 
                  [1, 1, 0],
                  [0, 1, 0]], dtype=np.uint8)

objSE = np.array([[0, 1, 1],
                  [0, 0, 1],
                  [0, 0, 0]], dtype=np.uint8)


# Llamamos a la función hit_or_miss
output_image = hit_or_miss(binary_image, objSE, bgSE)

SE = np.ones((13, 13), dtype=np.uint8)


#output_image = erode(binary_image, bgSE)

# Guardamos la imagen de salida
cv2.imwrite('imgp1/hitormiss.png', output_image)

# Mostrar la imagen original y la filtrada
cv2.imshow("Original Image", image)
cv2.imshow("HitOrMiss Image", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
