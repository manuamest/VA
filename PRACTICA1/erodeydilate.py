import numpy as np
import cv2

def extendImageDuplicate(inImage, arriba, abajo, derecha, izquierda):
    pad_width = ((arriba, abajo), (izquierda, derecha))
    outImage = np.pad(inImage, pad_width= pad_width, mode='edge')
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

def dilate(inImage, SE, center=None):
    # Si el centro no se proporciona, se calcula
    if center is None:
        center = [SE.shape[0]//2, SE.shape[1]//2]

    # Crear una imagen de salida del mismo tamaño que la imagen de entrada
    outImage = np.zeros_like(inImage)

    extendImageDuplicate(inImage, 1, 1, 1, 1)

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
if __name__ == "__main__":
    # Cargar una imagen en escala de grises (asegúrate de que está en [0, 255])
    image = cv2.imread("imgp1/ORIGINALERODE.png", cv2.IMREAD_GRAYSCALE) / 255
    _, binary_image = cv2.threshold(image, 0.5, 1, cv2.THRESH_BINARY)


    SE = np.ones((13, 13), dtype=np.uint8)

    # Aplicar el filtro de medianas
    output_image = dilate(binary_image, SE)

    cv2.imwrite('imgp1/erode.jpg', (output_image * 255).astype(np.uint8))

    # Mostrar la imagen original y la filtrada
    cv2.imshow("Original Image", image)
    cv2.imshow("Dilate Image", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
