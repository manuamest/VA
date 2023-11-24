import cv2
import numpy as np
import matplotlib

def medianFilter(inImage, filterSize):
    # Obtener las dimensiones de la imagen de entrada
    rows, cols = inImage.shape

    # Mitad del tamaño del filtro (radio)
    radius = filterSize // 2 

    # Crear una imagen de salida inicializada a ceros
    outImage = np.zeros_like(inImage)

    # Iterar sobre la imagen
    for i in range(rows):
        for j in range(cols):
            # Definir los límites de la ventana
            row_min = max(0, i - radius)
            row_max = min(rows, i + radius + 1)
            col_min = max(0, j - radius)
            col_max = min(cols, j + radius + 1)

            # Extraer la región de interés (ventana)
            window = inImage[row_min:row_max, col_min:col_max]

            # Calcular la mediana de la ventana y asignarla al píxel de salida
            outImage[i, j] = np.median(window)

    return outImage

# Ejemplo de uso
if __name__ == "__main__":
    # Cargar una imagen en escala de grises (asegúrate de que está en [0, 255])
    image = cv2.imread("imgp1/espacio.jpg", cv2.IMREAD_GRAYSCALE) / 255.0

    # Tamaño del filtro de mediana (por ejemplo, 3x3)
    filterSize = 7

    # Aplicar el filtro de medianas
    output_image = medianFilter(image, filterSize)

    # Mostrar la imagen original y la filtrada
    cv2.imshow("Original Image", image)
    cv2.imshow("Median Filtered Image", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
