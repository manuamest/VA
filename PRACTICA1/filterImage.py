import cv2
import numpy as np

def filterImage(inImage, kernel):
    # Obtener las dimensiones de la imagen de entrada y el kernel
    rows, cols = inImage.shape
    kRows, kCols = kernel.shape

    # Calcular el desplazamiento necesario para centrar el kernel
    dRow = kRows // 2
    dCol = kCols // 2

    # Crear una imagen de salida inicializada a ceros
    outImage = np.zeros_like(inImage, dtype=np.float32)

    # Convolución
    for i in range(dRow, rows - dRow):
        for j in range(dCol, cols - dCol):
            # Extraer la región de interés de la imagen de entrada
            roi = inImage[i - dRow:i + dRow + 1, j - dCol:j + dCol + 1]

            # Aplicar la convolución entre el kernel y la región de interés
            conv_result = np.sum(roi * kernel)

            # Asignar el resultado a la posición correspondiente en la imagen de salida
            outImage[i, j] = conv_result

    return outImage

# Ejemplo de uso
if __name__ == "__main__":
    # Cargar una imagen
    image = cv2.imread("imgp1/imagen_normalizada.png", cv2.IMREAD_GRAYSCALE) / 255.0

    # Definir un kernel de ejemplo (por ejemplo, un filtro de promedio)
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]]) / 9.0  # Normalización para el filtro de promedio

    # Aplicar el filtro a la imagen
    output_image = filterImage(image, kernel)

    cv2.imwrite('imgp1/filterImage.jpg', (output_image * 255).astype(np.uint8))

    # Mostrar la imagen original y la filtrada
    cv2.imshow("Original Image", image)
    cv2.imshow("Filtered Image", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
