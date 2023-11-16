import cv2
import numpy as np

def gaussKernel1D(sigma):
    # Calcular N a partir de σ
    N = int(2 * np.ceil(3 * sigma) + 1)

    # Calcular el centro del kernel
    center = N // 2

    # Inicializar el kernel como un vector de ceros
    kernel = np.zeros(N)

    # Calcular el valor del kernel Gaussiano
    for i in range(N):
        x = i - center
        kernel[i] = np.exp(-0.5 * (x / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

    # Normalizar el kernel para que la suma sea igual a 1
    kernel /= np.sum(kernel)

    return kernel

def gaussianFilter(inImage, sigma):
    # Calcular los kernels unidimensionales
    kernel1D = gaussKernel1D(sigma)

    # Aplicar la convolución horizontal (1D)
    filtered_horizontal = np.apply_along_axis(lambda x: np.convolve(x, kernel1D, mode='same'), axis=1, arr=inImage)

    # Aplicar la convolución vertical (1D) al resultado de la convolución horizontal
    filtered_image = np.apply_along_axis(lambda x: np.convolve(x, kernel1D, mode='same'), axis=0, arr=filtered_horizontal)

    return filtered_image

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

def LoG(inImage, sigma):

    laplace = np.array([[0, -1, 0],
                        [-1, 4, -1],
                        [0, -1, 0]], dtype=np.float32)
    
    gaussianImage = gaussianFilter(inImage, sigma)

    outImage = filterImage(gaussianImage, laplace)

    return outImage

# Ejemplo de uso
if __name__ == "__main__":
    # Cargar una imagen
    image = cv2.imread("imgp1/chica4k.png", cv2.IMREAD_GRAYSCALE) / 255
    #_, binary_image = cv2.threshold(image, 0.5, 1, cv2.THRESH_BINARY)

    sigma = 1.5

    # Aplicar el filtro a la imagen
    output_image = LoG(image, sigma)

    cv2.imwrite('imgp1/LoGImage.jpg', (output_image * 255).astype(np.float32))

    # Mostrar la imagen original y la filtrada
    cv2.imshow("Original Image", image)
    cv2.imshow("LoG Image", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()