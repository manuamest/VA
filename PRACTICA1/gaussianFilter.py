import numpy as np
import cv2

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

# Ejemplo de uso
if __name__ == "__main__":
    # Parámetro σ
    sigma = 1.5

    # Generar una imagen de ejemplo (reemplaza esta línea con tu imagen real)
    input_image = cv2.imread('imgp1/imagen_normalizada.png', cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    # Aplica el filtro gaussiano
    output_image = gaussianFilter(input_image, sigma)

    cv2.imwrite('imgp1/gaussianFilter.jpg', (output_image * 255).astype(np.uint8))

    # Muestra la imagen de entrada y la imagen de salida
    cv2.imshow('Imagen de Entrada', input_image)
    cv2.imshow('Imagen de Salida', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()